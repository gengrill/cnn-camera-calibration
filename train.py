#!/usr/bin/env python3
"""
Training loop for pitch and yaw prediction using labeled video inputs. Run like:

$ python3 ./train.py --labeled labeldir --output model-dir --feat_idx [0|1] [--batch_size 20 --epochs 100 --dimension 128 --lr 1e-3 --finetune True --input checkpoint-file.pyt]
"""
import os
import time
import torch
import logging
import argparse
import numpy as np
import torchvision as tv

from apex import amp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# local imports
import model
from opencv_pitchyaw import K, CENTER

logging.basicConfig(level=logging.INFO)

PITCH      = 0
YAW        = 1
TEST_SPLIT = 0.2
VAL_SPLIT  = 0.2
NO_CLASSES = 100 # pixel resolution of the VP in either x or y direction
BASE       = 'efficientnet_b7'

# globals for saving state
curr_epoch  = 0
curr_valacc = 0.0

def load_model(checkpoint=None, device='cpu'):
     net  = tv.models.efficientnet_b7(pretrained=True)
     net.classifier = torch.nn.Sequential(
         torch.nn.Dropout(p=net.classifier[0].p, inplace=True),
         torch.nn.Linear(net.classifier[1].in_features, NO_CLASSES))
     if checkpoint is not None:
         logging.info(f"Loading weights from {checkpoint}.")
         checkpoint = torch.load(checkpoint, map_location=device)
         net.load_state_dict(checkpoint)
     return net.to(device)

def load_data(inputpath, dimension):
    logging.info(f"Looking for video (*.hevc) and corresponding "
                 + f"label files (*.txt) under {inputpath}..")
    hevc_files = [
        os.path.join(inputpath, f)
        for f in os.listdir(inputpath)
        if f.endswith('.hevc')
    ]
    logging.info(f"Found {len(hevc_files)} video files, loading..")
    return [
        model.PitchYawVideoDataset(
            filename=filename[:-5],
            dimension=dimension,
            target_transform=lambda x : py_to_vp(*x)
        )
        for filename in hevc_files
    ]

def get_sample_weights(dataset, feat_idx, no_classes):
    sample_classes  = [label_to_class(labels.unsqueeze(0), feat_idx)[0].item() for frame, labels in dataset]
    class_counts    = [sample_classes.count(c) for c in range(no_classes)]
    sample_counts   = [class_counts[s] for s in sample_classes]
    sample_weights  = [1/sample_counts[s] if sample_counts[s]!=0 else 0 for s in range(len(dataset))]
    return torch.tensor(sample_weights)/no_classes

def label_to_class(labels, feat_idx, no_classes=NO_CLASSES, center=CENTER):
    return torch.clip(labels[:, feat_idx]-center[feat_idx], -no_classes//2, no_classes//2 - 1) + no_classes//2

def py_to_vp(pitch, yaw, K=K):
    vp_cam_x = np.tan(yaw)
    vp_cam_y = -np.tan(pitch / np.cos(yaw))
    return np.round(K.dot(np.array([vp_cam_x, vp_cam_y, 1]))[:2]).astype(int)

def step(loader, device, writer, feat_idx, optimizer, net, trans, criterion, batch_size, epoch, epochs, logtag, istraining):
    running_loss = torch.zeros(len(loader))
    running_err  = torch.zeros(len(loader))
    running_acc  = torch.zeros(len(loader))
    writertag    = "validation" if epochs == -1 else "train"*istraining+"test"*(not istraining)
    for i, data in enumerate(tqdm(loader)):
        frames, labels = data
        inputs  = trans(frames.to(device))
        targets = label_to_class(labels, feat_idx).to(device)
        preds = net(inputs)
        loss  = criterion(preds, targets)
        if istraining:
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        err = (preds.argmax(1)-targets).abs().sum().item()/batch_size
        acc = (preds.argmax(1)==targets).sum().item()/batch_size
        running_loss[i] = loss.item() * inputs.size(0)
        running_err[i]  = err
        running_acc[i]  = acc
        writer.add_scalar("Loss/"+writertag, loss.item(), i+epoch*len(loader))
        writer.add_scalar("Error/"+writertag,  err, i+epoch*len(loader))
        writer.add_scalar("Accuracy/"+writertag, acc, i+epoch*len(loader))
    avg_loss = torch.mean(running_loss).item()
    avg_err  = torch.mean(running_err).item()
    avg_acc  = torch.mean(running_acc).item()
    writer.add_scalar("Mean Loss/"+writertag, avg_loss, epoch)
    writer.add_scalar("Mean Error/"+writertag, avg_err, epoch)
    writer.add_scalar("Mean Accuracy/"+writertag, avg_acc, epoch)
    logging.info(logtag \
        + f"avg_loss = {avg_loss:10.6f}, " \
        + f"avg_err = {avg_err:10.6f}, " \
        + f"avg_acc = {avg_acc:10.6f}")
    logging.info(logtag \
        + f"var_loss = {torch.std(running_loss):10.6f}, " \
        + f"var_err = {torch.std(running_err):10.6f}, " \
        + f"var_acc = {torch.std(running_acc):10.6f}")
    return avg_loss, avg_err, avg_acc

def train(trainset, net, device, writer, epochs, initial_lr, batch_size, feat_idx, testset=None, finetune=False):
    if finetune:
        loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    else:
        weights    = get_sample_weights(trainset, feat_idx, NO_CLASSES)
        unisampler = torch.utils.data.WeightedRandomSampler(weights, len(trainset))
        loader     = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=unisampler)
    optimizer  = torch.optim.SGD(
        net.parameters(),
        lr=initial_lr,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, eps=1e-16)
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net, optimizer = amp.initialize(net, optimizer, opt_level="O2", num_losses=1, loss_scale=512.0)
    logging.info("========= Starting train loop =========")
    logging.info(f"Training with {len(trainset)} samples used for training and {epochs} epochs.")
    if testset is not None:
        logging.info(f"Testing with {len(testset)} unseen samples after each epoch.")
    for epoch in range(0, epochs):
        traintrans    = tv.transforms.Compose([
            tv.transforms.RandomAutocontrast(p=0.5),
            tv.transforms.RandomAdjustSharpness(.75, p=0.5),
            tv.transforms.RandomInvert(p=0.5),
            tv.transforms.RandomSolarize(5, p=0.5),
            tv.transforms.RandomErasing(p=0.75, scale=(0.01, 0.02),),
            tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])# if not finetune else tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        running_time = time.time()
        avg_loss, avg_err, avg_acc = step(loader, device, writer, feat_idx, optimizer, net, traintrans, criterion, batch_size, epoch, epochs, f"[{epoch+1:2d}/{epochs}] ", True)
        running_time = time.time() - running_time
        scheduler.step(avg_loss)
        if testset is not None:
            test(testset, net, device, feat_idx, criterion, writer, epoch, epochs)
        logging.info(f"learning_rate = {optimizer.param_groups[0]['lr']:.2e}, " \
            + f"running_time = {running_time:.2f}s (avg={running_time / len(loader):.2f})")
        if avg_loss < 0.001 or optimizer.param_groups[0]['lr']<=1e-15:
            break
    logging.info("========= Training loop ended =========")
    return net

def test(testset, net, device, feat_idx, criterion, writer, epoch, epochs):
    global curr_epoch, curr_valacc
    net.eval()
    with torch.no_grad():
        loader     = torch.utils.data.DataLoader(testset)
        testtrans  = tv.transforms.Compose([
            tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        avg_loss, avg_error, avg_acc = step(loader, device, writer, feat_idx, None, net, testtrans, criterion, 1, epoch, epochs, '[ TEST ] ', False)
        curr_epoch = epoch
        curr_valacc = avg_acc
    net.train()

def validate(dataset, net, device, writer, feat_idx):
    logging.info("========= Starting validation =========")
    logging.info(f"with {len(dataset)} samples (not seen during training).")
    criterion    = torch.nn.CrossEntropyLoss()
    test(dataset, net, device, feat_idx, criterion, writer, 0, -1)
    logging.info("========= Validation ended =========")
    return net

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    arg_parser.add_argument(
        '--labeled',
        help='path to labeled file(s)',
        required=True,
        type=str
    )
    arg_parser.add_argument(
        '--output',
        help='where to save the trained pytorch model',
        required=True,
        type=str
    )
    arg_parser.add_argument('--batch_size', type=int, default=2)
    arg_parser.add_argument('--epochs',     type=int, default=20)
    arg_parser.add_argument('--dimension',  type=int, default=128)
    arg_parser.add_argument('--feat_idx', type=int, default=PITCH)
    arg_parser.add_argument('--finetune', type=bool, default=False)
    arg_parser.add_argument(
        '--input',
        help='which model to finetune',
        required=False,
        type=str
    )
    arg_parser.add_argument(
        '--lr',
        help='learning rate',
        type=float,
        default=1e-3
    )

    writer  = SummaryWriter()
    args    = arg_parser.parse_args()
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.finetune and args.input is None:
        exit('Error: Pretrained model checkpoint required for finetuning (--finetune=True but --input was missing).')
    net = load_model(checkpoint=args.input, device=device)
    dataset = torch.utils.data.ConcatDataset(load_data(args.labeled, (args.dimension,)*2))
    testsetsize = round(TEST_SPLIT * len(dataset))
    valsetsize  = round(VAL_SPLIT  * len(dataset))
    trainset, testset, validationset = torch.utils.data.random_split(
        dataset,
        [len(dataset)-testsetsize-valsetsize, testsetsize, valsetsize]
    )
    logging.info(f"Model capacity is {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
    try:
        net = train(trainset, net, device, writer, args.epochs, args.lr, batch_size=args.batch_size, feat_idx=args.feat_idx, testset=testset, finetune=args.finetune)
    except KeyboardInterrupt:
        pass
    finally:
        featstr = 'pitch' if args.feat_idx==PITCH else 'yaw'
        trainmode = 'finetuned' if args.finetune else 'pretrained'
        output = args.output+'/'+featstr+'_'+trainmode+'_'+BASE+'_epoch'+str(curr_epoch)+'_valacc'+str(curr_valacc)+'.pyt'
        logging.info(f"Saving model to {output}")
        torch.save(net.state_dict(), output)
        validate(validationset, net, device, writer, feat_idx=args.feat_idx)
