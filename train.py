#!/usr/bin/env python3
"""
Training loop for pitch and yaw prediction using labeled video inputs. Run like:

$ python3 ./train.py --labeled labeldir --output trained.pyt [--batch_size 2 --epochs 20 --lr 1e-3]
"""
import os
import torch
import logging
import argparse

from torch.utils.tensorboard import SummaryWriter

import model

logging.basicConfig(level=logging.INFO)

def load_data(inputpath):
    logging.info(f"Looking for video (*.hevc) and corresponding "
        + f"label files (*.txt) under {inputpath}..")
    hevc_files = [
        os.path.join(inputpath, f)
        for f in os.listdir(inputpath)
        if f.endswith('.hevc')
    ]
    logging.info(f"Found {len(hevc_files)} video files, loading..")
    return [
        model.PitchYawVideoDataset(filename[:-5])
        for filename in hevc_files
    ]

def train(loader, net, device, epochs, initial_lr):
    writer = SummaryWriter()
    net.to(device)
    logging.info("========= Starting train loop =========")
    for epoch in range(0,epochs):
        running_loss = 0.0
        criterion = torch.nn.MSELoss()
        learning_rate = initial_lr     * (epoch < epochs/2) \
                      + initial_lr/100 * (epochs/2 <= epoch)
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            weight_decay=learning_rate/10,
            momentum=0.9,
            nesterov=True
        )
        for i, data in enumerate(loader, 0):
            frames, labels = data
            optimizer.zero_grad()
            inputs = torch.tensor(frames, requires_grad=True).to(device)
            targets = torch.tensor(labels, requires_grad=True).to(device)
            predictions = net(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), epoch)
        logging.info(f"[{epoch+1:2d}/{epochs}] " \
            + f"learning_rate = {learning_rate:.8f}, " \
            + f"avg_loss = {(running_loss / loader.length):.16f}")
    logging.info("========= Training loop ended =========")
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
    arg_parser.add_argument('--epochs', type=int, default=20)
    arg_parser.add_argument(
        '--lr',
        help='learning rate',
        type=float,
        default=1e-3
    )
    args = arg_parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = load_data(args.labeled)
    loader = model.PitchYawVideoChainLoader(
        datasets,
        batch_size=args.batch_size,
        shuffle=True
    )
    net = model.DaNet(args.batch_size)
    logging.info("Model capacity is " +
        f"{sum(p.numel() for p in net.parameters() if p.requires_grad)}")
    net = train(loader, net, device, args.epochs, args.lr)
    logging.info(f"Saving trained model to {args.output}")
    torch.save(net.state_dict(), args.output)
