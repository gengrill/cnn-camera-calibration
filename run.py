#!/usr/bin/env python3
"""
Pitch and yaw inference with a trained model. Run like:

$ python3 ./run.py --pitch_model trained_pitch.pyt --yaw_model trained_yaw.pyt --input video.hevc --output p_and_y.txt --dimension 128
"""
import torch
import logging
import argparse
import numpy as np
import torchvision as tv

from tqdm import tqdm

# local imports
import model
from train import load_model, NO_CLASSES

logging.basicConfig(level=logging.INFO)

def inference(trained_net, dataset, device):
    loader      = torch.utils.data.DataLoader(dataset)
    predictions = []
    logging.info(f"Found {len(dataset)} frames, starting inference..")
    trans = tv.transforms.Compose([tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trained_net.eval()
    for i, data in enumerate(tqdm(loader)):
        frame, label = data
        frame = trans(frame.to(device))
        preds = trained_net(frame)
        predictions += [preds.argmax(1).item()]
    logging.info(f"Done. Predicted {len(predictions)} values.")
    return predictions

def classes_to_pys(pitch_classes, yaw_classes, no_classes=100):
    from opencv_pitchyaw import CENTER, rpy_from_vp
    coords = np.array([pitch_classes, yaw_classes]).T - no_classes//2 + CENTER
    rpys   = np.array([np.abs(rpy_from_vp(coord)) for coord in coords])
    return rpys[:,1], rpys[:,2]

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    arg_parser.add_argument(
        '--pitch_model',
        help='trained pytorch model',
        required=True,
        type=str
    )
    arg_parser.add_argument(
        '--yaw_model',
        help='trained pytorch model',
        required=True,
        type=str
    )
    arg_parser.add_argument(
        '--input',
        help='unlabeled hevc video',
        required=True,
        type=str
    )
    arg_parser.add_argument(
        '--output',
        help='output text file for pitch and yaw in numpy format',
        required=True,
        type=str
    )
    arg_parser.add_argument(
        '--dimension',
        help='dimension of input frames (e.g., 128)',
        required=False,
        type=int,
        default=128
    )
    args      = arg_parser.parse_args()
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pitch_net = load_model(args.pitch_model, device)
    yaw_net   = load_model(args.yaw_model, device)
    dataset   = model.PitchYawVideoDataset(
        filename  = args.input,
        inference = True,
        dimension = (args.dimension,)*2,
    )
    pitch_preds   = inference(pitch_net, dataset, device)
    yaw_preds     = inference(yaw_net, dataset, device)
    pitches, yaws = classes_to_pys(pitch_preds, yaw_preds)
    logging.info(f"Saving model predictions to {args.output}.")
    np.savetxt(args.output, np.array([pitches, yaws]).T)
