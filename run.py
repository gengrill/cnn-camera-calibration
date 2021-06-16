#!/usr/bin/env python3
"""
Pitch and yaw inference with a trained model. Run like:

$ python3 ./run.py --model trained.pyt --input video.hevc --output p_and_y.txt [--batch_size 2]
"""
import torch
import logging
import argparse
import numpy as np

import model

logging.basicConfig(level=logging.INFO)

def inference(trained_net, dataset, batch_size):
    loader = model.PitchYawVideoLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    predictions = np.array([], dtype=np.float32)
    logging.info("Starting inference..")
    for i, data in enumerate(loader, 0):
        frames, labels = data
        inputs = torch.tensor(frames)
        batch_predictions = trained_net(inputs)
        predictions = np.concatenate([
            predictions, batch_predictions.cpu().detach().numpy()[0]
        ])
    logging.info(f"Done. Predicted {len(predictions)} values.")
    return (predictions / 100).reshape(len(predictions)//2,2)

def load_model(filepath, batch_size):
    logging.info(f"Loading trained model from {filepath}.")
    trained_net = model.DaNet(batch_size=batch_size)
    checkpoint  = torch.load(filepath, map_location="cpu")
    trained_net.load_state_dict(checkpoint)
    trained_net.eval()
    return trained_net

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    arg_parser.add_argument(
        '--model',
        help='trained pytorch model',
        required=True,
        type=str
    )
    arg_parser.add_argument(
        '--data',
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
        '--batch_size',
        help='batch size used in the trained model',
        required=False,
        type=int,
        default=2
    )
    args = arg_parser.parse_args()
    net = load_model(args.model, args.batch_size)
    dataset = model.PitchYawVideoDataset(args.data, inference=True)
    predictions = inference(net, dataset, args.batch_size)
    logging.info(f"Saving model predictions to {args.output}.")
    np.savetxt(args.output, predictions)
