#!/usr/bin/env python3
"""
Dataset class for loading comma.ai's coding challenge videos.
"""
import cv2
import torch
import logging
import numpy as np

torch.set_printoptions(precision=10)

class PitchYawVideoDataset(torch.utils.data.Dataset):
    def __init__(self, filename=None, inference=False, dimension=(128, 128), target_transform=None):
        self.filename = filename
        self.inference = inference
        self.W = self.H = 0 # original image dimensions
        self.length = 0
        self.dimension  = dimension # resized frame size
        self.frames = self.labels = None
        self.load_frames(filename)
        self.load_labels(filename)
        if not inference:
            self.check_invalid()
        if target_transform is not None:
            logging.info(f"Transforming labels.")
            self.labels = torch.tensor(np.array(list(map(target_transform, self.labels))))
    def check_invalid(self): # NaNs break gradients.. filter out
        invalid = torch.isnan(self.labels)
        self.frames = self.frames[~(invalid[:,0] | invalid[:,1])]
        self.labels = self.labels[~(invalid[:,0] | invalid[:,1])]
        if self.length != len(self.frames):
            logging.warn(f"Skipping {self.length-len(self.frames)} of " \
                + f"{self.length} total frames because label values are NaN.")
            self.length = len(self.frames)
    def load_labels(self, filename, ext='.txt'):
        if self.inference: # no labels in this case
            self.labels = torch.zeros(self.length * 2) \
                .reshape(self.length, 2).float()
        else:
            logging.info(f"Loading labels from {filename+ext}")
            self.labels = torch.from_numpy(np.loadtxt(filename+ext, dtype=np.float32))
    def load_frames(self, filename, ext='.hevc'):
        if filename[-5:] != ext:
            filename += ext
        logging.info(f"Loading frames from {filename}")
        cap = cv2.VideoCapture(filename)
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or len(frame) == 0:
                break
            frames += [cv2.resize(frame, self.dimension)]
        frames = torch.stack([torch.tensor(frame).permute(2,0,1) for frame in frames]).float()
        self.frames = frames
        self.length = len(frames)
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return self.frames[i], self.labels[i]
    def __iter__(self):
        return zip(self.frames, self.labels)
