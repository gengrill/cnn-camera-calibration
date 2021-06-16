#!/usr/bin/env python3
"""
Data and model classes for solving comma.ai's coding challenge.
"""
import cv2
import torch
import logging
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

torch.set_printoptions(precision=10)

class PitchYawVideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename=None, inference=False):
        self.filename = filename
        self.inference = inference
        self.W = self.H = self.length = 0
        self.frames = self.labels = None
        self.load_frames(filename)
        self.load_labels(filename)
        if not inference:
            self.check_invalid()
    def check_invalid(self): # NaNs break gradients.. filter out
        invalid = np.isnan(self.labels)
        self.frames = self.frames[~(invalid[:,0] | invalid[:,1])]
        self.labels = self.labels[~(invalid[:,0] | invalid[:,1])]
        logging.warn(f"Skipping {self.length-len(self.frames)} of " \
            + f"{self.length} total frames because label values are NaN.")
        self.length = len(self.frames)
    def load_labels(self, filename, ext='.txt'):
        if self.inference:
            self.labels = np.zeros(self.length * 2) \
                .reshape(self.length, 2) \
                .astype(np.float32)
        else: # all labels are stored as e-02 -> multiply by 100
            logging.info(f"Loading labels from {filename+ext}")
            self.labels = np.loadtxt(filename+ext, dtype=np.float32) * 100.0
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
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # grayscale
            frame = frame[self.H//2:, 0:self.W//2] # lower left quadrant
            frame = cv2.resize(frame, dsize=(128, 128), \
                interpolation=cv2.INTER_CUBIC)
            frames += [frame.astype(np.float32)]
        frames = np.array(frames, ndmin=3, dtype=np.float32)
        frames = (frames - np.mean(frames)) / np.std(frames)
        self.frames = frames
        self.length = len(frames)
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return np.array([self.frames[i], self.labels[i]])
    def __iter__(self):
        return zip(self.frames, self.labels)

class PitchYawVideoLoader: # single file loader
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self._iter = iter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = len(dataset)
        if self.length % batch_size != 0:
            logging.warn(f"Batch size {batch_size} results in " \
                + f"{len(dataset)%batch_size} skipped frames.")
    def random_sample_iterator(self):
        import random
        for i in random.sample(range(self.length), self.length):
            yield self.dataset[i]
    def batch_iterator(self, sample_iterator):
        for b in range(0,self.length,self.batch_size):
            frames, labels = zip(*[ \
                                 next(sample_iterator) \
                                 for _ in range(self.batch_size) \
                             ])
            frames = np.expand_dims(np.array(frames, dtype=np.float32), (1,))
            labels = np.array(labels, dtype=np.float32) \
                .reshape(1, self.batch_size*2) # hardcoded 2 for pitch and yaw
            yield frames, labels
    def __iter__(self):
        if self.shuffle: # random frame order
            if self.batch_size == 1:
                return self.random_sample_iterator()
            return self.batch_iterator(self.random_sample_iterator())
        else: # video frame order
            if self.batch_size == 1:
                return self._iter(self.dataset)
            return self.batch_iterator(self._iter(self.dataset))

class PitchYawVideoChainLoader(PitchYawVideoLoader):
    import itertools
    def __init__(self, datasets, batch_size=1, shuffle=False):
        self.dataset = datasets
        self._iter = itertools.chain
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ranges = dict()
        self.length = 0
        for i in range(len(datasets)):
            dataset_min = self.length
            self.length += len(datasets[i])
            self.ranges[i] = (dataset_min, self.length)
        if self.length % batch_size != 0:
            logging.warn(f"Batch size {batch_size} results in" \
                + f" {self.length%batch_size} skipped frames" \
                + f" ({self.length} total).")
    def random_sample_iterator(self):
        import random
        for i in random.sample(range(self.length), self.length):
            dataset = None
            for dataset_index in self.ranges:
                dataset_min, dataset_max = self.ranges[dataset_index]
                if dataset_min <= i < dataset_max:
                    dataset = self.dataset[dataset_index]
                    yield dataset[i-dataset_min]

class DaNet(nn.Module):
    def __init__(self, batch_size=1):
        super(DaNet, self).__init__()
        dim1 = DaNet._dim(128, batch_size); dim2 = DaNet._dim(dim1, batch_size)
        self.conv1 = nn.Conv2d(1, 3, batch_size)
        self.conv1_bn = nn.BatchNorm2d(3) # batch norm boosts vanishing grads
        self.conv2 = nn.Conv2d(3, 6, batch_size)
        self.conv2_bn = nn.BatchNorm2d(6) # and also helps with convergence
        self.flatten = lambda x : x.view(-1, batch_size*6*dim2*dim2)
        self.fc1 = nn.Linear(batch_size*6*dim2*dim2, batch_size*6*dim2)
        self.fc2 = nn.Linear(batch_size*6*dim2, 64)
        self.fc3 = nn.Linear(64, 2*batch_size)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.conv1_bn(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.conv2_bn(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def _dim(insize, batch_size):
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # divide by 2 because of max pooling with size 2
        return ((insize + 2 * 0 - 1 * (batch_size - 1) - 1)//1 + 1)//2
