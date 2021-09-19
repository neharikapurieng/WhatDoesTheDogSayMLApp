# Custom PyTorch DataLoader Class to load audio files
# and process .wav files
# Audio are transformed into mel spectrum tensors

import re
import shutil
import librosa.display
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import torchaudio
from torch.utils.data import DataLoader
from torchaudio import transforms
import numpy as np
import torchvision
import os
import torch
# import sys
# sys.path.insert(1, '../script')

from AudioUtil import AudioUtil
from model import EfficientNetNotRGB

from torch.utils.data import DataLoader, Dataset, random_split
#from efficientnet_pytorch import EfficientNet
#!/usr/bin/python
# -*- coding: utf-8 -*-

# ----------------------------
# Sound Dataset
# ----------------------------


class SoundDS(Dataset):

    def __init__(self, data_path, transform=None):
        self.data_path = str(data_path)
        self.duration = 5000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4
        self.transform = transform

  # ----------------------------
  # Number of items in dataset
  # ----------------------------

    def __len__(self):
        classes = os.listdir(self.data_path)
        num_classes = len(classes)
        tot = 0
        for clas in classes:
            files = os.path.join(self.data_path, clas)
            tot = tot + len(os.listdir(files))
        return tot

    def value_counts(self):
        tot = 0
        classes = os.listdir(self.data_path)
        value_count = {}
        for clas in classes:
            files = os.path.join(self.data_path, clas)
            tot = tot + len(os.listdir(files))
        return tot

  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------

    def get_labels(self):
        classes = sorted(os.listdir(self.data_path))
        num_classes = len(classes)
        labels = []
        for clas in classes:
            files = os.path.join(self.data_path, clas)
            for file in os.listdir(files):
                labels.append(classes.index(clas))
            return labels

    def __getitem__(self, idx):
        classes = sorted(os.listdir(self.data_path))
        num_classes = len(classes)
        tot = 0
        for clas in classes:
            files = os.path.join(self.data_path, clas)
            for file in os.listdir(files):
                if not '.aml' in file:
                    if tot == idx:
                        pth = os.path.join(files, file)
                        aud = AudioUtil.open(pth)

                    # Some sounds have a higher sample rate, or fewer channels compared to the
                    # majority. So make all sounds have the same number of channels and same
                    # sample rate. Unless the sample rate is the same, the pad_trunc will still
                    # result in arrays of different lengths, even though the sound duration is
                    # the same.

                        reaud = AudioUtil.resample(aud, self.sr)
                        rechan = AudioUtil.rechannel(reaud,
                                self.channel)
                        dur_aud = AudioUtil.pad_trunc(rechan,
                                self.duration)
                        shift_aud = AudioUtil.time_shift(dur_aud,
                                self.shift_pct)
                        sgram = AudioUtil.spectro_gram(shift_aud,
                                n_mels=64, n_fft=1024, hop_len=None)
                        aug_sgram = AudioUtil.spectro_augment(sgram,
                                max_mask_pct=0.1, n_freq_masks=2,
                                n_time_masks=2)
                        aug_sgram = self.transform(np.array(aug_sgram))
                        aug_sgram = torch.reshape(aug_sgram, (aug_sgram.shape[1],aug_sgram.shape[0],aug_sgram.shape[2]))
                        return (aug_sgram, classes.index(clas))
                    tot = tot + 1
