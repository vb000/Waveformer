"""
Torch dataset object for synthetically rendered spatial data.
"""

import os
import json
import random
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scaper
import torch
import torchaudio
import torchaudio.transforms as AT
from random import randrange

class FSDSoundScapesDataset(torch.utils.data.Dataset):  # type: ignore
    """
    Base class for FSD Sound Scapes dataset
    """

    _labels = [
    "Acoustic_guitar", "Applause", "Bark", "Bass_drum",
    "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet",
    "Computer_keyboard", "Cough", "Cowbell", "Double_bass",
    "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping",
    "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire",
    "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow",
    "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter",
    "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone",
    "Trumpet", "Violin_or_fiddle", "Writing"]

    def __init__(self, input_dir, dset='', sr=None,
                 resample_rate=None, max_num_targets=1):
        assert dset in ['train', 'val', 'test'], \
            "`dset` must be one of ['train', 'val', 'test']"
        self.dset = dset
        self.max_num_targets = max_num_targets
        self.fg_dir = os.path.join(input_dir, 'FSDKaggle2018/%s' % dset)
        if dset in ['train', 'val']:
            self.bg_dir = os.path.join(
                input_dir,
                'TAU-acoustic-sounds/'
                'TAU-urban-acoustic-scenes-2019-development')
        else:
            self.bg_dir = os.path.join(
                input_dir,
                'TAU-acoustic-sounds/'
                'TAU-urban-acoustic-scenes-2019-evaluation')
        logging.info("Loading %s dataset: fg_dir=%s bg_dir=%s" %
                     (dset, self.fg_dir, self.bg_dir))

        self.samples = sorted(list(
            Path(os.path.join(input_dir, 'jams', dset)).glob('[0-9]*')))

        jamsfile = os.path.join(self.samples[0], 'mixture.jams')
        _, jams, _, _ = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)
        _sr = jams['annotations'][0]['sandbox']['scaper']['sr']
        assert _sr == sr, "Sampling rate provided does not match the data"

        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = resample_rate
        else:
            self.resampler = lambda a: a
            self.sr = sr

    def _get_label_vector(self, labels):
        """
        Generates a multi-hot vector corresponding to `labels`.
        """
        vector = torch.zeros(len(FSDSoundScapesDataset._labels))

        for label in labels:
            idx = FSDSoundScapesDataset._labels.index(label)
            assert vector[idx] == 0, "Repeated labels"
            vector[idx] = 1 

        return vector

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        jamsfile = os.path.join(sample_path, 'mixture.jams')

        mixture, jams, ann_list, event_audio_list = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)
        isolated_events = {}
        for e, a in zip(ann_list, event_audio_list[1:]):
            # 0th event is background
            isolated_events[e[2]] = a
        gt_events = list(pd.read_csv(
            os.path.join(sample_path, 'gt_events.csv'), sep='\t')['label'])

        mixture = torch.from_numpy(mixture).permute(1, 0)
        mixture = self.resampler(mixture.to(torch.float))

        if self.dset == 'train':
            labels = random.sample(gt_events, randrange(1,self.max_num_targets+1))
        elif self.dset == 'val':
            labels = gt_events[:idx%self.max_num_targets+1]
        elif self.dset == 'test':
            labels = gt_events[:self.max_num_targets]
        label_vector = self._get_label_vector(labels)

        gt = torch.zeros_like(
            torch.from_numpy(event_audio_list[1]).permute(1, 0))
        for l in labels:
            gt = gt + torch.from_numpy(isolated_events[l]).permute(1, 0)
        gt = self.resampler(gt.to(torch.float))

        return mixture, label_vector, gt #, jams

def tensorboard_add_sample(writer, tag, sample, step, params):
    """
    Adds a sample of FSDSynthDataset to tensorboard.
    """
    if params['resample_rate'] is not None:
        sr = params['resample_rate']
    else:
        sr = params['sr']
    resample_rate = 16000 if sr > 16000 else sr

    m, l, gt, o = sample
    m, gt, o = (
        torchaudio.functional.resample(_, sr, resample_rate).cpu()
        for _ in (m, gt, o))

    def _add_audio(a, audio_tag, axis, plt_title):
        for i, ch in enumerate(a):
            axis.plot(ch, label='mic %d' % i)
            writer.add_audio(
                '%s/mic %d' % (audio_tag, i), ch.unsqueeze(0), step, resample_rate)
        axis.set_title(plt_title)
        axis.legend()

    for b in range(m.shape[0]):
        label = []
        for i in range(len(l[b, :])):
            if l[b, i] == 1:
                label.append(FSDSoundScapesDataset._labels[i])

        # Add waveforms
        rows = 3 # input, output, gt
        fig = plt.figure(figsize=(10, 2 * rows))
        axes = fig.subplots(rows, 1, sharex=True)
        _add_audio(m[b], '%s/sample_%d/0_input' % (tag, b), axes[0], "Mixed")
        _add_audio(o[b], '%s/sample_%d/1_output' % (tag, b), axes[1], "Output (%s)" % label)
        _add_audio(gt[b], '%s/sample_%d/2_gt' % (tag, b), axes[2], "GT (%s)" % label)
        writer.add_figure('%s/sample_%d/waveform' % (tag, b), fig, step)

def tensorboard_add_metrics(writer, tag, metrics, label, step):
    """
    Add metrics to tensorboard.
    """
    vals = np.asarray(metrics['scale_invariant_signal_noise_ratio'])

    writer.add_histogram('%s/%s' % (tag, 'SI-SNRi'), vals, step)

    label_names = [FSDSoundScapesDataset._labels[torch.argmax(_)] for _ in label]
    for l, v in zip(label_names, vals):
        writer.add_histogram('%s/%s' % (tag, l), v, step)
