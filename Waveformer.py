import argparse
import os

import wget
import torch
import torchaudio

from src.helpers import utils
from src.training.dcc_tf import Net as Waveformer

TARGETS = [
    "Acoustic_guitar", "Applause", "Bark", "Bass_drum",
    "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet",
    "Computer_keyboard", "Cough", "Cowbell", "Double_bass",
    "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping",
    "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire",
    "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow",
    "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter",
    "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone",
    "Trumpet", "Violin_or_fiddle", "Writing"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type=str, default=None,
        help="Path to the input audio file.")
    parser.add_argument(
        'output', type=str, default=None,
        help="Path to the output audio file (output is written in the .wav format)."
    )
    parser.add_argument(
        '--targets', nargs='+', type=str, default=[],
        help="Targets to output. Pick a subset of: %s" % TARGETS)
    args = parser.parse_args()

    if not os.path.exists('default_config.json'):
        config_url = 'https://targetsound.cs.washington.edu/files/default_config.json'
        print("Downloading model configuration from %s:" % config_url)
        wget.download(config_url)

    if not os.path.exists('default_ckpt.pt'):
        ckpt_url = 'https://targetsound.cs.washington.edu/files/default_ckpt.pt'
        print("\nDownloading the checkpoint from %s:" % ckpt_url)
        wget.download(ckpt_url)

    # Instantiate model
    params = utils.Params('default_config.json')
    model = Waveformer(**params.model_params)
    utils.load_checkpoint('default_ckpt.pt', model)
    model.eval()

    # Read input audio
    mixture, fs = torchaudio.load(args.input)
    assert fs == 44100, "Input sampling rate must be 44.1 khz."
    mixture = torchaudio.functional.resample(mixture, orig_freq=fs, new_freq=44100)
    mixture = mixture.unsqueeze(0)
    print("Loaded input audio from %s" % args.input)

    # Construct the query vector
    if len(args.targets) == 0:
        query = torch.ones(1, len(TARGETS))
    else:
        query = torch.zeros(1, len(TARGETS))
        for t in args.targets:
            query[0, TARGETS.index(t)] = 1.

    with torch.no_grad():
        output = model(mixture, query)
    print("Inference done. Saving output audio to %s" % args.output)

    assert not os.path.exists(args.output), "Output file already exists."
    torchaudio.save(args.output, output.squeeze(0), fs)
