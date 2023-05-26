# Waveformer (a DNN for low-latency audio processing)

[![Gradio demo](https://img.shields.io/badge/arxiv-abs-green)](https://arxiv.org/abs/2211.02250) [![Gradio demo](https://img.shields.io/badge/ICASSP_2023-pdf-green)](https://arxiv.org/pdf/2211.02250) [![Gradio demo](https://img.shields.io/badge/Gradio-app-blue)](https://huggingface.co/spaces/uwx/waveformer)

This repository provides code for the Waveformer architecture proposed in the paper, __Real-Time Target Sound Extraction__, presented at ICASSP 2023. Waveformer is a low-latency audio processing model implementing streaming inference -- the model process a ~10 ms input audio chunk at each time step, while only looking at past chunks and no future chunks. On a Core i5 CPU using a single thread, real-time factors (RTFs) of different model configurations range from 0.66 to 0.94, with an end-to-end latency less than 20 ms.

https://github.com/vb000/Waveformer/assets/16723254/b0ac45bf-2718-4beb-9514-19752b400606

## Architecture

![Screenshot 2023-05-26 at 1 20 52 PM](https://github.com/vb000/Waveformer/assets/16723254/749ee264-ff71-4d13-8b7b-ed0c073d7bb7)


## Non-causal Waveformer

For the purpose of comparing the Waveformer architecture with other non-causal source separation and target source extraction architectures, we provide a non-causal version of the architecture at [src/training/non_causal_dcc_tf.py](src/training/non_causal_dcc_tf.py).

## Setup


    # Commands in all sections except the Dataset section are run from repo's toplevel directory
    conda create --name waveformer python=3.8
    conda activate waveformer
    pip install -r requirements.txt

## Bring Your Own Audio

You could run the model on your audio files using the `Waveformer.py` script. Example commands below use the sample audio mixture provided at `data/Sample.wav`. If running for the first time, the script downloads the default configuration file and checkpoint to the current directory.

    # Usage: python Waveformer.py [-h] [--targets TARGETS [TARGETS ...]] input output
    
    # Single-target extraction
    python Waveformer.py data/Sample.wav output_typing.wav --targets Computer_keyboard
    
    # Multi-target extraction
    python Waveformer.py data/Sample.wav output_bark_cough.wav --targets Bark Cough

List of all possible targets can be found using:

    python Waveformer.py -h

## Training and Evaluation

### Dataset

We use [Scaper](https://github.com/justinsalamon/scaper) toolkit to synthetically generate audio mixtures. Each audio mixture is generated on-the-fly, during training or evaluation, using Scaper's `generate_from_jams` function on a [`.jams`](https://jams.readthedocs.io/en/stable/) specification file. We provide (in the step 3 below) `.jams` specification files for all training, validation and evaluation samples used in our experiments. The `.jams` specifications are generated using [FSDKaggle2018](https://zenodo.org/record/2552860) and [TAU Urban Acoustic Scenes 2019](https://dcase.community/challenge2019/task-acoustic-scene-classification) datasets as sources for foreground and background sounds, respectively. Steps to create the dataset:

1. Go to the `data` directory:

        cd data

2. Download [FSDKaggle2018](https://zenodo.org/record/2552860), [TAU Urban Acoustic Scenes 2019, Development dataset](https://zenodo.org/record/2589280) and [TAU Urban Acoustic Scenes 2019, Evaluation dataset](https://zenodo.org/record/3063822) datasets using the `data/download.py` script:

        python download.py

3. Download and uncompress [FSDSoundScapes](https://targetsound.cs.washington.edu/files/FSDSoundScapes.zip) dataset:

        wget https://targetsound.cs.washington.edu/files/FSDSoundScapes.zip
        unzip FSDSoundScapes.zip

    This step creates the `data/FSDSoundScapes` directory. `FSDSoundScapes` would contain `.jams` specifications for training, validation and test samples used in the paper. Training and evaluation pipeline expect source samples (samples in `FSDKaggle2018` and `TAU Urban Acoustic Scenes 2019` datasets) at specific locations realtive to `FSDSoundScapes`. Following steps move source samples to appropriate locations.

4. Uncompress FSDKaggle2018 dataset and create scaper source:

        unzip FSDKaggle2018/\*.zip -d FSDKaggle2018
        python fsd_scaper_source_gen.py FSDKaggle2018 ./FSDSoundScapes/FSDKaggle2018 ./FSDSoundScapes/FSDKaggle2018

5. Uncompress TAU Urban Acoustic Scenes 2019 dataset to `FSDSoundScapes` directory:

        unzip TAU-acoustic-sounds/\*.zip -d FSDSoundScapes/TAU-acoustic-sounds/

### Training

    python -W ignore -m src.training.train experiments/<Experiment dir with config.json> --use_cuda

### Evaluation

Pretrained checkpoints are available at [experiments.zip](https://targetsound.cs.washington.edu/files/experiments.zip). These can be downloaded and uncompressed to appropriate locations using:

    wget https://targetsound.cs.washington.edu/files/experiments.zip
    unzip -o experiments.zip -d experiments

Run evaluation script:

    python -W ignore -m src.training.eval experiments/<Experiment dir with config.json and checkpoints> --use_cuda

### Note

During the sample generation, when the amplitude of mixture sum to greater than 1, peak normalization is used to renormalize the mixtures. This results in a bunch of Scaper warnings during training and evaluation. `-W ignore` flag is used for a clearner output to the console.

## Citation

    @misc{veluri2022realtime,
      title={Real-Time Target Sound Extraction}, 
      author={Bandhav Veluri and Justin Chan and Malek Itani and Tuochao Chen and Takuya Yoshioka and Shyamnath Gollakota},
      year={2022},
      eprint={2211.02250},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
    }
