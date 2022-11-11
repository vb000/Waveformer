import json
import os

import gradio as gr
import lightning as L
import torch
import torchaudio
import wget
from lightning.app.components.serve import ServeGradio

from Waveformer import TARGETS
from Waveformer import Waveformer as WaveformerModel


class ModelDemo(ServeGradio):
    inputs = [
        gr.Audio(label="Input audio"),
        gr.CheckboxGroup(choices=TARGETS, label="Extract target sound"),
    ]
    outputs = gr.Audio(label="Output audio")
    examples = [["data/Sample.wav"]]
    enable_queue: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(cloud_compute=L.CloudCompute("cpu-medium"), **kwargs)

    def build_model(self):
        if not os.path.exists("default_config.json"):
            config_url = (
                "https://targetsound.cs.washington.edu/files/default_config.json"
            )
            print("Downloading model configuration from %s:" % config_url)
            wget.download(config_url)

        if not os.path.exists("default_ckpt.pt"):
            ckpt_url = "https://targetsound.cs.washington.edu/files/default_ckpt.pt"
            print("\nDownloading the checkpoint from %s:" % ckpt_url)
            wget.download(ckpt_url)

        # Instantiate model
        with open("default_config.json") as f:
            params = json.load(f)
        model = WaveformerModel(**params["model_params"])
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"loading model on {device}")
        model.load_state_dict(
            torch.load("default_ckpt.pt", map_location=device)["model_state_dict"]
        )
        return model.to(device).eval()

    @torch.inference_mode()
    def predict(self, audio, label_choices):
        # Read input audio
        fs, mixture = audio
        if fs!=44100:
            mixture = torchaudio.functional.resample(
                torch.as_tensor(mixture, dtype=torch.float32), orig_freq=fs, new_freq=44100
            ).numpy()

        mixture = torch.from_numpy(mixture).unsqueeze(0).unsqueeze(0).to(
            torch.float
        ) / (2.0**15)

        # Construct the query vector
        query = torch.zeros(1, len(TARGETS))
        for t in label_choices:
            query[0, TARGETS.index(t)] = 1.0

        with torch.inference_mode():
            output = (2.0**15) * self.model(mixture, query)

        return fs, output.squeeze(0).squeeze(0).to(torch.short).numpy()


app = L.LightningApp(ModelDemo())
