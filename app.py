import lightning as L
import os, wget, json
import gradio as gr
from lightning.app.components.serve import ServeGradio
from Waveformer import TARGETS, Waveformer

class ModelDemo(ServeGradio):
    inputs = [gr.Audio(label="Input audio"), gr.CheckboxGroup(choices=TARGETS, label="Input target selection(s)")]
    outputs = gr.Audio(label="Output audio")
    examples = [["data/Sample.wav"]]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        import torch

        if not os.path.exists('default_config.json'):
            config_url = 'https://targetsound.cs.washington.edu/files/default_config.json'
            print("Downloading model configuration from %s:" % config_url)
            wget.download(config_url)

        if not os.path.exists('default_ckpt.pt'):
            ckpt_url = 'https://targetsound.cs.washington.edu/files/default_ckpt.pt'
            print("\nDownloading the checkpoint from %s:" % ckpt_url)
            wget.download(ckpt_url)

        # Instantiate model
        with open('default_config.json') as f:
            params = json.load(f)
        model = Waveformer(**params['model_params'])
        model.load_state_dict(
            torch.load('default_ckpt.pt', map_location=torch.device('cpu'))['model_state_dict'])
        model.eval()
        return model
    
    def predict(self, audio, label_choices):
        import torch, torchaudio
        # Read input audio
        fs, mixture = audio
        mixture = torchaudio.functional.resample(torch.as_tensor(mixture, dtype=torch.float32), orig_freq=fs, new_freq=44100).numpy()
        # if fs != 44100:
        #     raise ValueError("Sampling rate must be 44100, but got %d" % fs)
        mixture = torch.from_numpy(
            mixture).unsqueeze(0).unsqueeze(0).to(torch.float) / (2.0 ** 15)

        # Construct the query vector
        query = torch.zeros(1, len(TARGETS))
        for t in label_choices:
            query[0, TARGETS.index(t)] = 1.

        with torch.no_grad():
            output = (2.0 ** 15) * self.model(mixture, query)

        return fs, output.squeeze(0).squeeze(0).to(torch.short).numpy()

app = L.LightningApp(ModelDemo())
