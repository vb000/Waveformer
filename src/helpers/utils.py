"""A collection of useful helper functions"""

import os
import logging
import json

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)
import matplotlib.pyplot as plt

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def save_graph(train_metrics, test_metrics, save_dir):
    metrics = [snr, si_snr]
    results = {'train_loss': train_metrics['loss'],
               'test_loss' : test_metrics['loss']}

    for m_fn in metrics:
        results["train_"+m_fn.__name__] = train_metrics[m_fn.__name__]
        results["test_"+m_fn.__name__] = test_metrics[m_fn.__name__]

    results_pd = pd.DataFrame(results)

    results_pd.to_csv(os.path.join(save_dir, 'results.csv'))

    fig, temp_ax = plt.subplots(2, 3, figsize=(15,10))
    axs=[]
    for i in temp_ax:
        for j in i:
            axs.append(j)

    x = range(len(train_metrics['loss']))
    axs[0].plot(x, train_metrics['loss'], label='train')
    axs[0].plot(x, test_metrics['loss'], label='test')
    axs[0].set(ylabel='Loss')
    axs[0].set(xlabel='Epoch')
    axs[0].set_title('loss',fontweight='bold')
    axs[0].legend()

    for i in range(len(metrics)):
        axs[i+1].plot(x, train_metrics[metrics[i].__name__], label='train')
        axs[i+1].plot(x, test_metrics[metrics[i].__name__], label='test')
        axs[i+1].set(xlabel='Epoch')
        axs[i+1].set_title(metrics[i].__name__,fontweight='bold')
        axs[i+1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results.png'))
    plt.close(fig)

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

def load_checkpoint(checkpoint, model, optim=None, lr_sched=None, data_parallel=False):
    """Loads model parameters (state_dict) from file_path.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        data_parallel: (bool) if the model is a data parallel model
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))

    state_dict = torch.load(checkpoint)

    if data_parallel:
        state_dict['model_state_dict'] = {
            'module.' + k: state_dict['model_state_dict'][k]
            for k in state_dict['model_state_dict'].keys()}
    model.load_state_dict(state_dict['model_state_dict'])

    if optim is not None:
        optim.load_state_dict(state_dict['optim_state_dict'])

    if lr_sched is not None:
        lr_sched.load_state_dict(state_dict['lr_sched_state_dict'])

    return state_dict['epoch'], state_dict['train_metrics'], \
           state_dict['val_metrics']

def save_checkpoint(checkpoint, epoch, model, optim=None, lr_sched=None,
                    train_metrics=None, val_metrics=None, data_parallel=False):
    """Saves model parameters (state_dict) to file_path.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        data_parallel: (bool) if the model is a data parallel model
    """
    if os.path.exists(checkpoint):
        raise("File already exists {}".format(checkpoint))

    model_state_dict = model.state_dict()
    if data_parallel:
        model_state_dict = {
            k.partition('module.')[2]:
            model_state_dict[k] for k in model_state_dict.keys()}

    optim_state_dict = None if not optim else optim.state_dict()
    lr_sched_state_dict = None if not lr_sched else lr_sched.state_dict()

    state_dict = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optim_state_dict': optim_state_dict,
        'lr_sched_state_dict': lr_sched_state_dict,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

    torch.save(state_dict, checkpoint)

def model_size(model):
    """
    Returns size of the `model` in millions of parameters.
    """
    num_train_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    return num_train_params / 1e6

def run_time(model, inputs, profiling=False):
    """
    Returns runtime of a model in ms.
    """
    # Warmup
    for _ in range(100):
        output = model(*inputs)

    with profile(activities=[ProfilerActivity.CPU],
                 record_shapes=True) as prof:
        with record_function("model_inference"):
            output = model(*inputs)

    # Print profiling results
    if profiling:
        print(prof.key_averages().table(sort_by="self_cpu_time_total",
                                        row_limit=20))

    # Return runtime in ms
    return prof.profiler.self_cpu_time_total / 1000

def format_lr_info(optimizer):
    lr_info = ""
    for i, pg in enumerate(optimizer.param_groups):
        lr_info += " {group %d: params=%.5fM lr=%.1E}" % (
            i, sum([p.numel() for p in pg['params']]) / (1024 ** 2), pg['lr'])
    return lr_info

