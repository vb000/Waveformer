"""
The main training script for training on synthetic data
"""

import argparse
import multiprocessing
import os
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # pylint: disable=unused-import
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

from src.helpers import utils
from src.training.eval import test_epoch
from src.training.synthetic_dataset import FSDSoundScapesDataset as Dataset
from src.training.synthetic_dataset import tensorboard_add_sample

def train_epoch(model: nn.Module, device: torch.device,
                optimizer: optim.Optimizer,
                train_loader: torch.utils.data.dataloader.DataLoader,
                n_items: int, epoch: int = 0,
                writer: SummaryWriter = None, data_params = None) -> float:

    """
    Train a single epoch.
    """
    # Set the model to training.
    model.train()

    # Training loop
    losses = []
    metrics = {}

    with tqdm(total=len(train_loader), desc='Train', ncols=100) as t:
        for batch_idx, (mixed, label, gt) in enumerate(train_loader):
            mixed = mixed.to(device)
            label = label.to(device)
            gt = gt.to(device)

            # Reset grad
            optimizer.zero_grad()

            # Run through the model
            output = model(mixed, label)

            # Compute loss
            loss = network.loss(output, gt)

            losses.append(loss.item())

            # Backpropagation
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # Update the weights
            optimizer.step()

            metrics_batch = network.metrics(mixed.detach(), output.detach(),
                                            gt.detach())
            for k in metrics_batch.keys():
                if not k in metrics:
                    metrics[k] = metrics_batch[k]
                else:
                    metrics[k] += metrics_batch[k]

            if writer is not None and batch_idx == 0:
                tensorboard_add_sample(
                    writer, tag='Train',
                    sample=(mixed.detach()[:8], label.detach()[:8],
                            gt.detach()[:8], output.detach()[:8]),
                    step=epoch, params=data_params)

            # Show current loss in the progress meter
            t.set_postfix(loss='%.05f'%loss.item())
            t.update()

            if n_items is not None and batch_idx == n_items:
                break

    avg_metrics = {k: np.mean(metrics[k]) for k in metrics.keys()}
    avg_metrics['loss'] = np.mean(losses)
    avg_metrics_str = "Train:"
    for m in avg_metrics.keys():
        avg_metrics_str += ' %s=%.04f' % (m, avg_metrics[m])
    logging.info(avg_metrics_str)

    return avg_metrics


def train(args: argparse.Namespace):
    """
    Train the network.
    """

    # Load dataset
    data_train = Dataset(**args.train_data)
    logging.info("Loaded train dataset at %s containing %d elements" %
                 (args.train_data['input_dir'], len(data_train)))
    data_val = Dataset(**args.val_data)
    logging.info("Loaded test dataset at %s containing %d elements" %
                 (args.val_data['input_dir'], len(data_val)))

    # Set up the device and workers.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        gpu_ids = args.gpu_ids if args.gpu_ids is not None\
                        else range(torch.cuda.device_count())
        device_ids = [_ for _ in gpu_ids]
        data_parallel = len(device_ids) > 1
        device = 'cuda:%d' % device_ids[0]
        torch.cuda.set_device(device_ids[0])
        logging.info("Using CUDA devices: %s" % str(device_ids))
    else:
        data_parallel = False
        device = torch.device('cpu')
        logging.info("Using device: CPU")

    # Set multiprocessing params
    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    } if use_cuda else {}

    # Set up data loaders
    #print(args.batch_size, args.eval_batch_size)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=args.eval_batch_size,
                                             **kwargs)

    # Set up model
    model = network.Net(**args.model_params)

    # Add graph to tensorboard with example train samples
    # _mixed, _label, _ = next(iter(val_loader))
    # args.writer.add_graph(model, (_mixed, _label))

    if use_cuda and data_parallel:
        model = nn.DataParallel(model, device_ids=device_ids)
        logging.info("Using data parallel model")
    model.to(device)

    # Set up the optimizer
    logging.info("Initializing optimizer with %s" % str(args.optim))
    optimizer = network.optimizer(model, **args.optim, data_parallel=data_parallel)
    logging.info('Learning rates initialized to:' + utils.format_lr_info(optimizer))

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **args.lr_sched)
    logging.info("Initialized LR scheduler with params: fix_lr_epochs=%d %s"
                 % (args.fix_lr_epochs, str(args.lr_sched)))

    base_metric = args.base_metric
    train_metrics = {}
    val_metrics = {}

    # Load the model if `args.start_epoch` is greater than 0. This will load the
    # model from epoch = `args.start_epoch - 1`
    assert args.start_epoch >=0, "start_epoch must be greater than 0."
    if args.start_epoch > 0:
        checkpoint_path = os.path.join(args.exp_dir,
                                       '%d.pt' % (args.start_epoch - 1))
        _, train_metrics, val_metrics = utils.load_checkpoint(
            checkpoint_path, model, optim=optimizer, lr_sched=lr_scheduler,
            data_parallel=data_parallel)
        logging.info("Loaded checkpoint from %s" % checkpoint_path)
        logging.info("Learning rates restored to:" + utils.format_lr_info(optimizer))

    # Training loop
    try:
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        for epoch in range(args.start_epoch, args.epochs + 1):
            logging.info("Epoch %d:" % epoch)
            checkpoint_file = os.path.join(args.exp_dir, '%d.pt' % epoch)
            assert not os.path.exists(checkpoint_file), \
                "Checkpoint file %s already exists" % checkpoint_file
            #print("---- begin trianivg")
            curr_train_metrics = train_epoch(model, device, optimizer,
                                             train_loader, args.n_train_items,
                                             epoch=epoch, writer=args.writer,
                                             data_params=args.train_data)
            #raise KeyboardInterrupt
            curr_test_metrics = test_epoch(model, device, val_loader,
                                           args.n_test_items, network.loss,
                                           network.metrics, epoch=epoch,
                                           writer=args.writer,
                                           data_params=args.val_data)
            # LR scheduler
            if epoch >= args.fix_lr_epochs:
                lr_scheduler.step(curr_test_metrics[base_metric])
                logging.info(
                    "LR after scheduling step: %s" %
                    [_['lr'] for _ in optimizer.param_groups])

            # Write metrics to tensorboard
            args.writer.add_scalars('Train', curr_train_metrics, epoch)
            args.writer.add_scalars('Val', curr_test_metrics, epoch)
            args.writer.flush()

            for k in curr_train_metrics.keys():
                if not k in train_metrics:
                    train_metrics[k] = [curr_train_metrics[k]]
                else:
                    train_metrics[k].append(curr_train_metrics[k])

            for k in curr_test_metrics.keys():
                if not k in val_metrics:
                    val_metrics[k] = [curr_test_metrics[k]]
                else:
                    val_metrics[k].append(curr_test_metrics[k])

            if max(val_metrics[base_metric]) == val_metrics[base_metric][-1]:
                logging.info("Found best validation %s!" % base_metric)

            utils.save_checkpoint(
                checkpoint_file, epoch, model, optimizer, lr_scheduler,
                train_metrics, val_metrics, data_parallel)
            logging.info("Saved checkpoint at %s" % checkpoint_file)

            utils.save_graph(train_metrics, val_metrics, args.exp_dir)

        return train_metrics, val_metrics


    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:  # pylint: disable=broad-except
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('exp_dir', type=str,
                        default='./experiments/fsd_mask_label_mult',
                        help="Path to save checkpoints and logs.")

    parser.add_argument('--n_train_items', type=int, default=None,
                        help="Number of items to train on in each epoch")
    parser.add_argument('--n_test_items', type=int, default=None,
                        help="Number of items to test.")
    parser.add_argument('--start_epoch', type=int, default=0,
                        help="Start epoch")
    parser.add_argument('--pretrain_path', type=str,
                        help="Path to pretrained weights")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help="List of GPU ids used for training. "
                        "Eg., --gpu_ids 2 4. All GPUs are used by default.")
    parser.add_argument('--detect_anomaly', dest='detect_anomaly',
                        action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--wandb', dest='wandb', action='store_true',
                        help="Whether to sync tensorboard to wandb")

    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    random.seed(230)
    np.random.seed(230)
    if args.use_cuda:
        torch.cuda.manual_seed(230)

    # Set up checkpoints
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    utils.set_logger(os.path.join(args.exp_dir, 'train.log'))

    # Load model and training params
    params = utils.Params(os.path.join(args.exp_dir, 'config.json'))
    for k, v in params.__dict__.items():
        vars(args)[k] = v

    # Initialize tensorboard writer
    tensorboard_dir = os.path.join(args.exp_dir, 'tensorboard')
    args.writer = SummaryWriter(tensorboard_dir, purge_step=args.start_epoch)
    if args.wandb:
        import wandb
        wandb.init(
            project='Semaudio', sync_tensorboard=True,
            dir=tensorboard_dir, name=os.path.basename(args.exp_dir))

    exec("import %s as network" % args.model)
    logging.info("Imported the model from '%s'." % args.model)

    train(args)

    args.writer.close()
    if args.wandb:
        wandb.finish()
