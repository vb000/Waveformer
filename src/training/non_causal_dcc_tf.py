import math
from collections import OrderedDict
from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding

from src.training.dcc_tf import mod_pad, DepthwiseSeparableConv, LayerNormPermuted

class DilatedConvEncoder(nn.Module):
    """
    A dilated causal convolution based encoder for encoding
    time domain audio input into latent space.
    """
    def __init__(self, channels, num_layers, kernel_size=3):
        super(DilatedConvEncoder, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Kernel size must be odd."

        # Dilated causal conv layers aggregate previous context to obtain
        # contexful encoded input.
        _dcc_layers = OrderedDict()
        for i in range(num_layers):
            dcc_layer = DepthwiseSeparableConv(
                channels, channels, kernel_size=3, stride=1,
                padding=(kernel_size // 2) * 2**i, dilation=2**i)
            _dcc_layers.update({'dcc_%d' % i: dcc_layer})
        self.dcc_layers = nn.Sequential(_dcc_layers)

    def forward(self, x):
        for layer in self.dcc_layers:
            x = x + layer(x)
        return x

class LinearTransformerDecoder(nn.Module):
    """
    A casual transformer decoder which decodes input vectors using
    precisely `ctx_len` past vectors in the sequence, and using no future
    vectors at all.
    """
    def __init__(self, model_dim, chunk_size, num_layers,
                 nhead, use_pos_enc, ff_dim):
        super(LinearTransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.chunk_size = chunk_size
        self.nhead = nhead
        self.use_pos_enc = use_pos_enc
        self.unfold = nn.Unfold(kernel_size=(3 * chunk_size, 1), stride=chunk_size)
        self.pos_enc = PositionalEncoding(model_dim, max_len=10 * chunk_size)
        self.tf_dec_layers = nn.ModuleList([nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim,
            batch_first=True) for _ in range(num_layers)])

    def _permute_and_unfold(self, x):
        """
        Unfolds the sequence into a batch of sequences.

        Args:
            x: [B, C, L]
        Returns:
            [B * (L // chunk_size), 3 * chunk_size, C]
        """
        B, C, L = x.shape
        x = F.pad(x, (self.chunk_size, self.chunk_size)) # [B, C, L + 2 * chunk_size]
        x = self.unfold(x.unsqueeze(-1)) # [B, C * 3 * chunk_size, -1]
        x = x.view(B, C, 3 * self.chunk_size, -1).permute(0, 3, 2, 1) # [B, -1, 3 * chunk_size, C]
        x = x.reshape(-1, 3 * self.chunk_size, C) # [B * (L // chunk_size), 3 * chunk_size, C]
        return x

    def forward(self, tgt, mem, K=1000):
        """
        Args:
            tgt: [B, model_dim, T]
            mem: [B, model_dim, T]
            K: Number of chunks to process at a time to avoid OOM.
        """
        mem, _ = mod_pad(mem, self.chunk_size, (0, 0))
        tgt, mod = mod_pad(tgt, self.chunk_size, (0, 0))

        # Input sequence length
        B, C, T = tgt.shape

        tgt = self._permute_and_unfold(tgt) # [B * (T // chunk_size), 3 * chunk_size, C]
        mem = self._permute_and_unfold(mem) # [B * (T // chunk_size), 3 * chunk_size, C]

        # Positional encoding
        if self.use_pos_enc:
            mem = mem + self.pos_enc(mem)
            tgt = tgt + self.pos_enc(tgt)

        for i, tf_dec_layer in enumerate(self.tf_dec_layers):
            _tgt = torch.zeros_like(tgt)
            for i in range(int(math.ceil(tgt.shape[0] / K))):
                _tgt[i*K:(i+1)*K] = tf_dec_layer(tgt[i*K:(i+1)*K], mem[i*K:(i+1)*K])
            tgt = _tgt

        # Permute back to [B, T, C]
        tgt = tgt[:, self.chunk_size:-self.chunk_size, :]
        tgt = tgt.reshape(B, -1, C) # [B, T, C]
        tgt = tgt.permute(0, 2, 1)
        if mod != 0:
            tgt = tgt[..., :-mod]

        return tgt

class MaskNet(nn.Module):
    def __init__(self, enc_dim, num_enc_layers, dec_dim, dec_chunk_size,
                 num_dec_layers, use_pos_enc, skip_connection, proj):
        super(MaskNet, self).__init__()
        self.skip_connection = skip_connection
        self.proj = proj

        # Encoder based on dilated causal convolutions.
        self.encoder = DilatedConvEncoder(channels=enc_dim,
                                          num_layers=num_enc_layers)

        # Project between encoder and decoder dimensions
        self.proj_e2d_e = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())
        self.proj_e2d_l = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())
        self.proj_d2e = nn.Sequential(
            nn.Conv1d(dec_dim, enc_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())

        # Transformer decoder that operates on chunks of size
        # buffer size.
        self.decoder = LinearTransformerDecoder(
            model_dim=dec_dim, chunk_size=dec_chunk_size, num_layers=num_dec_layers,
            nhead=8, use_pos_enc=use_pos_enc, ff_dim=2 * dec_dim)

    def forward(self, x, l):
        """
        Generates a mask based on encoded input `e` and the one-hot
        label `label`.

        Args:
            x: [B, C, T]
                Input audio sequence
            l: [B, C]
                Label embedding
        """
        # Enocder the label integrated input
        e = self.encoder(x)

        # Label integration
        l = l.unsqueeze(2) * e

        # Project to `dec_dim` dimensions
        if self.proj:
            e = self.proj_e2d_e(e)
            m = self.proj_e2d_l(l)
            # Cross-attention to predict the mask
            m = self.decoder(m, e)
        else:
            # Cross-attention to predict the mask
            m = self.decoder(l, e)

        # Project mask to encoder dimensions
        if self.proj:
            m = self.proj_d2e(m)

        # Final mask after residual connection
        if self.skip_connection:
            m = l + m

        return m

class Net(nn.Module):
    def __init__(self, label_len, L=8,
                 enc_dim=512, num_enc_layers=10,
                 dec_dim=256, dec_buf_len=100, num_dec_layers=2,
                 dec_chunk_size=72, use_pos_enc=True, skip_connection=True,
                 proj=True, lookahead=True):
        super(Net, self).__init__()
        self.L = L
        self.enc_dim = enc_dim
        self.lookahead = lookahead

        # Input conv to convert input audio to a latent representation
        kernel_size = 3 * L if lookahead else L
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=enc_dim, kernel_size=kernel_size, stride=L,
                      padding=0, bias=False),
            nn.ReLU())

        # Label embedding layer
        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.ReLU())

        # Mask generator
        self.mask_gen = MaskNet(
            enc_dim=enc_dim, num_enc_layers=num_enc_layers,
            dec_dim=dec_dim, dec_chunk_size=dec_chunk_size,
            num_dec_layers=num_dec_layers, use_pos_enc=use_pos_enc,
            skip_connection=skip_connection, proj=proj)

        # Output conv layer
        self.out_conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=enc_dim, out_channels=1,
                kernel_size=3 * L,
                stride=L,
                padding=L, bias=False),
            nn.Tanh())

    def forward(self, x, label):
        """
        Extracts the audio corresponding to the `label` in the given
        `mixture`. Generates `chunk_size` samples per iteration.

        Args:
            mixed: [B, n_mics, T]
                input audio mixture
            label: [B, num_labels]
                one hot label
        Returns:
            out: [B, n_spk, T]
                extracted audio with sounds corresponding to the `label`
        """
        mod = 0
        pad_size = (self.L, self.L) if self.lookahead else (0, 0)
        x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)

        # Generate latent space representation of the input
        x = self.in_conv(x)

        # Generate label embedding
        l = self.label_embedding(label) # [B, label_len] --> [B, channels]

        # Generate mask corresponding to the label
        m = self.mask_gen(x, l)

        # Apply mask and decode
        x = x * m
        x = self.out_conv(x)

        # Remove mod padding, if present.
        if mod != 0:
            x = x[:, :, :-mod]

        return x

# Define optimizer, loss and metrics

def optimizer(model, data_parallel=False, **kwargs):
    return optim.Adam(model.parameters(), **kwargs)

def loss(pred, tgt):
    return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()

def metrics(mixed, output, gt):
    """ Function to compute metrics """
    metrics = {}

    def metric_i(metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append((metric(p, t) - metric(s, t)).cpu().item())
        return _vals

    for m_fn in [snr, si_snr]:
        metrics[m_fn.__name__] = metric_i(m_fn,
                                          mixed[:, :gt.shape[1], :],
                                          output,
                                          gt)

    return metrics

if __name__ == '__main__':
    model = Net(label_len=41, L=8, enc_dim=256, num_enc_layers=10,
                dec_dim=128, num_dec_layers=1, dec_chunk_size=100)
    x = torch.randn(2, 1, 48001)
    y = torch.zeros(2, 41)
    y[:, 0] = 1
    z = model(x, y)
    print(z.shape)
