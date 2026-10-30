import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


DEFAULT_PATCH_CONFIG = {
    24: ([3, 6], [1, 3]),
    48: ([6, 12], [3, 6]),
    96: ([12, 24, 48], [6, 12, 24]),
    144: ([24, 48, 96], [12, 24, 48]),
    288: ([48, 96, 144], [24, 48, 96]),
}


def _resolve_patch_config(period_len):
    if period_len in DEFAULT_PATCH_CONFIG:
        return DEFAULT_PATCH_CONFIG[period_len]

    if period_len <= 48:
        patch_sizes = [max(2, period_len // 8), max(4, period_len // 4)]
    else:
        patch_sizes = [max(4, period_len // 6), max(8, period_len // 3), max(12, period_len // 2)]
    patch_strides = [max(1, size // 2) for size in patch_sizes]
    return patch_sizes, patch_strides


def _compute_patch_count(seq_len, patch_len, stride):
    if seq_len <= patch_len:
        return 1
    return 1 + math.ceil((seq_len - patch_len) / stride)


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        # x: [B, L, C]
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        x_pad = F.pad(x.permute(0, 2, 1), (pad_left, pad_right), mode='replicate')
        trend = self.avg(x_pad).permute(0, 2, 1)
        residual = x - trend
        return trend, residual


class FrequencyGatedBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, N, D]
        freq = torch.fft.rfft(x, dim=2, norm='ortho')
        freq_feat = torch.cat([freq.real, freq.imag], dim=-1)
        gate = self.gate(freq_feat)
        gated = freq * gate
        return torch.fft.irfft(gated, n=x.size(2), dim=2, norm='ortho')


class FourierPatchProcessor(nn.Module):
    def __init__(self, patch_len, num_patches, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = num_patches

        self.input_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=max(1, min(n_heads, d_model)),
            batch_first=True,
            dropout=dropout,
        )
        self.freq_gate = FrequencyGatedBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, patches, exog_patches=None):
        # patches: [B, C, N, P]
        bsz, channels, patch_count, _ = patches.shape
        z = self.input_proj(patches)
        z = z + self.pos_emb[:, :patch_count, :].unsqueeze(1)

        # Channel attention: query from target channels, key/value from exogenous patches when available.
        channel_tokens = z.permute(0, 2, 1, 3).reshape(bsz * patch_count, channels, -1)
        if exog_patches is not None:
            exog_embed = self.input_proj(exog_patches)
            exog_tokens = exog_embed.permute(0, 2, 1, 3).reshape(bsz * patch_count, exog_embed.size(1), -1)
            attn_out = self.channel_attn(channel_tokens, exog_tokens, exog_tokens)[0]
        else:
            attn_out = self.channel_attn(channel_tokens, channel_tokens, channel_tokens)[0]
        attn_out = attn_out.reshape(bsz, patch_count, channels, -1).permute(0, 2, 1, 3)
        z = self.norm1(z + attn_out)

        freq_out = self.freq_gate(z)
        z = self.norm2(z + freq_out)
        z = self.norm3(z + self.ffn(z))
        return z.reshape(bsz, channels, -1)


class TrendNet(nn.Module):
    def __init__(self, configs, period_len):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = period_len
        self.use_revin = bool(configs.use_revin)

        if self.use_revin:
            self.revin = RevIN(self.enc_in, affine=bool(configs.affine), subtract_last=bool(configs.subtract_last))

        self.num_patches = max(1, math.ceil(self.seq_len / self.period_len))
        self.processor = FourierPatchProcessor(
            patch_len=self.period_len,
            num_patches=self.num_patches,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            n_heads=configs.n_heads,
            dropout=configs.dropout,
        )
        self.head = nn.Linear(self.num_patches * configs.d_model, self.pred_len)

    def _build_period_patches(self, x):
        # x: [B, L, C] -> [B, C, N, P]
        bsz, seq_len, channels = x.shape
        x = x.permute(0, 2, 1)
        total_len = self.num_patches * self.period_len
        if total_len > seq_len:
            x = F.pad(x, (total_len - seq_len, 0), mode='replicate')
        patches = x.reshape(bsz, channels, self.num_patches, self.period_len)
        return patches

    def forward(self, x, exog=None):
        # x: [B, L, C]
        if self.use_revin:
            x = self.revin(x, 'norm')

        patches = self._build_period_patches(x)
        features = self.processor(patches)
        y = self.head(features).permute(0, 2, 1)

        if self.use_revin:
            y = self.revin(y, 'denorm')
        return y


class ResidualScaleBlock(nn.Module):
    def __init__(self, configs, patch_len, stride):
        super().__init__()
        self.seq_len = configs.seq_len
        self.patch_len = patch_len
        self.stride = stride

        self.num_patches = _compute_patch_count(self.seq_len, self.patch_len, self.stride)
        self.processor = FourierPatchProcessor(
            patch_len=patch_len,
            num_patches=self.num_patches,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            n_heads=configs.n_heads,
            dropout=configs.dropout,
        )

    def _patchify(self, x):
        # x: [B, L, C] -> [B, C, N, P]
        x = x.permute(0, 2, 1)
        seq_len = x.size(-1)
        if seq_len < self.patch_len:
            x = F.pad(x, (0, self.patch_len - seq_len), mode='replicate')
        else:
            remainder = (seq_len - self.patch_len) % self.stride
            if remainder != 0:
                x = F.pad(x, (0, self.stride - remainder), mode='replicate')
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return patches

    def forward(self, x, exog=None):
        patches = self._patchify(x)
        return self.processor(patches)


class ResidualNet(nn.Module):
    def __init__(self, configs, period_len):
        super().__init__()
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.use_revin = bool(configs.use_revin)

        patch_sizes, patch_strides = _resolve_patch_config(period_len)
        self.patch_sizes = patch_sizes
        self.patch_strides = patch_strides

        if self.use_revin:
            self.revin = RevIN(self.enc_in, affine=bool(configs.affine), subtract_last=bool(configs.subtract_last))

        self.scale_blocks = nn.ModuleList(
            [ResidualScaleBlock(configs, patch_len=p, stride=s) for p, s in zip(patch_sizes, patch_strides)]
        )
        fusion_dim = sum(block.num_patches * configs.d_model for block in self.scale_blocks)
        self.head = nn.Linear(fusion_dim, self.pred_len)

    def forward(self, x, exog=None):
        if self.use_revin:
            x = self.revin(x, 'norm')

        multi_scale = [block(x, exog=exog) for block in self.scale_blocks]
        fused = torch.cat(multi_scale, dim=-1)
        y = self.head(fused).permute(0, 2, 1)

        if self.use_revin:
            y = self.revin(y, 'denorm')
        return y


class Model(nn.Module):
    """
    Repository implementation name: MPFGL
    Paper structure reproduced as:
    - dual-level decomposition
    - TrendNet
    - ResidualNet
    - Fourier Patch Processor (channel attention + FFT gating)
    - final trend + residual fusion

    Note:
    The current repo data pipeline does not yet expose NWP covariates, so the
    FPP channel-attention block falls back to self-attention when exogenous
    inputs are unavailable.
    """

    def __init__(self, configs):
        super().__init__()
        self.period_len = configs.cycle
        self.decomposition = MovingAverageDecomposition(kernel_size=self.period_len)
        self.trend_net = TrendNet(configs, period_len=self.period_len)
        self.residual_net = ResidualNet(configs, period_len=self.period_len)

    def forward(self, x, exog=None):
        trend, residual = self.decomposition(x)
        trend_pred = self.trend_net(trend, exog=exog)
        residual_pred = self.residual_net(residual, exog=exog)
        return trend_pred + residual_pred
