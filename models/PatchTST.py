__all__ = ['PatchTST']

from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

# 可选：如果需要 RevIN 支持
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    单类版 PatchTST（整合 PatchTST_Simplified + Model）
    - 使用 PyTorch 官方 TransformerEncoder
    - 支持 RevIN、patch 分割、可学习位置编码、Flatten Head
    """
    def __init__(self, configs,
                 max_seq_len: Optional[int] = 1024,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 norm: str = 'BatchNorm',
                 attn_dropout: float = 0.,
                 act: str = "gelu",
                 key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None,
                 res_attention: bool = True,
                 pre_norm: bool = False,
                 store_attn: bool = False,
                 pe: str = 'zeros',
                 learn_pe: bool = True,
                 pretrain_head: bool = False,
                 head_type: str = 'flatten',
                 verbose: bool = False,
                 **kwargs):
        super().__init__()

        # ===== 从 configs 读取参数 =====
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.use_revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        # ===== RevIN =====
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # ===== Patchify =====
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.patch_num = patch_num

        # ===== 输入投影 + 位置编码 =====
        self.input_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, patch_num, d_model)) if learn_pe else None

        # ===== Transformer 编码器 =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=act,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ===== 输出 Head =====
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.individual = individual

        if individual:
            self.linears = nn.ModuleList([
                nn.Sequential(
                    nn.Flatten(start_dim=-2),
                    nn.Linear(self.head_nf, target_window),
                    nn.Dropout(head_dropout)
                )
                for _ in range(self.n_vars)
            ])
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(self.head_nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

        if verbose:
            print(f"[PatchTST] patch_num={patch_num}, d_model={d_model}, layers={n_layers}, heads={n_heads}")

    # ===== Forward =====
    def forward(self, x):
        """
        x: [Batch, Input length, Channel]
        return: [Batch, Target length, Channel]
        """
        x = x.permute(0, 2, 1)  # [B, C, L]
        bs, nvars, seq_len = x.shape

        # --- RevIN 归一化 ---
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        # --- Patch 化 ---
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, patch_num, patch_len]

        # --- 投影 + 位置编码 ---
        x = self.input_proj(x)  # [B, C, patch_num, d_model]
        if self.pos_emb is not None:
            x = x + self.pos_emb

        # --- Transformer 编码器 ---
        x = x.view(bs * nvars, self.patch_num, -1)
        x = self.encoder(x)
        x = x.view(bs, nvars, self.patch_num, -1)

        # --- Flatten Head ---
        if self.individual:
            outs = []
            for i in range(nvars):
                z = self.linears[i](x[:, i, :, :])
                outs.append(z)
            x = torch.stack(outs, dim=1)
        else:
            x = x.reshape(bs, nvars, -1)
            x = self.linear(x)
            x = self.dropout(x)

        # --- RevIN 反归一化 ---
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'denorm')
            x = x.permute(0, 2, 1)

        x = x.permute(0, 2, 1)  # [B, T_pred, C]
        return x
