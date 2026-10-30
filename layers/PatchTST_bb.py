import torch
from torch import nn
import torch.nn.functional as F

# 可选：如果你想继续保留 RevIN（可关闭）
from layers.RevIN import RevIN


class PatchTST_Simplified(nn.Module):
    """
    简化版 PatchTST_backbone —— 使用 PyTorch 官方 Transformer 实现
    接口兼容原版，支持 RevIN、patch 分割、可学习位置编码、flatten head。
    """
    def __init__(self,
                 c_in: int,                   # 输入变量个数
                 context_window: int,         # 历史时间窗口
                 target_window: int,          # 预测窗口
                 patch_len: int,              # 每个 patch 长度
                 stride: int,                 # patch 滑动步长
                 max_seq_len: int = 1024,
                 n_layers: int = 3,
                 d_model: int = 128,
                 n_heads: int = 16,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 act: str = "gelu",
                 revin: bool = True,
                 affine: bool = True,
                 subtract_last: bool = False,
                 learn_pe: bool = True,
                 padding_patch: str = None,
                 head_dropout: float = 0.1,
                 individual: bool = False,
                 verbose: bool = False,
                 **kwargs):
        super().__init__()

        # ========= RevIN =========
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # ========= Patchify =========
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.patch_num = patch_num

        # ========= 输入投影 + 位置编码 =========
        self.input_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, patch_num, d_model)) if learn_pe else None

        # ========= Transformer 编码器 =========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=act,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ========= 输出头 (Flatten Head) =========
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.individual = individual

        if individual:
            # 每个变量独立线性映射
            self.linears = nn.ModuleList([
                nn.Sequential(
                    nn.Flatten(start_dim=-2),
                    nn.Linear(self.head_nf, target_window),
                    nn.Dropout(head_dropout)
                )
                for _ in range(self.n_vars)
            ])
        else:
            # 所有变量共享线性映射
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(self.head_nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

        if verbose:
            print(f"[Simplified PatchTST] patch_num={patch_num}, d_model={d_model}, layers={n_layers}, heads={n_heads}")

    # ========= Forward =========
    def forward(self, x):
        """
        x: [bs, nvars, seq_len]
        return: [bs, nvars, target_window]
        """
        bs, nvars, seq_len = x.shape

        # --- RevIN 归一化 ---
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        # --- Patch 化 ---
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [bs, nvars, patch_num, patch_len]

        # --- 线性映射 + 位置编码 ---
        x = self.input_proj(x)  # [bs, nvars, patch_num, d_model]
        if self.pos_emb is not None:
            x = x + self.pos_emb

        # --- 送入 Transformer 编码器 ---
        x = x.view(bs * nvars, self.patch_num, -1)  # 合并变量维度
        x = self.encoder(x)                         # [bs*nvars, patch_num, d_model]
        x = x.view(bs, nvars, self.patch_num, -1)   # 还原维度 [bs, nvars, patch_num, d_model]

        # --- Flatten Head ---
        if self.individual:
            out = []
            for i in range(nvars):
                z = self.linears[i](x[:, i, :, :])  # [bs, target_window]
                out.append(z)
            x = torch.stack(out, dim=1)             # [bs, nvars, target_window]
        else:
            # flatten 每个变量的 patch 表示
            x = x.reshape(bs, nvars, -1)            # [bs, nvars, patch_num*d_model]
            x = self.linear(x)                      # [bs, nvars, target_window]
            x = self.dropout(x)

        # --- RevIN 反归一化 ---
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'denorm')
            x = x.permute(0, 2, 1)

        return x

