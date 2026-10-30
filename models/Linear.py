import torch.nn as nn


class Model(nn.Module):
    """Plain channel-independent linear forecasting baseline."""

    def __init__(self, configs):
        super().__init__()
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x):
        return self.projection(x.permute(0, 2, 1)).permute(0, 2, 1)
