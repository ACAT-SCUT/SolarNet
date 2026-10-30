import os
import sys
from copy import copy
from pathlib import Path


def _ridebench_algorithms_dir():
    default_root = Path(__file__).resolve().parents[2] / 'RideBench' / 'ridebench' / 'algorithms'
    return Path(os.environ.get('RIDEBENCH_ALGORITHMS_DIR', default_root)).resolve()


_ALGORITHMS_DIR = _ridebench_algorithms_dir()
if not (_ALGORITHMS_DIR / 'TimeBase' / 'model.py').is_file():
    raise ImportError(f'RideBench TimeBase implementation not found under {_ALGORITHMS_DIR}')
if str(_ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGORITHMS_DIR))

from TimeBase.model import Model as _RideBenchTimeBase  # noqa: E402


class Model(_RideBenchTimeBase):
    """SolarNet adapter that exposes TimeBase's optional orthogonal loss."""

    auxiliary_loss = None

    def __init__(self, configs):
        self.requested_seq_len = int(configs.seq_len)
        effective_configs = configs
        if configs.seq_len < configs.period_len:
            effective_configs = copy(configs)
            effective_configs.seq_len = int(configs.period_len)
        super().__init__(effective_configs)

    def forward(self, x, *args, **kwargs):
        if x.shape[1] != self.requested_seq_len:
            raise ValueError(
                f'TimeBase expected {self.requested_seq_len} input steps, got {x.shape[1]}'
            )
        if self.requested_seq_len < self.seq_len:
            repeats = (self.seq_len + self.requested_seq_len - 1) // self.requested_seq_len
            x = x.repeat(1, repeats, 1)[:, -self.seq_len:, :]
        output = super().forward(x, *args, **kwargs)
        if isinstance(output, tuple):
            output, self.auxiliary_loss = output
        else:
            self.auxiliary_loss = None
        return output
