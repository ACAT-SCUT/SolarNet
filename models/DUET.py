import os
import sys
from pathlib import Path


def _ridebench_algorithms_dir():
    default_root = Path(__file__).resolve().parents[2] / 'RideBench' / 'ridebench' / 'algorithms'
    return Path(os.environ.get('RIDEBENCH_ALGORITHMS_DIR', default_root)).resolve()


_ALGORITHMS_DIR = _ridebench_algorithms_dir()
if not (_ALGORITHMS_DIR / 'DUET' / 'model.py').is_file():
    raise ImportError(f'RideBench DUET implementation not found under {_ALGORITHMS_DIR}')
if str(_ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGORITHMS_DIR))

from DUET.model import Model as _RideBenchDUET  # noqa: E402


class Model(_RideBenchDUET):
    """SolarNet adapter that exposes DUET's router loss to the trainer."""

    auxiliary_loss = None

    def forward(self, x, *args, **kwargs):
        output, auxiliary_loss = super().forward(x, *args, **kwargs)
        self.auxiliary_loss = auxiliary_loss
        return output

