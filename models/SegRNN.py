import os
import sys
from pathlib import Path


def _ridebench_algorithms_dir():
    default_root = Path(__file__).resolve().parents[2] / 'RideBench' / 'ridebench' / 'algorithms'
    return Path(os.environ.get('RIDEBENCH_ALGORITHMS_DIR', default_root)).resolve()


_ALGORITHMS_DIR = _ridebench_algorithms_dir()
if not (_ALGORITHMS_DIR / 'SegRNN' / 'model.py').is_file():
    raise ImportError(f'RideBench SegRNN implementation not found under {_ALGORITHMS_DIR}')
if str(_ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGORITHMS_DIR))

from SegRNN.model import Model as _RideBenchSegRNN  # noqa: E402


class Model(_RideBenchSegRNN):
    """SolarNet experiment adapter for RideBench's SegRNN implementation."""

