"""Joint susceptibility prediction and causal collateral-resistance estimation on ARMD."""

__version__ = "2.0.0"

try:
    import xgboost as _xgboost
except ImportError:
    _xgboost = None

from armd.config import Config, resolve_data_dir
