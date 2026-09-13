"""Select standard feature columns produced by the feature engine."""

from qj.features.engine import MOMENTUM_WINDOWS, VOL_WINDOWS, ZSCORE_WINDOWS


def default_feature_cols() -> list[str]:
    """The standardized feature set the models consume."""
    cols = ["log_return", "simple_return"]
    cols += [f"mom_{w}" for w in MOMENTUM_WINDOWS]
    cols += [f"vol_{w}" for w in VOL_WINDOWS]
    cols += [f"zscore_{w}" for w in ZSCORE_WINDOWS]
    cols += [f"min_{w}" for w in ZSCORE_WINDOWS]
    cols += [f"max_{w}" for w in ZSCORE_WINDOWS]
    if "vol_ratio" not in cols:
        cols += ["vol_ratio"]
    return cols
