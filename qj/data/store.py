"""Local storage for OHLCV data as Parquet files (with a cache layer).

Parquet is the workhorse columnar format for quant/data workflows: it is
fast to read for analytics, compressed on disk, and preserves dtypes/dates.
We key files by (source, symbol, freq) so providers are interchangeable and
repeated fetches hit the disk cache instead of the network.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qj.data.base import MarketDataFrame


class ParquetStore:
    """Write/read normalized OHLCV frames to a folder of Parquet files."""

    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, source: str, symbol: str, freq: str) -> Path:
        return self.root / f"{source}_{symbol}_{freq}.parquet"

    def save(self, mdf: MarketDataFrame, overwrite: bool = True) -> Path:
        """Persist a frame. Returns the written path."""
        path = self._path(mdf.source, mdf.symbol, mdf.freq)
        if path.exists() and not overwrite:
            # merge with existing history
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, mdf.df]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.to_parquet(path)
        else:
            mdf.df.to_parquet(path)
        return path

    def load(self, source: str, symbol: str, freq: str) -> MarketDataFrame:
        """Load a previously saved frame. Raises FileNotFoundError if absent."""
        path = self._path(source, symbol, freq)
        if not path.exists():
            raise FileNotFoundError(f"No cached data at {path}; fetch it first.")
        df = pd.read_parquet(path)
        return MarketDataFrame(df=df, symbol=symbol, source=source, freq=freq)

    def has(self, source: str, symbol: str, freq: str) -> bool:
        return self._path(source, symbol, freq).exists()
