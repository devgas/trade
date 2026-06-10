import pandas as pd

from src.features import FEATURE_COLUMNS, add_features
from src.signal_engine import latest_feature_frame
from tests.test_features import sample_candles


def test_latest_feature_frame_keeps_numeric_dtypes_for_lightgbm():
    featured = add_features(sample_candles()).dropna(subset=FEATURE_COLUMNS)
    latest = featured.iloc[-1]

    frame = latest_feature_frame(latest)

    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.shape == (1, len(FEATURE_COLUMNS))
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)
