import numpy as np
import pandas as pd

from src.model_diagnostics import feature_importance_frame, probability_bucket_performance


class ImportanceModel:
    feature_importances_ = np.array([3, 1, 2])


class ProbabilityModel:
    def predict_proba(self, frame):
        probabilities = np.array([0.1, 0.2, 0.4, 0.8])[: len(frame)]
        return np.column_stack([1 - probabilities, probabilities])


def test_feature_importance_frame_sorts_descending():
    result = feature_importance_frame(ImportanceModel(), ["a", "b", "c"])

    assert list(result["feature"]) == ["a", "c", "b"]
    assert list(result["importance"]) == [3, 2, 1]


def test_probability_bucket_performance_groups_predictions():
    frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "target": [0, 0, 1, 1]})

    result = probability_bucket_performance(
        ProbabilityModel(),
        frame,
        feature_columns=["feature"],
        bins=[0.0, 0.5, 1.0],
    )

    assert list(result["bucket"]) == ["0.0-0.5", "0.5-1.0"]
    assert list(result["samples"]) == [3, 1]
    assert list(result["target_rate"]) == [1 / 3, 1.0]
