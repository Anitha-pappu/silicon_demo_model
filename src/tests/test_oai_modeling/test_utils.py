# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""
Tests for utility functions.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from modeling.models.metrics_utils import evaluate_regression_metrics
from modeling.utils import check_model_features

_NEGATIVE_FLOAT_CLOSE_TO_ZERO = -1e-6


def _check_has_only_positive_values(df: pd.DataFrame) -> bool:
    """Check that ``df`` has only positive values."""
    return bool((df.values > 0).all().all())


class TestCheckModelFeatures(object):
    def test_raises_all_none(self):
        with pytest.raises(ValueError, match="Must specify"):
            check_model_features()

    def test_raises_no_td_indicator_column(self):
        with pytest.raises(ValueError, match="boolean column"):
            check_model_features(td=1)

    def test_returns_list(self):
        expected = ["some", "list", "of", "strings"]
        actual = check_model_features(model_features=expected)
        assert actual == expected  # noqa: S101


class TestEvaluateRegressionMetrics(object):
    def test_perfect_metrics_are_returned(self):
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        prediction = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        perfect_metrics = {
            "mae": 0.0,
            "rmse": 0.0,
            "mse": 0.0,
            "mape": 0.0,
            "r_squared": 1.0,
            "var_score": 1.0,
        }
        metrics = evaluate_regression_metrics(target, prediction)
        assert metrics.keys() == perfect_metrics.keys()
        for metric_name in perfect_metrics.keys():
            assert pytest.approx(metrics[metric_name]) == perfect_metrics[metric_name]

    def test_no_mape_when_zero_in_target(self):
        target = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        prediction = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        perfect_metrics = {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "r_squared": 1.0,
            "var_score": 1.0,
        }
        metrics = evaluate_regression_metrics(target, prediction)
        assert "mape" not in metrics
        assert metrics.keys() == perfect_metrics.keys()
        for metric_name in perfect_metrics.keys():
            assert pytest.approx(metrics[metric_name]) == perfect_metrics[metric_name]

    def test_warns_when_zero_in_target(self, caplog):
        target = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        prediction = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        with caplog.at_level(logging.WARNING):
            evaluate_regression_metrics(target, prediction)
        assert (
            "MAPE was excluded from regression metrics"
            in caplog.text
        )


@pytest.mark.parametrize(
    "df,expected_answer",
    [
        (pd.DataFrame({"a": (1, 1, 1)}), True),
        (pd.DataFrame({"a": (1, 0, 1)}), False),
        (pd.DataFrame({"a": (0, 0, 0)}), False),
        (pd.DataFrame({"a": (1, np.nan, np.nan)}), False),
        (
            pd.DataFrame(
                {
                    "a": (1, 1, 1),
                    "b": (1, 1, 1),
                },
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "a": (1, 0, 1),
                    "b": (1, 1, 1),
                },
            ),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "a": (1, np.nan, 1),
                    "b": (1, 1, 1),
                },
            ),
            False,
        ),
    ],
)
def test_check_has_only_positive_values(
    df: pd.DataFrame,
    expected_answer: bool,
) -> None:
    assert _check_has_only_positive_values(df) == expected_answer
