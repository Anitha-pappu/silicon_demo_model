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


import logging
import typing as tp
import warnings

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from modeling import types

logger = logging.getLogger(__name__)


def evaluate_regression_metrics(
    target: types.Vector,
    prediction: types.Vector,
) -> tp.Dict[str, float]:
    """
    Calculate standard set of regression metrics
    using target and prediction vectors:
        * Mean absolute error
        * Rooted mean squared error
        * Mean squared error
        * Mean absolute percentage error
        * R^2` (coefficient of determination)
        * Explained variance

    Args:
        target: vector used as a target for metrics evaluation
        prediction: vector used as a prediction and produces by model

    Notes:
        If target vector contains zeros, then MAPE will be dropped from resulting dict

    Returns:
        Dictionary from metric name into metric value
    """

    is_nan_target = np.isnan(target)
    if is_nan_target.any():
        warnings.warn(
            "Target contains nan values that will be omitted during metrics evaluation",
        )
        target = target[~is_nan_target]
        prediction = prediction[~is_nan_target]

    regression_metrics = {
        "mae": mean_absolute_error(target, prediction),
        "rmse": mean_squared_error(target, prediction, squared=False),
        "mse": mean_squared_error(target, prediction),
        "mape": mean_absolute_percentage_error(target, prediction),
        "r_squared": r2_score(target, prediction),
        "var_score": explained_variance_score(target, prediction),
    }
    if np.equal(target, 0).any():
        regression_metrics.pop("mape")
        logger.warning(
            "MAPE was excluded from regression metrics since target contains zeros.",
        )
    return {name: float(number) for name, number in regression_metrics.items()}
