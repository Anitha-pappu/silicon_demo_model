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


import typing as tp

import pandas as pd

from ..model_base import ModelBase

MODEL_PREDICTION = "model_prediction"
ERROR = "error"


def calculate_model_predictions(
    data: pd.DataFrame,
    model: ModelBase,
    keep_input: bool = False,
    add_error: bool = False,
    target_column: tp.Optional[str] = None,
    **predict_kwargs: tp.Any,
) -> pd.DataFrame:
    """Append predictions for the given model to the provided static_features.

    Args:
        data: dataset for making predictions with model
        model: trained instance of ``ModelBase``
        keep_input: if True, keep the input data in the output DataFrame
        add_error: if True, add a column with the error of the prediction
        target_column: name of the target column in the data
        predict_kwargs: keyword arguments to predict function.

    Returns:
        Copy of DataFrame with predictions.

    Raises:
        ValueError: if ``target_column`` is None and ``add_error`` is True
    """
    predictions = model.predict(data, **predict_kwargs)

    if add_error:
        if target_column is None:
            raise ValueError("target_column must be provided to calculate error")
        error = predictions - data[target_column]
        predictions_data = pd.DataFrame(
            data={
                MODEL_PREDICTION: predictions,
                ERROR: error,
            },
            index=data.index,
        )
    else:
        predictions_data = pd.DataFrame(
            data={MODEL_PREDICTION: predictions},
            index=data.index,
        )
    if keep_input:
        predictions_data = data.merge(
            predictions_data, left_index=True, right_index=True,
        )

    return predictions_data


def calculate_model_prediction_bounds(
    data: pd.DataFrame,
    model: ModelBase,
    model_metrics: tp.Dict[str, float],
    error_metric: str = "rmse",
    error_multiplier: float = 1.96,
) -> pd.DataFrame:
    """Calculate the upper and lower bounds for the model prediction.

    Args:
        data: dataset for making predictions with model
        model: trained instance of ``ModelBase``
        model_metrics: calculated model performance metrics
        error_metric: the metric to be used for calculating the bounds,
         typically the standard deviation of the model metric on the
         test set
        error_multiplier: the multiplier to be used for error bounds,
         typically 1.96 for a 95% confidence interval

    Returns:
        DataFrame with columns for actual, predicted and upper and lower bounds.
        The lower and upper bounds are calculated using the error_multiplier and
        the error_metric, representing the approximate confidence interval for
        the model predictions.
    """

    table = pd.DataFrame(
        columns=["timestamp", "actuals", "predictions", "lower_bound", "upper_bound"],
    )

    model_error = model_metrics[error_metric]
    table["timestamp"] = data["timestamp"]
    table["actuals"] = data[model.target]
    table["predictions"] = model.predict(data)
    table["lower_bound"] = table["predictions"] - error_multiplier * model_error
    table["upper_bound"] = table["predictions"] + error_multiplier * model_error

    return table
