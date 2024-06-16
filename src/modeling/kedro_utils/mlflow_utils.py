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
"""Functional APIs related to mlflow workflow."""
import typing as tp

_TKedroMLflowMetrics = tp.Dict[
    str,
    tp.Dict[tp.Literal["value", "step"], tp.Optional[float]],
]


def convert_metrics_to_nested_mlflow_dict(
    metrics: tp.Dict[str, float],
    step: tp.Optional[int] = None,
) -> _TKedroMLflowMetrics:
    """
    Converts intuitive ``Dict[str, float]`` metrics produced by OAI APIs
    into a nested `Dict` that `kedro-mlflow` custom Dataset can use.

    From::

        >>> {
        ...     "R2": 0.9,
        ...     "MAE": 42,
        ... }

    To::

        >>> {
        ...     "R2": {
        ...         "value" : 0.9,
        ...         "step" : None,
        ...     },
        ...     "MAE": {
        ...         "value" : 42,
        ...         "step" : None,
        ...     },
        ... }

    The only reason this API exists is that none of ``kedro-mlflow`` custom datasets,
    for some reason, can save the original ``Dict[str, float]`` format. However they
    should, given that ``mlflow.log_metrics()`` works exactly with this  format.

    This function should be used only in kedro workflow and its usage is as follows:
        1. You have a node that produces ``Dict[str, float]`` metrics. All provided
            OAI APIs for metrics, feature importances etc. do that.
        2. If you want to log those metrics to ``MLflow`` using ``kedro-mlflow``,
            you need to add another node after that that wraps current function.
        3. Output of this node should be mentioned and typed in kedro catalog as
            ``kedro_mlflow.io.metrics.MlflowMetricsDataSet``

    Overall, this function would be useless in any other context than described above.

    Args:
        metrics: to log to mlflow via ``kedro_mlflow.io.metrics.MlflowMetricsDataSet``.
        step: a single integer step at which to log the specified metrics.
            If unspecified, each metric is logged at step zero. Default behavior is
            suitable for all cases where metrics are produced once per run. Specifying
            a ``step`` explicitly would make sense, for example, in case of logging
            how neural network loss is reducing as new epochs pass throughout the run.
            See here: https://snyk.io/advisor/python/mlflow/functions/mlflow.log_metric

    Returns:
        Same metrics but converted into a nested dictionary required by `kedro-mlflow`.
    """

    return {
        name: {
            "value": number,
            "step": step,
        }
        for name, number in metrics.items()
    }
