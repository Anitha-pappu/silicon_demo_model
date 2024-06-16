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
import warnings

import pandas as pd

from ..solution import Solutions


@tp.runtime_checkable
class ActualValueMethod(tp.Protocol):
    def __call__(
        self,
        data: pd.DataFrame,
        datetime_column: str,
        **kwargs: tp.Any,
    ) -> pd.DataFrame:
        """
        Signature of functions that can be passed in ``actual_value``, where ``kwargs``
        includes parameters required to calculate the actual value.

        Function should return a dataframe with a ``value_after_recs`` column and the
        datetime column.
        """


def get_value_after_recs_impact(
    data: pd.DataFrame,
    datetime_column: str = "timestamp",
    value_after_recs: ActualValueMethod | str = "value_after_recs",
    value_after_recs_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> pd.DataFrame:
    """
    Calculate the value after recommendations are implemented to calculate impact
    estimation.

    Args:
        data: data to calculate the actual value from.
        datetime_column: Name of the column in the data that contains the datetime.
        value_after_recs: If it is a function, it must have ``ActualValueMethod``
            signature. If it is a string, it must be the name of the column in the
            ``data`` that contains the actual value.
        value_after_recs_kwargs: Arguments to pass to the actual value function.

    Returns:
        Dataframe with value after recommendations.

    Raises:
        TypeError: If the actual value type is not valid.

    """
    if isinstance(value_after_recs, str):
        return data[[datetime_column, value_after_recs]].rename(
            {value_after_recs: "value_after_recs"}, axis=1,
        ).copy()
    elif isinstance(value_after_recs, ActualValueMethod):
        return value_after_recs(
            data,
            datetime_column,
            **(value_after_recs_kwargs or {}),
        )
    raise TypeError(
        "Unknown actual_value type. Valid values are complying with "
        "`ActualValueMethod` protocol or a string.",
    )


def get_value_after_recs_counterfactual(  # noqa: WPS231
    counterfactual_type: tp.Literal["predicted", "uplift"],
    solutions: Solutions,
    data: tp.Optional[pd.DataFrame] = None,
    datetime_column: str = "timestamp",
    actual_value: tp.Optional[ActualValueMethod | str] = None,
    actual_value_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> pd.DataFrame:
    """
    Calculate the value after recommendations are implemented to perform counterfactual
    analysis.

    Counterfactual analysis compares the optimization objective with the baseline
    value to get a sense of the impact before implementation. This function calculates
    the optimization objective.

    It can be calculated using two methods. The selection is made using the
    ``counterfactual_type`` argument:
        - 'predicted': It uses the objective used in optimization. Only the
            ``solutions`` from the optimization process are required.
        - 'uplift': It adds to the actual value the uplift in the optimization
            objective. It requires the ``actual_value`` to be provided.
    None of these methods is necessarily more accurate than the other. However, the
    'uplift' method is believed to be more accurate in general, so it should be chosen
    when ``actual_value`` is available.

    Args:
        counterfactual_type: Type of calculation used to calculate the value after
            recommendations.

            - 'predicted': Use the optimization objective
            - 'uplift': Add to the actual value the uplift in the optimization objective
        solutions: Solutions object obtained from the recommendations.
        data: data to calculate the actual value from. Only required if the type is
            'uplift'.
        datetime_column: Name of the column in the data that contains the datetime.
        actual_value: If it is a function, it must have ``ActualValueMethod`` signature.
            If it is a string, it must be the name of the column in the ``data`` that
            contains the actual value. Only required if the type is 'uplift'.
        actual_value_kwargs: Arguments to pass to the actual value function.

    Returns:
        Dataframe with value after recommendations.

    Raises:
        ValueError: If the counterfactual type is not valid.
        ValueError: If no ``actual_value`` or ``data`` are provided when the
            ``counterfactual_type`` is set to 'uplift'.

    """
    if counterfactual_type == "predicted":
        return _get_value_after_recs_predicted(
            solutions=solutions,
            datetime_column=datetime_column,
        )
    elif counterfactual_type == "uplift":
        if actual_value is None:
            raise ValueError(
                "No actual_value is provided for counterfactual analysis using the "
                "'uplift' method.",
            )
        if data is None:
            raise ValueError(
                "No data is provided for counterfactual analysis using the 'uplift' "
                "method.",
            )
        return _get_value_after_recs_uplift(
            data=data,
            solutions=solutions,
            datetime_column=datetime_column,
            actual_value=actual_value,
            actual_value_kwargs=actual_value_kwargs,
        )
    raise ValueError(
        f"Unknown counterfactual type: {counterfactual_type}. "
        f"Valid values are 'predicted' and 'uplift'.",
    )


def _get_value_after_recs_predicted(
    solutions: Solutions,
    datetime_column: str,
) -> pd.DataFrame:
    """
    Calculates the value after recommendations using the optimization objective using
    the 'predicted' method. Its value is the optimized objective obtained in the
    optimization process. It is obtained from the ``solutions`` object.

    Args:
        solutions: Solutions object obtained from the recommendations.
        datetime_column: Name of the column that contains the datetime.

    Returns:
        Dataframe with value after recommendations.

    """
    solutions_df = solutions.to_frame()[
        [(datetime_column, "initial"), ("objective", "optimized")]
    ]
    solutions_df.columns = ["timestamp", "value_after_recs"]
    solutions_df["value_after_recs"] = solutions_df["value_after_recs"].astype(float)

    return solutions_df


def _get_value_after_recs_uplift(
    data: pd.DataFrame,
    solutions: Solutions,
    datetime_column: str,
    actual_value: ActualValueMethod | str,
    actual_value_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> pd.DataFrame:
    """
    Calculates the value after recommendations using the optimization objective using
    the 'uplift' method. Its value is the actual value of the objective (calculated from
    the ``actual_value`` argument) plus the optimization uplift obtained in the
    optimization process (available from the ``solutions`` object).

    Args:
        data: data to calculate the actual value from.
        solutions: Solutions object obtained from the recommendations.
        datetime_column: Name of the column that contains the datetime.
        actual_value: If it is a function, it must have ``ActualValueMethod`` signature.
            If it is a string, it must be the name of the column in the ``data`` that
            contains the actual value.
        actual_value_kwargs: Arguments to pass to the actual value function.

    Returns:
        Dataframe with value after recommendations.

    """
    actual_value_calculated = get_value_after_recs_impact(
        data=data,
        datetime_column=datetime_column,
        value_after_recs=actual_value,
        value_after_recs_kwargs=actual_value_kwargs,
    )
    actual_value_calculated = actual_value_calculated.rename(
        {"value_after_recs": "actual_value"}, axis=1,
    )
    solutions_df = solutions.to_frame()[[
        (datetime_column, "initial"), ("uplift", ""),
    ]]
    solutions_df.columns = [datetime_column, "uplift"]

    value_after_recs = actual_value_calculated.merge(solutions_df, on=datetime_column)
    value_after_recs["value_after_recs"] = (
        value_after_recs["actual_value"] + value_after_recs["uplift"]
    ).astype(float)

    value_after_recs = value_after_recs[[datetime_column, "value_after_recs"]]

    if value_after_recs["value_after_recs"].isna().any():
        warnings.warn(
            message="Some value after recommendations are missing using 'uplift' "
            "method. 'predicted' method will be used to fill them.",
            category=UserWarning,
        )
        value_after_recs = _fix_value_after_recs_uplift(
            value_after_recs=value_after_recs,
            solutions=solutions,
            datetime_column=datetime_column,
        )

    return value_after_recs


def _fix_value_after_recs_uplift(
    value_after_recs: pd.DataFrame,
    solutions: Solutions,
    datetime_column: str,
) -> pd.DataFrame:
    """
    Fills missing values when calculating the counterfactual analysis data using the
    'uplift' method with the values from the 'predicted' method.

    Args:
        value_after_recs: Dataframe with value after recommendations.
        solutions: Solutions object obtained from the recommendations.
        datetime_column: Name of the column that contains the datetime.

    Returns:
        Dataframe with value after recommendations.

    """
    value_after_recs_predict = _get_value_after_recs_predicted(
        solutions=solutions,
        datetime_column=datetime_column,
    )
    value_after_recs_predict = value_after_recs_predict.rename(
        {"value_after_recs": "value_after_recs_predict"}, axis=1,
    )
    value_after_recs = value_after_recs.merge(
        value_after_recs_predict, on=datetime_column, how="left",
    )
    value_after_recs["value_after_recs"] = (
        value_after_recs["value_after_recs"].fillna(
            value_after_recs["value_after_recs_predict"],
        )
    )

    return value_after_recs[[datetime_column, "value_after_recs"]]
