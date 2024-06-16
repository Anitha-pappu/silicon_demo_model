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

import plotly.graph_objects as go

EPSILON_FOR_BORDERS = 1e-4
EXTRA_RANGE_FRACTION = 0.1


def create_layout_for_optimization_explainer(
    optimizable_parameter: str,
    dependent_function_values: tp.Iterable[float],
    initial_dependent_function_value: float,
    dependent_function_name: str,
    x_axis_name: tp.Optional[str] = None,
    y_axis_name: tp.Optional[str] = None,
) -> go.Layout:
    dependent_function_values = (
        list(dependent_function_values)
        + [initial_dependent_function_value]
    )
    x_axis_name = x_axis_name if x_axis_name is not None else optimizable_parameter
    y_axis_name = (
        y_axis_name
        if y_axis_name is not None
        else dependent_function_name
    )
    axis_common_setting = {
        "ticks": "inside",
        "showline": True,
        "linecolor": "black",
    }
    return go.Layout(
        xaxis={"title": x_axis_name, **axis_common_setting},
        yaxis={
            "title": y_axis_name,
            "range": calculate_y_range_limits(dependent_function_values),
            **axis_common_setting,
        },
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            title_text="Repairs' Violations",
        ),
    )


def calculate_y_range_limits(
    dependent_function_values: tp.Iterable[float],
) -> tp.Tuple[float, float]:
    dependent_function_values = list(dependent_function_values)
    dependent_function_min = min(dependent_function_values)
    dependent_function_max = max(dependent_function_values)
    y_range_lower_limit = (
        dependent_function_min
        - (dependent_function_max - dependent_function_min) * EXTRA_RANGE_FRACTION
        - EPSILON_FOR_BORDERS
    )
    y_range_upper_limit = (
        dependent_function_max
        + (dependent_function_max - dependent_function_min) * EXTRA_RANGE_FRACTION
        + EPSILON_FOR_BORDERS
    )
    return y_range_lower_limit, y_range_upper_limit
