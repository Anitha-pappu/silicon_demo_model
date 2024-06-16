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

# fmt: off

import typing as tp
from enum import Enum

import pandas as pd
import plotly.graph_objects as go

from reporting.charts.primitives import plot_lines

from ._utils import add_time_step_column  # noqa: WPS436


class TrendType(Enum):
    RANGE = "range"
    OVERLAY = "overlay"


# Ok to ignore WPS211 for plot functions with many plot options
def plot_one_sensor_trend(  # noqa: WPS211
    sensor_data: pd.DataFrame,
    sensor_to_plot: str,
    datetime_col: str,
    batch_id_col: str,
    trend_type: TrendType,
    time_unit: str = "__time_step",
    drop_nan_readings: bool = False,
    hue: tp.Optional[str] = None,
    alpha: tp.Optional[float] = 0.5,
    color_map: tp.Optional[tp.Dict[str, str]] = None,
    x_lim_quantile: tp.Optional[float] = 1.0,
    error_method: tp.Optional[str] = "ci",
    error_level: tp.Optional[int] = 95,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
) -> go.Figure:
    if drop_nan_readings:
        sensor_data = sensor_data.dropna(subset=[sensor_to_plot])
    sensor_data = add_time_step_column(
        sensor_data=sensor_data,
        timestamp=datetime_col,
        batch_id_col=batch_id_col,
        time_step_column=time_unit,
    )
    x_lim_min, x_lim_max = _get_x_limit(
        sensor_data, batch_id_col, time_unit, x_lim_quantile,
    )
    fig = plot_lines(
        data=sensor_data,
        x=time_unit,
        y=sensor_to_plot,
        color=hue,
        error_method=error_method,
        error_level=error_level,
        opacity=alpha,
        color_map=color_map,
        title=f"sensor {trend_type.value} for {sensor_to_plot}",
        layout_params=dict(xaxis_range=(x_lim_min, x_lim_max)),
        height=height,
        width=width,
        **_configure_plot_params(trend_type, batch_id_col),
    )
    fig.update_xaxes(title="time step", selector=-1)
    return fig


def _get_x_limit(
    sensor_data: pd.DataFrame,
    batch_id_col: str,
    time_unit: str,
    x_lim_quantile: float,
) -> tp.Tuple[float, float]:
    x_lim_max = (
        sensor_data.groupby(batch_id_col)[time_unit].max().quantile(q=x_lim_quantile)
    )
    x_lim_min = sensor_data.groupby(batch_id_col)[time_unit].min().min()
    return x_lim_min, x_lim_max


def _configure_plot_params(
    trend_type: TrendType, batch_id_col: str,
) -> tp.Dict[str, tp.Optional[str]]:
    if trend_type is TrendType.OVERLAY:
        return dict(estimator=None, units=batch_id_col)
    elif trend_type is TrendType.RANGE:
        return dict(estimator="mean", units=None)
    raise NotImplementedError(f"Unknown trend type: {trend_type}")
