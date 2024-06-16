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
import plotly.graph_objects as go

from ...charts.batchplot import TrendType, plot_one_sensor_trend


# Ok to ignore WPS211 for plot functions with many plot options
def plot_sensor_trends(  # noqa: WPS211
    batch_meta: pd.DataFrame,
    sensor_data: pd.DataFrame,
    sensors_to_plot: tp.List[str],
    datetime_col: str,
    batch_id_col: str,
    trend_type: str,
    time_unit: str = "__time_step",  # todo: remove this argument
    drop_nan_readings: bool = False,
    hue: tp.Optional[str] = None,
    alpha: tp.Optional[float] = 0.5,
    color_map: tp.Optional[tp.Dict[str, str]] = None,
    x_lim_quantile: tp.Optional[float] = 1.0,
    error_method: tp.Optional[str] = "ci",
    error_level: tp.Optional[int] = 95,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
) -> tp.Dict[str, go.Figure]:
    """Create sensor trends plot for batch analytics, chart type includes range or
    overlay.

    Args:
        batch_meta: metadata with batch level information
        sensor_data: sensors' time series; each timestamp is assigned to a batch
        sensors_to_plot: the list of sensors to plot
        datetime_col: name of the column storing timestamp in the data
        batch_id_col: the column in batches_time_series to use as batch id
        trend_type: either "range" or "overlay" - the type of sensor trend to plot
        time_unit: the column of batches_time_series to use on the x-axis
        drop_nan_readings: if true, then nan values from each sensor will be ommited
        hue: the column of batches_header to use as hue on the plot
        alpha: the transparency for the line in the plot
        color_map: mapping from `color` column groups to line color
        x_lim_quantile: sets a limit on the x-axis using a quantile approach
        error_method: sets the method to determine the size of the confidence interval
                      to draw when aggregating with an estimator
        error_level: level of the confidence interval
        height: the height of the exported image in layout pixels
        width: the width of the exported image in layout pixels

    Returns: sensor trend plots
    """

    sensor_data = sensor_data.copy()
    validated_trend_type = _validate_trend_type(trend_type)
    sensor_data = sensor_data.merge(batch_meta, how="right", on=batch_id_col)
    _validate_columns(sensor_data, sensors_to_plot, datetime_col, batch_id_col, hue)

    all_sensor_trends_fig = {}
    for sensor in sensors_to_plot:
        fig = plot_one_sensor_trend(
            sensor_data=sensor_data,
            sensor_to_plot=sensor,
            datetime_col=datetime_col,
            batch_id_col=batch_id_col,
            trend_type=validated_trend_type,
            time_unit=time_unit,
            drop_nan_readings=drop_nan_readings,
            hue=hue,
            alpha=alpha,
            color_map=color_map,
            x_lim_quantile=x_lim_quantile,
            error_method=error_method,
            error_level=error_level,
            height=height,
            width=width,
        )
        all_sensor_trends_fig[f"sensor_trend_{sensor}"] = fig
    return all_sensor_trends_fig


def _validate_trend_type(requested_trend_type: str) -> TrendType:
    try:
        trend_type = TrendType(requested_trend_type)
    except ValueError:
        allowed_trend_types = [trend.value for trend in TrendType]
        raise ValueError(
            f"`trend_type` must be one of: {allowed_trend_types}",
        )
    return trend_type


def _validate_columns(
    sensor_data: pd.DataFrame,
    sensors_to_plot: tp.List[str],
    datetime_col: str,
    batch_id_col: str,
    hue: tp.Optional[str],
) -> None:
    requested_columns = [
        *sensors_to_plot,
        datetime_col,
        batch_id_col,
    ]
    if hue is not None:
        requested_columns.append(hue)
    missing_columns = set(requested_columns).difference(sensor_data.columns)
    if missing_columns:
        raise ValueError(
            f"Found missing columns that are passed as plotting arguments: "
            f"{missing_columns}",
        )
