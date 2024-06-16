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
from itertools import cycle

import pandas as pd
import plotly.graph_objects as go

from reporting.config import COLORS

from ._profile_layout import get_profile_composition_layout  # noqa: WPS436

logger = logging.getLogger(__name__)

TRange = tp.Tuple[tp.Optional[float], tp.Optional[float]]
TMappingToRange = tp.Dict[str, TRange]


def plot_batch_profile(
    one_batch_sensor_data: pd.DataFrame,
    sensors_to_plot_by_phases: tp.Dict[str, tp.List[str]],
    title: str = "",
    sensors_plot_range: tp.Optional[TMappingToRange] = None,
    time_step_range: TRange = (None, None),
) -> go.Figure:
    """Create a figure with the traces for selected batch sensors

    Args:
        one_batch_sensor_data: sensors' time series for one batch
        sensors_to_plot_by_phases: maps phase to a list of sensor names to plot
        title: the tile to give to the figure
        sensors_plot_range: dictionary of limits for time series plotting
        time_step_range: absolute range for the x-axis

    Returns: plotly figure of a batch profile chart
    """

    # creating our subplots with as many rows as computed for the line plots
    profile_layout, traces_locations = get_profile_composition_layout(
        sensors_to_plot_by_phases,
    )
    fig = go.Figure(layout=profile_layout)

    color_cycle = cycle(COLORS)

    for phase_id, sensors_to_plot in enumerate(sensors_to_plot_by_phases.values()):
        for sensor_id, sensor_name in enumerate(sensors_to_plot):
            _draw_trace_for_one_batch_sensor_data(
                fig,
                one_batch_sensor_data,
                sensor_name,
                sensors_plot_range,
                time_step_range,
                traces_locations,
                phase_id,
                sensor_id,
                color=next(color_cycle),
            )
    # add x-axis label for bottom x-axis
    fig.update_xaxes(title="time step", selector=-1)
    fig.update_layout(title=title, margin=dict(l=0))
    return fig


def _get_sensor_data_for_available_batch_ids(
    batch_id_col: str, batch_meta: pd.DataFrame, sensor_data: pd.DataFrame,
) -> pd.DataFrame:
    missing_batches = set(sensor_data[batch_id_col]) - set(batch_meta.index)
    if missing_batches:
        logger.warning(f"Following batches are missing `batch_meta`: {missing_batches}")

    missing_batch_series = set(batch_meta.index) - set(sensor_data[batch_id_col])
    if missing_batch_series:
        logger.warning(
            f"Following batches are missing in `sensor_data`: {missing_batch_series}",
        )

    return sensor_data.loc[sensor_data[batch_id_col].isin(batch_meta.index)]


def _draw_trace_for_one_batch_sensor_data(
    fig: go.Figure,
    one_batch_sensor_data: pd.DataFrame,
    sensor_name,
    sensors_plot_range,
    time_step_range,
    traces_locations,
    phase_id,
    sensor_id,
    color,
):
    """ Adds a trace with sensor data for one batch to ``fig``

    Modifies ``fig`` in place.
    """
    x_axis, y_axis = traces_locations[phase_id, sensor_id]
    fig.add_trace(
        go.Scatter(
            x=one_batch_sensor_data.index,
            y=one_batch_sensor_data[sensor_name],
            name=sensor_name,
            yaxis=f"y{y_axis}",
            xaxis=f"x{x_axis}",
            marker_color=color,
        ),
    )
    # we use positional index as a selector
    fig.update_yaxes(tickfont=dict(color=color), selector=y_axis - 1)
    fig.update_layout(
        {
            f"yaxis{y_axis}_range": sensors_plot_range.get(sensor_name, None),
            f"xaxis{x_axis}_range": time_step_range,
        },
    )
