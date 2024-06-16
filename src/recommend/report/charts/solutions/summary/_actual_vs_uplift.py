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
from plotly.subplots import make_subplots

from reporting.charts.utils import check_data
from reporting.types import TColumn

_HORIZONTAL_SPACING = 0.12

_FEATURE_COMPARISON_RIGHT_LEGEND_OFFSET_X = 1.0
_FEATURE_COMPARISON_LEFT_LEGEND_OFFSET_X = 0.48
_FEATURE_COMPARISON_LEGEND_OFFSET_Y = 1.15


def plot_actual_vs_uplift(
    data: pd.DataFrame,
    actual_target_column: TColumn,
    uplift_column: TColumn = "uplift",
    timestamp_column: TColumn = "timestamp",
    actual_target_with_uplift_column: tp.Optional[TColumn] = None,
    target_name: str = "target",
) -> go.Figure:
    """

    Args:
        data: data frame with timestamp, actual target, its uplift
            and (optional) actual target with uplift
        timestamp_column:
        actual_target_column:
        uplift_column:
        actual_target_with_uplift_column: actual target with optimization uplift;
            if not provided, then calculated as
            ``df[actual_target_column] + df[uplift_column]``
        target_name: target name (used for axes titles)
    """
    check_data(
        data,
        timestamp_column,
        actual_target_column,
        uplift_column,
        actual_target_with_uplift_column,
    )
    actual_target_with_uplift = (
        data[actual_target_column] + data[uplift_column]
        if actual_target_with_uplift_column is None
        else data[actual_target_with_uplift_column]
    )

    fig = make_subplots(1, 2, horizontal_spacing=_HORIZONTAL_SPACING)

    fig.add_scatter(
        x=data[timestamp_column],
        y=data[actual_target_column],
        name="Actual Target<br>Value",
        showlegend=True,
        legendrank=2,
        marker_color="darkblue",
        # todo: update datasets
        legend="legend1",
        row=1,
        col=1,
    )

    fig.add_scatter(
        x=data[timestamp_column],
        y=actual_target_with_uplift,
        name="Actual Target<br>with Uplift",
        marker=dict(color="green", opacity=0.1),
        legend="legend1",
        showlegend=True,
        legendrank=1,
        fill="tonexty",
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=data[actual_target_column],
        y=data[uplift_column],
        mode="markers",
        name="Actual Target<br>vs Uplift",
        marker_color="darkviolet",
        legend="legend2",
        legendrank=0,
        row=1,
        col=2,
    )

    fig.add_shape(
        type="line",
        xref="x domain",
        yref="y2",
        x0=0,
        y0=0,
        x1=1,
        y1=0,
        line=dict(dash="dot", color="black"),
        row=1,
        col=2,
    )

    _update_layout(fig, target_name)
    return fig


def _update_layout(fig: go.Figure, target_name: str) -> None:
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        ticks="outside",
        hoverformat=".2f",
    )
    fig.update_yaxes(
        title=f"{target_name} uplift",
        col=2,
    )
    fig.update_xaxes(
        title=f"actual {target_name}",
        col=2,
    )
    fig.update_yaxes(title=target_name, col=1)
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        ticks="outside",
    )
    fig.update_xaxes(row=1, col=2, matches="y1")
    fig.update_layout(
        title="Actual Target with Optimization Uplift",
        plot_bgcolor="rgba(0,0,0,0)",
        legend1=dict(
            title_text=None,
            orientation="h",
            x=_FEATURE_COMPARISON_LEFT_LEGEND_OFFSET_X,
            y=_FEATURE_COMPARISON_LEGEND_OFFSET_Y,
            xanchor="right",
            yanchor="top",
        ),
        legend2=dict(
            title_text=None,
            orientation="h",
            x=_FEATURE_COMPARISON_RIGHT_LEGEND_OFFSET_X,
            y=_FEATURE_COMPARISON_LEGEND_OFFSET_Y,
            xanchor="right",
            yanchor="top",
        ),
    )
