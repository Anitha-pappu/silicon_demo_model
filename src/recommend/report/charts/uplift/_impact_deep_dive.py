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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from reporting.charts.primitives import plot_bar, plot_timeline
from reporting.charts.utils import apply_chart_style
from reporting.config import COLORS

_BASELINE_COLOR = "#FFC107"
_ACTUAL_COLOR = "#D81B60"
_OPTIMAL_COLOR = "#1E88E5"
_DEFAULT_COLOR = "#545454"
_CONNECTOR_COLOR = "#000000"
_PLOT_HEIGHT = 400
_PLOT_WIDTH = 1000
_TITLE_SIZE = 20
_TEXT_SIZE = 12
_CUMULATIVE_OPACITY = 0.7


def plot_impact_waterfall(
    impact: pd.DataFrame,
) -> list[tp.Optional[go.Figure]]:
    """
    Generates waterfall plot of impact by groups if there are groups in ``impact``.

    Args:
        impact: Dataframe with impact analysis

    Returns:
        If group column is present, returns a list with the waterfall figure, otherwise
        returns an empty list

    """

    if len(impact) > 1:
        waterfall = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative" for _ in range(impact.shape[0] - 1)] + ["total"],
            x=impact["group"].iloc[np.r_[1:len(impact), 0]],
            text=np.round(impact["uplift"].iloc[np.r_[1:len(impact), 0]], 2),
            y=impact["uplift"].iloc[np.r_[1:len(impact), 0]],
            connector={"line": {"color": _CONNECTOR_COLOR}},
            increasing={"marker": {"color": _DEFAULT_COLOR}},
            decreasing={"marker": {"color": _DEFAULT_COLOR}},
            totals={"marker": {"color": "green"}},
            hoverinfo="x+text",
        ))

        apply_chart_style(
            fig=waterfall,
            height=_PLOT_HEIGHT,
            width=_PLOT_WIDTH,
            title="Impact by group",
            xaxis_title="",
            yaxis_title="Impact",
            layout_params={
                "title_font_size": _TITLE_SIZE,
                "font_size": _TEXT_SIZE,
            },
        )

        return [waterfall]

    return []


def plot_impact_timeline(
    data: pd.DataFrame,
    timestamp_column: str,
) -> go.Figure:
    """
    Generates a figure with impact timeline.

    Args:
        data: Dataframe with baseline, optimal and actual values
        timestamp_column: Name of the timestamp column

    Returns:
        Impact timeline figure

    """

    data_uplift = data[data["value_type"].isin(["actual_uplift"])]
    color = data_uplift["group"] if "group" in data.columns else None
    uplift = px.bar(
        x=data_uplift[timestamp_column],
        y=data_uplift["value"],
        color=color,
        color_discrete_sequence=COLORS if "group" in data.columns else [_DEFAULT_COLOR],
    )
    apply_chart_style(
        fig=uplift,
        title="Uplift",
        xaxis_title="Time",
        yaxis_title="Uplift value",
        height=_PLOT_HEIGHT,
        width=_PLOT_WIDTH,
        layout_params={
            "title_font_size": _TITLE_SIZE,
            "font_size": _TEXT_SIZE,
            "legend_title_text": "Uplift groups",
            "bargap": 0,
        },
    )
    return uplift


def plot_impact_timeline_cumulative(
    data: pd.DataFrame,
    timestamp_column: str,
) -> go.Figure:
    """
    Generates a figure with cumulative impact timeline.

    Args:
        data: Dataframe with baseline, optimal and actual values
        timestamp_column: Name of the timestamp column

    Returns:
        Cumulative impact timeline figures

    """

    data_uplift_cum = data[data["value_type"].isin(["actual_uplift"])]
    data_uplift_cum = data_uplift_cum.sort_values("timestamp")
    if "group" in data.columns:
        data_uplift_cum["value"] = data_uplift_cum.groupby("group")[
            "value"
        ].transform("cumsum")
        data_uplift_cum = (
            data_uplift_cum
            .pivot_table(index="timestamp", columns="group", values="value")
            .ffill()
            .fillna(0)
            .reset_index()
            .melt(id_vars=["timestamp"])
        )
    else:
        data_uplift_cum["value"] = data_uplift_cum["value"].cumsum()

    return plot_bar(
        data=data_uplift_cum,
        x=timestamp_column,
        y="value",
        color="group" if "group" in data.columns else None,
        color_discrete_map=COLORS if "group" in data.columns else [_DEFAULT_COLOR],
        orientation="v",
        barmode="stack",
        title="Cumulative uplift",
        xaxis_title="Time",
        yaxis_title="Cumulative uplift value",
        height=_PLOT_HEIGHT,
        width=_PLOT_WIDTH,
        fig_params={
            "opacity": _CUMULATIVE_OPACITY,
        },
        layout_params={
            "title_font_size": _TITLE_SIZE,
            "font_size": _TEXT_SIZE,
            "legend_title_text": "Uplift groups",
            "bargap": 0,
        },
    )


def plot_objective_values(
    data: pd.DataFrame,
    timestamp_column: str,
) -> go.Figure:
    """
    Generates a figure with objective values.

    Args:
        data: Dataframe with baseline, optimal and actual values
        timestamp_column: Name of the timestamp column

    Returns:
        List of objective values figures

    """
    return plot_timeline(
        data=data[data["value_type"].isin(["baseline", "actual", "optimized"])],
        x=timestamp_column,
        y="value",
        color="value_type",
        color_discrete_map=[_BASELINE_COLOR, _OPTIMAL_COLOR, _ACTUAL_COLOR],
        title="Impact timeline",
        xaxis_title="Time",
        yaxis_title="Objective values",
        height=_PLOT_HEIGHT,
        width=_PLOT_WIDTH,
        layout_params={
            "title_font_size": _TITLE_SIZE,
            "font_size": _TEXT_SIZE,
            "legend_title_text": "Objective type",
        },
    )


def plot_gap_to_optimal(
    data: pd.DataFrame,
    timestamp_column: str,
) -> go.Figure:
    """
    Generates a figure with gap to optimal.

    Args:
        data: Dataframe with baseline, optimal and actual values
        timestamp_column: Name of the timestamp column

    Returns:
        List of gap to optimal figures

    """

    return plot_timeline(
        data=data[data["value_type"] == "gap_to_optimal"],
        x=timestamp_column,
        y="value",
        color_discrete_map=[_DEFAULT_COLOR],
        title="Gap to optimal timeline",
        xaxis_title="Time",
        yaxis_title="Gap",
        height=_PLOT_HEIGHT,
        width=_PLOT_WIDTH,
        layout_params={
            "title_font_size": _TITLE_SIZE,
            "font_size": _TEXT_SIZE,
        },
    )
