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


import math
import typing as tp
from types import MappingProxyType

import pandas as pd
from dash import dcc, html
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .... import ModelBase
from ...callbacks_registry import CALLBACK_ID_GRAPH
from ...configs import LayoutConfig
from ...model_predictions import ModelPrediction
from ..text_components import CONTENTS_TEXT_TITLE
from .traces import (
    add_histogram_traces,
    add_manipulable_sensitivity_traces,
    add_saved_sensitivity_traces,
)

CONTENT_STYLE = MappingProxyType({"padding": "2rem 1rem"})

Y_RANGE_OFFSET_MULTIPLIER = 0.05
X_RANGE_OFFSET_MULTIPLIER = 0.05
HORIZONTAL_SPACING_BETWEEN_SUBPLOTS = 0.05
VERTICAL_SPACING_BETWEEN_SUBPLOTS = 0.1
DEFAULT_SUBPLOT_HEIGHT = 450


def create_content(
    model_predictions: tp.Mapping[str, ModelPrediction],
    data: pd.DataFrame,
    model: ModelBase,
    layout_config: LayoutConfig,
) -> html.Div:
    div_style = dict(
        CONTENT_STYLE,
        # dash expects "margin-left" to be the key of the dict
        **{"margin-left": layout_config.sidebar_width},  # noqa: WPS445, WPS517
    )
    return html.Div(
        [
            html.H1(
                [
                    CONTENTS_TEXT_TITLE,
                    html.Span(model.target),
                ],
                style={"fontSize": 25},
            ),
            dcc.Graph(
                id=CALLBACK_ID_GRAPH,
                figure=create_figure(model_predictions, data, layout_config),
            ),
        ],
        style=div_style,
    )


def create_figure(
    model_predictions: tp.Mapping[str, ModelPrediction],
    data: pd.DataFrame,
    layout_config: LayoutConfig,
) -> go.Figure:
    n_subplots = len(model_predictions)
    n_rows = math.ceil(n_subplots / layout_config.max_plots_in_row)
    n_cols = (
        n_subplots if n_subplots < layout_config.max_plots_in_row
        else layout_config.max_plots_in_row
    )
    subplot_titles = [
        layout_config.visualization_mapping[feature]
        if feature in layout_config.visualization_mapping else feature
        for feature in model_predictions.keys()
    ]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_yaxes=layout_config.share_y_axis,
        specs=[
            [{"secondary_y": True} for _ in range(n_cols)]
            for _ in range(n_rows)
        ],
        horizontal_spacing=HORIZONTAL_SPACING_BETWEEN_SUBPLOTS,
        vertical_spacing=VERTICAL_SPACING_BETWEEN_SUBPLOTS,
    )
    add_histogram_traces(
        fig,
        data,
        model_predictions.keys(),
        max_plots_in_row=layout_config.max_plots_in_row,
        visible=layout_config.show_histogram,
    )
    add_saved_sensitivity_traces(fig, model_predictions.keys(), layout_config)
    add_manipulable_sensitivity_traces(fig, model_predictions.keys(), layout_config)
    update_subplots_layout(fig, model_predictions, layout_config)
    return fig


def update_subplots_layout(
    fig: go.Figure,
    model_predictions: tp.Mapping[str, ModelPrediction],
    layout_config: LayoutConfig,
) -> None:
    n_subplots = len(model_predictions)
    n_rows = math.ceil(n_subplots / layout_config.max_plots_in_row)
    y_axis_ranges = _calculate_y_axis_ranges(model_predictions, layout_config)
    x_axis_ranges = _calculate_x_axis_range(model_predictions)
    for plot_parameter_index, plot_parameter in enumerate(model_predictions):
        row = plot_parameter_index // layout_config.max_plots_in_row + 1
        col = plot_parameter_index % layout_config.max_plots_in_row + 1
        fig.update_xaxes(
            showline=True,
            linecolor="black",
            ticks="outside",
            hoverformat=".2f",
            row=row,
            col=col,
            range=x_axis_ranges[plot_parameter],
        )
        fig.update_yaxes(
            showline=True,
            linecolor="black",
            ticks="outside",
            zerolinewidth=2,
            zerolinecolor="lightgrey",
            row=row,
            col=col,
            secondary_y=False,
            side="left",
            showticklabels=True,
            range=y_axis_ranges[plot_parameter],
        )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0},
        height=DEFAULT_SUBPLOT_HEIGHT * n_rows,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.12,
            "xanchor": "right",
            "x": 1,
        },
    )


def _calculate_y_axis_ranges(
    model_predictions: tp.Mapping[str, ModelPrediction],
    layout_config: LayoutConfig,
) -> tp.Dict[str, tp.Tuple[float, float]]:
    y_axis_ranges = {}
    for parameter_to_plot, prediction in model_predictions.items():
        y_axis_ranges[parameter_to_plot] = (
            prediction.prediction_min_value,
            prediction.prediction_max_value,
        )
    if layout_config.share_y_axis:
        y_ranges_for_shared_y_axis = (
            min(range_start for range_start, _ in y_axis_ranges.values()),
            max(range_end for _, range_end in y_axis_ranges.values()),
        )
        y_axis_ranges = {
            parameter_name: y_ranges_for_shared_y_axis
            for parameter_name in model_predictions
        }
    y_axis_ranges_with_offsets = {}
    for parameter_to_plot, (range_start, range_end) in y_axis_ranges.items():  # noqa: WPS440,E501
        offset = (range_end - range_start) * Y_RANGE_OFFSET_MULTIPLIER
        y_axis_ranges_with_offsets[parameter_to_plot] = (
            range_start - offset,
            range_end + offset,
        )
    return y_axis_ranges_with_offsets


def _calculate_x_axis_range(
    model_predictions: tp.Mapping[str, ModelPrediction],
) -> tp.Dict[str, tp.Tuple[float, float]]:
    x_axis_range_with_offsets = {}
    for parameter_to_plot, prediction in model_predictions.items():
        range_min = prediction.feature_to_plot_min_value
        range_max = prediction.feature_to_plot_max_value
        offset = (range_max - range_min) * X_RANGE_OFFSET_MULTIPLIER
        x_axis_range_with_offsets[parameter_to_plot] = (
            range_min - offset,
            range_max + offset,
        )
    return x_axis_range_with_offsets
