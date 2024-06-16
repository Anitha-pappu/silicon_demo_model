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
from plotly import graph_objects as go

from modeling.report.charts._histogram_utils import (  # noqa: WPS436
    add_histogram_trace,
)

from ...configs import LayoutConfig
from ..text_components import (
    CURRENT_SLIDER_POSITION_LEGEND,
    SAVED_SLIDER_POSITION_LEGEND,
)

_TRange = tp.Tuple[float, float]


def add_saved_sensitivity_traces(
    fig: go.Figure,
    parameters_to_plot: tp.Iterable[str],
    layout_config: LayoutConfig,
) -> None:
    for plot_parameter_index, _ in enumerate(parameters_to_plot):
        row = plot_parameter_index // layout_config.max_plots_in_row + 1
        col = plot_parameter_index % layout_config.max_plots_in_row + 1
        showlegend = layout_config.show_legend and row == 1 and col == 1
        fig.add_trace(
            row=row,
            col=col,
            trace=go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                marker={"color": "black", "size": 3},
                line={"color": "black", "width": 1},
                showlegend=showlegend,
                name=SAVED_SLIDER_POSITION_LEGEND,
                legendgroup="saved_sensitivity",
                visible=layout_config.save_slider_position,
            ),
            secondary_y=False,
        )


def add_manipulable_sensitivity_traces(
    fig: go.Figure,
    model_predictions: tp.Iterable[str],
    layout_config: LayoutConfig,
) -> None:
    for plot_parameter_index, _ in enumerate(model_predictions):
        row = plot_parameter_index // layout_config.max_plots_in_row + 1
        col = plot_parameter_index % layout_config.max_plots_in_row + 1
        showlegend = layout_config.show_legend and row == 1 and col == 1
        fig.add_trace(
            row=row,
            col=col,
            trace=go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                marker={"color": "#2a9cc1"},
                legendgroup="manipulable_sensitivity",
                name=CURRENT_SLIDER_POSITION_LEGEND,
                showlegend=showlegend,
            ),
            secondary_y=False,
        )


def add_histogram_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    parameters_to_plot: tp.Iterable[str],
    max_plots_in_row: int,
    visible: bool = True,
) -> None:
    for parameter_idx, plot_parameter in enumerate(parameters_to_plot):
        row = parameter_idx // max_plots_in_row + 1
        col = parameter_idx % max_plots_in_row + 1
        add_histogram_trace(
            fig,
            feature_values=data[plot_parameter],
            row=row,
            column=col,
            visible=visible,
        )
