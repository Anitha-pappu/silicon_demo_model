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

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reporting.charts.utils import check_data
from reporting.types import TColumn

_MIN_HEIGHT = 300

_FEATURE_COMPARISON_RIGHT_LEGEND_OFFSET_X = 1.0
_FEATURE_COMPARISON_LEGEND_OFFSET_Y = 1.15

OPT_COLOR = "blue"
REF_COLOR = "darkorange"

LEGEND_GROUP_OPT = "data to optimize"
LEGEND_GROUP_REF = "reference data"


def plot_features_comparison(
    data: pd.DataFrame,
    columns: tp.Sequence[TColumn],
    reference_data: pd.DataFrame,
    per_row_height: float = 100,
) -> go.Figure:
    check_data(data, *columns)
    check_data(reference_data, *columns)

    show_legend = True
    n_rows = int(math.ceil(len(columns) / 2))
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=columns,
    )
    for column_index, column in enumerate(columns):
        row = column_index // 2 + 1
        col = column_index % 2 + 1
        fig.add_box(
            y=data[column],
            line_color=OPT_COLOR,
            row=row,
            col=col,
            showlegend=show_legend,
            name=LEGEND_GROUP_OPT,
            legendgroup=LEGEND_GROUP_OPT,
        )
        fig.add_box(
            y=reference_data[column],
            line_color=REF_COLOR,
            row=row,
            col=col,
            showlegend=show_legend,
            name=LEGEND_GROUP_REF,
            legendgroup=LEGEND_GROUP_REF,
        )
        show_legend = False
    _update_layout(fig, columns, n_rows, per_row_height)
    return fig


def _update_layout(
    fig: go.Figure,
    columns: tp.Sized,
    n_rows: int,
    per_row_height: float,
) -> None:
    fig.update_layout(
        title="Controls Distributions Comparison",
        plot_bgcolor="rgba(0,0,0,0)",
        height=per_row_height * n_rows + _MIN_HEIGHT,
        legend1=dict(
            title_text=None,
            orientation="h",
            x=_FEATURE_COMPARISON_RIGHT_LEGEND_OFFSET_X,
            y=_FEATURE_COMPARISON_LEGEND_OFFSET_Y,
            xanchor="right",
            yanchor="top",
        ),
    )
    is_even_number_of_cols = len(columns) % 2
    last_row_in_second_col = n_rows - 1
    if is_even_number_of_cols:
        fig.update_xaxes(
            showticklabels=True,
            col=2,
            row=last_row_in_second_col,
        )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        ticks="outside",
        hoverformat=".2f",
    )
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        ticks="outside",
    )
