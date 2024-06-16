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

"""
Pairplot functions.
"""

import logging
import math

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from reporting.charts.utils import get_num_lines_for_str
from reporting.config import COLORS

logger = logging.getLogger(__name__)


_TITLES_FONT_SIZE = 12
_SUBPLOT_HEIGHT = 200
_SUBPLOT_WIDTH = 200
_MIN_PLOT_WIDTH = 1000
_MIN_PLOT_HEIGHT = 300
_DEFAULT_LEFT_MARGIN = 80
_EXTRA_LEFT_MARGIN_PER_TARGET_LINE = 10


def plot_focused_pairplot(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    n_columns: int = 6,
) -> go.Figure:
    """
    Plots scatter plots of all variables versus the target

    Args:
        data: dataset with target_column and feature_columns columns
        target_column: target column name
        feature_columns: list of feature column names to plot against the target
        n_columns: number of plots per line

    Returns:
        A plotly figure object containing plots of target versus all other columns.
    """
    if feature_columns is None:
        feature_columns = data.columns.drop(target_column).to_list()
    n_features = len(feature_columns)
    n_rows = math.ceil(n_features / n_columns)
    n_columns = min(n_columns, len(feature_columns))

    fig = make_subplots(
        rows=n_rows,
        cols=n_columns,
        subplot_titles=feature_columns,
        y_title=target_column,
    )
    fig.update_annotations(font_size=_TITLES_FONT_SIZE)

    for num_fig in range(n_features):
        fig.add_trace(
            go.Scatter(
                x=data[feature_columns[num_fig]],
                y=data[target_column],
                mode="markers",
                name=feature_columns[num_fig],
                marker=dict(color=COLORS[num_fig]),
            ),
            row=num_fig // n_columns + 1,
            col=num_fig % n_columns + 1,
        )

    num_lines_target = get_num_lines_for_str(target_column)
    fig.update_layout(
        autosize=False,
        width=max(n_columns * _SUBPLOT_WIDTH, _MIN_PLOT_WIDTH),
        height=max(n_rows * _SUBPLOT_HEIGHT, _MIN_PLOT_HEIGHT),
        margin=dict(
            l=(
                _DEFAULT_LEFT_MARGIN
                + num_lines_target * _EXTRA_LEFT_MARGIN_PER_TARGET_LINE
            ),
        ),
    )

    return fig
