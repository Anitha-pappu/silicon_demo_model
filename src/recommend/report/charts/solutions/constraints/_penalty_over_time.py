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

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from ._summary_table import TOTAL_PENALTY, TOTAL_VIOLATIONS  # noqa: WPS436

FEATURE_COMPARISON_LEGEND_OFFSET_X = 1.0
FEATURE_COMPARISON_LEGEND_OFFSET_Y = 1.05

penalty_colors = qualitative.Dark24
COLOR_PENALTY_TOTAL = "darkorange"
COLOR_PENALTY_COUNT = "darkblue"


def plot_penalty_over_time(
    df_by_index_penalties: pd.DataFrame,
) -> go.Figure:
    fig = make_subplots(1, 1, specs=[[{"type": "xy", "secondary_y": True}]])

    fig.add_scatter(
        x=df_by_index_penalties.index,
        y=df_by_index_penalties[TOTAL_PENALTY],
        name="total_penalty",
        marker_color=COLOR_PENALTY_COUNT,
    )
    fig.add_scatter(
        x=df_by_index_penalties.index,
        y=df_by_index_penalties[TOTAL_VIOLATIONS],
        name="n_penalties_violated",
        marker_color=COLOR_PENALTY_TOTAL,
        secondary_y=True,
    )

    penalties = df_by_index_penalties.columns.drop([TOTAL_VIOLATIONS, TOTAL_PENALTY])
    for index, penalty in enumerate(penalties):
        penalty_color = penalty_colors[index % len(penalty_colors)]
        fig.add_scatter(
            x=df_by_index_penalties.index,
            y=df_by_index_penalties[penalty],
            name=penalty,
            marker_color=penalty_color,
            visible="legendonly",
        )

    fig.update_xaxes(showline=True, linecolor="black")
    fig.update_yaxes(showline=True, linecolor=COLOR_PENALTY_COUNT)

    max_violations = df_by_index_penalties[TOTAL_VIOLATIONS].max()
    discrete_values_to_show_for_count = sorted({
        0, *df_by_index_penalties[TOTAL_VIOLATIONS], max_violations + 1,
    })
    fig.update_yaxes(
        showline=True,
        linecolor=COLOR_PENALTY_TOTAL,
        secondary_y=True,
        range=[-0.5, max_violations + 1],
        tickvals=discrete_values_to_show_for_count,
    )

    fig.update_layout(
        title="Penalty Over Time",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="overlay",
        legend=dict(
            title_text=None,
            orientation="v",
            x=FEATURE_COMPARISON_LEGEND_OFFSET_X,
            y=FEATURE_COMPARISON_LEGEND_OFFSET_Y,
            xanchor="left",
            yanchor="top",
        ),
    )
    return fig
