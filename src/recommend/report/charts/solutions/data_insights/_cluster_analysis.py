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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reporting.types import TColumn

_RANGE_SLIDE_THICKNESS = 0.05

LEGEND_OFFSET_X = 1.0
LEGEND_OFFSET_Y = 1.0

_COLORS = px.colors.qualitative.Dark24


def plot_cluster_analysis(
    df: pd.DataFrame, cluster_column: TColumn, timestamp_column: TColumn = "timestamp",
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=["Cluster Counts", "Cluster Timeline"],
    )
    # todo: fix flusters with single row (sometimes not shown well by plotly)
    cat_counts = df[cluster_column].groupby(df[cluster_column]).count()
    for cluster_id, cluster in enumerate(cat_counts.index):
        cluster_data = df[df[cluster_column] == cluster]
        cluster_color = _COLORS[cluster_id % len(_COLORS)]
        cluster_name = str(cluster)
        fig.add_bar(
            x=[cluster_name],
            y=[cat_counts[cluster]],
            marker=dict(
                color=cluster_color, line=dict(color="white", width=5),
            ),
            name=cluster_name,
            legendgroup=cluster_name,
            hovertemplate=cluster_name + "count: %{y}",  # noqa: WPS336 (due to alloy)
            showlegend=False,
            row=1,
            col=1,
        )
        fig.add_bar(
            x=cluster_data[timestamp_column],
            y=np.ones(cluster_data.shape[0]),
            marker=dict(color=cluster_color, line_width=0),
            name=f"cluster #{cluster_id}: {cluster}",
            hovertemplate="%{x}<br>" f"cluster #{cluster_id}: {cluster}<extra></extra>",
            legendgroup=cluster_name,
            row=2,
            col=1,
        )

    fig.update_xaxes(
        row=2,
        showline=True,
        rangeslider=dict(visible=True, thickness=_RANGE_SLIDE_THICKNESS),
    )
    fig.update_yaxes(row=2, showline=False, visible=False)

    fig.update_xaxes(showline=True, linecolor="black", row=1)
    fig.update_yaxes(showline=True, linecolor="black", row=1)

    # don't provide `barmode="overlay"` here – brakes representation for some reason
    fig.update_layout(
        title="Cluster Analysis",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0,
        legend=dict(
            title_text=None,
            orientation="v",
            x=LEGEND_OFFSET_X,
            y=LEGEND_OFFSET_Y,
            xanchor="left",
            yanchor="top",
        ),
    )

    return fig
