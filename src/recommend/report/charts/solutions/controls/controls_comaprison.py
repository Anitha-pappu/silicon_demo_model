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

_TRange = tp.Optional[tp.Tuple[float, float]]

_FEATURE_COMPARISON_LEGEND_OFFSET_X = 0.48
_FEATURE_COMPARISON_LEGEND_OFFSET_Y = 1.15
_SUBPLOTS_VERTICAL_SPACING = 0.005

_INITIAL = "initial"
_OPTIMIZED = "optimized"
_COLOR_MAP = {_INITIAL: "grey", _OPTIMIZED: "green"}  # noqa: WPS407
_DIFF = "diff"
_TIMESTAMP = "timestamp"

_HIST_NORM = "percent"

_CONTROL_DOMAIN_LINE = dict(color="red", width=1, dash="dash")


def plot_controls_comparison(
    df_before_after: pd.DataFrame,
    before_column: TColumn = _INITIAL,
    after_column: TColumn = _OPTIMIZED,
    timestamp_column: TColumn = _TIMESTAMP,  # todo: make optional
    control_name: str = "control",
    control_domain: _TRange = None,
) -> go.Figure:
    """
    Returns a figure with three charts: box plot & histogram for before and after
    distributions comparison and a parallel coords plot to see the direction of changes
    for one control

    Args:
        df_before_after: dataframe with two columns before the optimization and after
            for one control
        control_name: the control name (used for histogram x-axis label & hover info)
        before_column: column with control before optimization
        after_column: column with control after optimization
        timestamp_column: columns with timestamp
            (used for showing ts on parallel coords plot hover)
        control_domain: lower and upper bounds of control's domain;
            are shown as red dashed lines on all subplots if provided
    """

    check_data(df_before_after, before_column, after_column)

    df_before_after = df_before_after.copy()
    df_before_after = (
        df_before_after
        # reorder & reset index; helps to reset multi-indexing
        [[timestamp_column, before_column, after_column]]
        .set_axis([_TIMESTAMP, _INITIAL, _OPTIMIZED], axis=1)
    )
    df_before_after[_DIFF] = df_before_after[_OPTIMIZED] - df_before_after[_INITIAL]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"rowspan": 2, "type": "xy", "secondary_y": True}],
            [{}, None],
        ],
        vertical_spacing=_SUBPLOTS_VERTICAL_SPACING,
        row_heights=[0.2, 0.8],
        column_widths=[0.6, 0.4],
    )
    _add_histogram(fig, df_before_after, control_name, control_domain)
    _add_par_coord_plot(df_before_after, fig, control_domain)

    fig.update_yaxes(showline=True, linecolor="black")
    fig.update_layout(
        title="Control Comparison<br><sup>Before and After Optimization</sup>",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="overlay",
        legend=dict(
            title_text=None,
            orientation="h",
            x=_FEATURE_COMPARISON_LEGEND_OFFSET_X,
            y=_FEATURE_COMPARISON_LEGEND_OFFSET_Y,
            xanchor="right",
            yanchor="top",
        ),
    )
    return fig


def _add_histogram(
    fig: go.Figure,
    df_before_after: pd.DataFrame,
    control_name: str,
    control_domain: _TRange,
) -> None:
    control_hover = f"{control_name}=" "%{x}<br>" if control_name is not None else ""
    for type_ in (_INITIAL, _OPTIMIZED):
        fig.add_histogram(
            x=df_before_after[type_],
            marker=dict(color=_COLOR_MAP[type_], opacity=0.5),
            showlegend=True,
            legendgroup=type_,
            hovertemplate=(
                f"type={type_}<br>"
                f"{control_hover}"
                f"{_HIST_NORM}=" "%{y}"
                "<extra></extra>"
            ),
            histnorm=_HIST_NORM,
            name=type_,
            col=1,
            row=2,
        )
        fig.add_box(
            x=df_before_after[type_],
            marker=dict(color=_COLOR_MAP[type_], opacity=0.5),
            showlegend=False,
            legendgroup=type_,
            notched=True,
            name=type_,
            col=1,
            row=1,
        )

    if control_domain is not None:
        for bound in control_domain:
            for row in (1, 2):
                fig.add_shape(
                    x0=bound,
                    x1=bound,
                    y0=0,
                    y1=1,
                    row=row,
                    col=1,
                    yref="y domain",
                    line=_CONTROL_DOMAIN_LINE,
                )

    fig.update_xaxes(row=1, col=1, matches="x3", showticklabels=False)
    fig.update_xaxes(
        col=1,
        showline=True,
        linecolor="black",
        hoverformat=".2f",
    )
    fig.update_xaxes(row=2, col=1, title=control_name)
    fig.update_yaxes(row=1, col=1, showticklabels=False)
    fig.update_yaxes(
        row=2,
        col=1,
        title=f"Histogram<br>histnorm={_HIST_NORM}",
        hoverformat=".2f",
    )


def _add_par_coord_plot(
    df_before_after: pd.DataFrame, fig: go.Figure, control_domain: _TRange,
) -> None:
    for row_index, row in df_before_after.iterrows():
        fig.add_scattergl(
            x=[0, 1],
            y=[
                row[_INITIAL],
                row[_OPTIMIZED],
            ],
            showlegend=False,
            mode="lines+markers",
            customdata=df_before_after.loc[
                [row_index, row_index],
                [_TIMESTAMP, _INITIAL, _OPTIMIZED, _DIFF],
            ].values,
            hovertemplate=(
                "timestamp: %{customdata[0]}<br>"
                "initial: %{customdata[1]:.2f}<br>"
                "optimized: %{customdata[2]:.2f}<br>"
                "diff: %{customdata[3]:.2f}<br>"
                "<extra></extra>"
            ),
            marker_color="rgba(189, 195, 199, 1)",
            line=dict(color="rgba(189, 195, 199, 0.5)", width=1),
            row=1,
            col=2,
        )
    # make the secondary axis visible
    fig.add_scatter(x=[], y=[], secondary_y=True, row=1, col=2)
    # style lines
    fig.update_xaxes(
        row=1,
        col=2,
        showticklabels=False,
        range=(0, 1),
        title="Changes' Trends",
    )

    if control_domain is not None:
        for bound in control_domain:
            fig.add_shape(
                x0=0,
                x1=1,
                y0=bound,
                y1=bound,
                row=1,
                col=2,
                xref="x domain",
                line=_CONTROL_DOMAIN_LINE,
            )

    fig.update_yaxes(row=1, col=2, title=_INITIAL.capitalize())
    fig.update_yaxes(
        row=1,
        col=2,
        secondary_y=True,
        matches="y2",
        showline=True,
        title=_OPTIMIZED.capitalize(),
    )
