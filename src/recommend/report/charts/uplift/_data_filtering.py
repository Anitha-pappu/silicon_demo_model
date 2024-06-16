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

from reporting.charts.primitives import plot_bar, plot_box
from reporting.charts.utils import apply_chart_style

_CONNECTOR_COLOR = "#000000"
_PERFORMANCE_COLOR_SCALE = [  # noqa: WPS407
    "#FFB000",
    "#FE6100",
    "#648FFF",
    "#DC267F",
    "#785EF0",
]
_AREA_COLOR_SCALE = [  # noqa: WPS407
    "#CC6677",
    "#332288",
    "#DDCC77",
    "#117733",
    "#88CCEE",
    "#882255",
    "#44AA99",
    "#999933",
    "#AA4499",
]
_TOTAL_COLOR = "#40B0A6"
_DECREASING_COLOR = "#FFB000"
_PLOT_HEIGHT = 400
_PLOT_WIDTH = 1000
_TITLE_SIZE = 20
_TEXT_SIZE = 12


def clean_data_filtering(
    data_filtering: pd.DataFrame,
    uptime_column: str = "uptime",
    reviewed_column: str = "reviewed",
    approved_column: str = "approved",
    implemented_column: str = "implemented",
) -> pd.DataFrame:
    """
    Ensures that data provided is consistent.
        - For an observation to be reviewed, it needs to be provided during uptime.
        - For an observation to be approved, it needs to have been reviewed.
        - For an observation to be implemented, it needs to have been approved.

    Args:
        data_filtering: Dataframe with acceptance ratio of recommendations and operators
            adherence to them.
        uptime_column: Name of the uptime column
        reviewed_column: Name of the reviewed column
        approved_column: Name of the approved column
        implemented_column: Name of the implemented column

    Returns:
        Cleaned dataframe

    """
    data_filtering[reviewed_column] = (
        data_filtering[uptime_column] & data_filtering[reviewed_column]
    )
    data_filtering[approved_column] = (
        data_filtering[reviewed_column] & data_filtering[approved_column]
    )
    data_filtering[implemented_column] = (
        data_filtering[approved_column] & data_filtering[implemented_column]
    )

    return data_filtering


def plot_performance_summary(
    data_filtering: pd.DataFrame,
    uptime_column: str = "uptime",
    reviewed_column: str = "reviewed",
    approved_column: str = "approved",
    implemented_column: str = "implemented",
) -> go.Figure:
    """
    Plots the performance summary.

    Args:
        data_filtering: Dataframe with acceptance ratio of recommendations and operators
            adherence to them.
        uptime_column: Name of the uptime column
        reviewed_column: Name of the reviewed column
        approved_column: Name of the approved column
        implemented_column: Name of the implemented column

    Returns:
        List of performance overview figures

    """

    total = data_filtering.shape[0]
    uptime = data_filtering[data_filtering[uptime_column]].shape[0]
    reviewed = data_filtering[data_filtering[reviewed_column]].shape[0]
    approved = data_filtering[data_filtering[approved_column]].shape[0]
    implemented = data_filtering[data_filtering[implemented_column]].shape[0]

    waterfall_values = [
        total,
        -(total - uptime),
        uptime,
        -(uptime - reviewed),
        reviewed,
        -(reviewed - approved),
        approved,
        -(approved - implemented),
        implemented,
    ]

    waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=[
            "absolute",
            "relative",
            "absolute",
            "relative",
            "absolute",
            "relative",
            "absolute",
            "relative",
            "absolute",
        ],
        x=[
            "Total",
            "Downtime",
            "Uptime",
            "Not reviewed",
            "Reviewed",
            "Rejected",
            "Approved",
            "Not implemented",
            "Implemented",
        ],
        text=[str(abs(col)) for col in waterfall_values],
        y=waterfall_values,
        connector={"line": {"color": _CONNECTOR_COLOR}},
        decreasing={"marker": {"color": _DECREASING_COLOR}},
        totals={"marker": {"color": _TOTAL_COLOR}},
        hoverinfo="x+text",
    ))

    apply_chart_style(
        fig=waterfall,
        height=_PLOT_HEIGHT,
        width=_PLOT_WIDTH,
        title="Global performance summary",
        xaxis_title="",
        yaxis_title="# of recommendations",
        layout_params={
            "title_font_size": _TITLE_SIZE,
            "font_size": _TEXT_SIZE,
        },
    )

    return waterfall


def plot_performance_summary_by_tag(
    data_filtering: pd.DataFrame,
    uptime_column: str = "uptime",
    reviewed_column: str = "reviewed",
    approved_column: str = "approved",
    implemented_column: str = "implemented",
    tag_column: str = "tag",
    area_column: str = "area",
) -> list[go.Figure]:
    """
    Generates a list of figures with performance overview by tag. Each figure represents
        one area.

    Args:
        data_filtering: Dataframe with acceptance ratio of recommendations and operators
            adherence to them.
        uptime_column: Name of the uptime column
        reviewed_column: Name of the reviewed column
        approved_column: Name of the approved column
        implemented_column: Name of the implemented column
        tag_column: Name of the tag column
        area_column: Name of the area column

    Returns:
        List of figures with performance overview by tag.

    """
    data_by_tag = data_filtering.groupby(
        [tag_column, area_column], as_index=False,
    )[
        [uptime_column, reviewed_column, approved_column, implemented_column]
    ].mean()
    data_by_tag["Downtime"] = 1 - data_by_tag[uptime_column]
    data_by_tag["Not reviewed"] = (
        data_by_tag[uptime_column] - data_by_tag[reviewed_column]
    )
    data_by_tag["Rejected"] = (
        data_by_tag[reviewed_column] - data_by_tag[approved_column]
    )
    data_by_tag["Not implemented"] = (
        data_by_tag[approved_column] - data_by_tag[implemented_column]
    )
    data_by_tag = data_by_tag.rename(
        {implemented_column: "Implemented"}, axis=1,
    )

    plot_tags = [
        "Downtime",
        "Not reviewed",
        "Rejected",
        "Not implemented",
        "Implemented",
    ]

    areas = list(data_by_tag[area_column].drop_duplicates())
    plots: list[go.Figure] = []
    for area in areas:
        data_area = data_by_tag[data_by_tag[area_column] == area]
        data_area = data_area[
            [tag_column] + plot_tags
        ].melt(id_vars=[tag_column])
        fig = plot_bar(
            data=data_area,
            x=tag_column,
            y="value",
            color="variable",
            color_discrete_map=_PERFORMANCE_COLOR_SCALE,
            orientation="v",
            barmode="stack",
            title=f"Performance summary for {area}",
            xaxis_title="Tag",
            yaxis_title="% of recommendations",
            height=_PLOT_HEIGHT,
            width=_PLOT_WIDTH,
            fig_params={
                "hovertemplate": "variable=%{fullData.name}"
                    "<br>tag=%{label}"
                    "</br>value=%{value:.0%}"
                    "<extra></extra>",
            },
            layout_params={
                "title_font_size": _TITLE_SIZE,
                "font_size": _TEXT_SIZE,
                "legend_title_text": "",
                "yaxis_tickformat": ".0%",
                "yaxis": {
                    'categoryorder': 'array',
                    'categoryarray': plot_tags,
                },
            },
        )

        plots = plots + [fig]

    return plots


def plot_implementation_ratio(
    data_filtering: pd.DataFrame,
    approved_column: str = "approved",
    implementation_status_column: str = "implementation_status",
    tag_column: str = "tag",
    area_column: str = "area",
) -> go.Figure:
    """
    Generates a figure with implementation ratio of recommendations by tag.

    Args:
        data_filtering: Dataframe with acceptance ratio of recommendations and operators
            adherence to them.
        approved_column: Name of the approved column
        implementation_status_column: Name of the implementation status column
        tag_column: Name of the tag column
        area_column: Name of the area column

    Returns:
        Figure with acceptance ratio of recommendations.

    """
    data_approved = data_filtering[data_filtering[approved_column]].copy()

    return plot_box(
        data=data_approved,
        x=tag_column,
        y=implementation_status_column,
        color=area_column,
        color_discrete_map=_AREA_COLOR_SCALE,
        title="Implementation status of approved recommendations by tag",
        xaxis_title="Tag",
        yaxis_title="Implementation status",
        height=_PLOT_HEIGHT,
        width=_PLOT_WIDTH,
        fig_params={
            "yhoverformat": ".2f",
        },
        layout_params={
            "title_font_size": _TITLE_SIZE,
            "font_size": _TEXT_SIZE,
            "legend_title_text": "Area",
        },
    )
