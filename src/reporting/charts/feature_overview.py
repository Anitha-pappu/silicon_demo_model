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
Contains functionality for generating the feature overviews report
`plot_feature_overviews`
"""

import re
import typing as tp

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reporting.charts.utils import (
    add_watermark,
    apply_chart_style,
    check_data,
    get_num_lines_for_str,
    wrap_string,
)
from reporting.config import COLORS

_DEFAULT_TABLE_FONT_SIZE = 11
_DEFAULT_GRAPH_FONT_SIZE = 11
_DEFAULT_ANNOTATION_TEXT_ANGLE = 90
_DEFAULT_X_TITLE_NORMALIZED_POSITION = 0.15
_DEFAULT_FIGURE_HEIGHT = 820
_DEFAULT_FIGURE_WIDTH = 1070
_DEFAULT_OPACITY_MULTIPLE = 0.6
_DEFAULT_NAN_VALUES_BACKGROUND_COLOR = "#E5ECF6"
_DEFAULT_PROP_HEIGHT_MISSING_VALUES_PLOT = 0.14
_DEFAULT_VERTICAL_SUBPLOT_SPACING = 0.12
_DEFAULT_HORIZONTAL_SUBPLOT_SPACING = 0.1
_DEFAULT_EXTRA_SPACING_PER_LINE = 0.015
_DEFAULT_EXTRA_PERCENTAGE_HEIGHT_PER_LINE = 0.04
_DEFAULT_EXTRA_PERCENTAGE_WIDTH_PER_LINE = 0.001
_DEFAULT_SUBPLOT_TITLE_Y_SHIFT = -40
_DEFAULT_EXTRA_TITLE_SHIFT_PER_LINE_PER_FONT = 1.6
_DEFAULT_EXTRA_TITLE_SHIFT_PER_LINE = (
    -_DEFAULT_GRAPH_FONT_SIZE * _DEFAULT_EXTRA_TITLE_SHIFT_PER_LINE_PER_FONT
)

TDict = tp.Dict[str, tp.Any]
TRange = tp.Tuple[float, float]
TData = pd.DataFrame | list[pd.DataFrame] | dict[str, pd.DataFrame]


def plot_feature_overview(
    data: TData,
    feature: str,
    timestamp: str,
    target: tp.Optional[str] = None,
    tag_range: tp.Optional[TRange] = None,
    title: tp.Optional[str] = None,
    labels_length_limit: int = 20,
    fig_params: tp.Optional[TDict] = None,
    layout_params: tp.Optional[TDict] = None,
) -> go.Figure:
    """
    Create plots a collection of plots representing chosen feature "overview".
    This includes a boxplot and histogram to understand the distribution of values,
    a scatter-plot vs the `target` variable, and a time-series plot of the `feature` and
    `target`.

    Args:
        data: data to plot. It can be either a dataframe or a list or dictionary of them
        feature: column name of the chosen feature
        timestamp: column name of the timestamp associated with the feature
        tag_range: range of a feature to show on a plot
        target: column name of the target variable
        title: title of the chart
        labels_length_limit: limits feature name to `name[:feature_name_limit]...`
            in case it's too long
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating
            the plotly fig layout

    Returns:
        plotly feature overview chart

    Raises:
        TypeError: if data is not a pandas.DataFrame, list of pandas.DataFrame or
            dict of pandas.DataFrame
    """

    data = transform_data_to_dict(data, labels_length_limit)

    for data_df in data.values():
        check_data(data_df, timestamp, feature, target)

    feature_compact_name = wrap_string(feature, labels_length_limit)
    target_compact_name = wrap_string(target, labels_length_limit)
    num_lines_feature = get_num_lines_for_str(feature_compact_name)
    num_lines_target = (
        get_num_lines_for_str(target_compact_name) if target is not None else 0
    )

    missing_data = _calculate_missing_data(data, feature, timestamp)

    fig = _generate_subplots(
        num_lines_feature=num_lines_feature,
        num_lines_target=num_lines_target,
    )

    _add_overview_first_row_plots(
        fig,
        data=data,
        feature=feature,
        target=target,
    )

    _add_overview_second_row_plot(
        fig,
        data=data,
        feature=feature,
        timestamp=timestamp,
    )

    _add_overview_third_row_plot(
        fig,
        missing_data=missing_data,
        feature_compact_name=feature_compact_name,
    )

    # NOTE: The ranges _must_ be added
    # - after the plots have been generated
    # - before the table is added
    # This is due to a bug (feature?:D) in plotly, where vlines and hlines can be
    # added in a make_subplots if there are no go.Table present.
    # Otherwise, it generates this cryptic and difficult to debug error.
    # ```
    # _plotly_utils.exceptions.PlotlyKeyError: Invalid property specified for object of
    #  type plotly.graph_objs.Table: 'xaxis'
    #
    # Did you mean "cells"?
    # ````
    # See here https://github.com/plotly/plotly.py/issues/3424
    if tag_range is not None:
        _add_ranges(fig, tag_range, target)

    _add_overview_second_row_table(fig, data, missing_data, feature)

    _update_overview_layout(
        fig,
        title,
        feature,
        feature_compact_name,
        target_compact_name,
        num_lines_feature,
        num_lines_target,
    )
    apply_chart_style(
        fig=fig, title=title, fig_params=fig_params, layout_params=layout_params,
    )
    return fig


def transform_data_to_dict(
    data: TData,
    labels_length_limit: int,
) -> dict[str, pd.DataFrame]:
    """
    Transform data to be plotted to a dictionary of dataframes.
    - If data is a dataframe, it will be transformed to a dictionary with an empty key.
    - If data is a list of dataframes, each dataframe will be transformed to a
        dictionary with a key "data {i + 1}".
    - If data is a dictionary of dataframes, it will be returned as is.

    Args:
        data: data to be transformed
        labels_length_limit: limits data name in case it's too long

    Returns:
        dictionary of dataframes from data

    Raises:
        TypeError: if data is not a pandas.DataFrame, list of pandas.DataFrame or
            dict of pandas.DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return {"": data}
    elif isinstance(data, list):
        return {f"data {pos + 1}": df for pos, df in enumerate(data)}
    if isinstance(data, dict):
        return {
            wrap_string(data_name, labels_length_limit): data_df
            for data_name, data_df in data.items()
        }
    raise TypeError(
        f"Expected data to be a pandas.DataFrame, list of pandas.DataFrame or "
        f"dict of pandas.DataFrame, got {type(data)}",
    )


def _generate_subplots(
    num_lines_feature: int, num_lines_target: int,
) -> go.Figure:
    """
    Generates a plotly figure with the subplots for the feature overview.

    Args:
        num_lines_feature: number of lines the feature name has
        num_lines_target: number of lines the target name has

    Returns:
        plotly figure with the subplots for the feature overview

    """
    prop_height_first_second_row_plots = (
        (1 - _DEFAULT_PROP_HEIGHT_MISSING_VALUES_PLOT) / 2
    )
    return make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{}, {}, {}],
            [
                {"colspan": 2, "secondary_y": True},
                None,
                {"colspan": 1, "rowspan": 2, "type": "table"},
            ],
            [{"colspan": 2}, None, None],
        ],
        subplot_titles=(
            "Boxplot",
            "Histogram",
            "Scatter vs target",
            "Timeseries",
            "Summary statistics",
            "Missing values",
        ),
        column_widths=[0.33, 0.33, 0.33],
        row_heights=[
            prop_height_first_second_row_plots,
            prop_height_first_second_row_plots,
            _DEFAULT_PROP_HEIGHT_MISSING_VALUES_PLOT,
        ],
        vertical_spacing=(
            _DEFAULT_VERTICAL_SUBPLOT_SPACING
            + num_lines_feature * _DEFAULT_EXTRA_SPACING_PER_LINE
        ),
        horizontal_spacing=(
            _DEFAULT_HORIZONTAL_SUBPLOT_SPACING
            + num_lines_target * _DEFAULT_EXTRA_SPACING_PER_LINE
        ),
    )


def _generate_stats(
    data: dict[str: pd.DataFrame],
    missing_data: pd.DataFrame,
    feature: str,
    precision: int = 4,
) -> pd.DataFrame:
    """Generate descriptive statistics of a feature

    Args:
        data: dataframe that contains the feature data
        missing_data: dataframe with missing data information
        feature: name of the feature
        precision: number of decimal places to round each column to

    Returns: summary statistics of the feature provided.
    """
    data_stats = {}
    for data_name in data.keys():
        data_stats[data_name] = data[data_name][feature].describe().round(precision)
        data_stats[data_name] = pd.concat([
            data_stats[data_name],
            pd.Series(
                data=[
                    (100 * missing_data[data_name].mean())
                    .round(precision)
                    .astype("str") + "%",
                ],
                index=["missing"],
            ),
        ])
    stats = pd.DataFrame(
        data_stats,
        index=data_stats[list(data.keys())[0]].index,
    )
    if len(data) == 1:
        stats.columns = ["value"]
    return stats.reset_index().rename(columns={"index": "statistic"})


def _add_range(
    fig: go.Figure,
    row: int,
    col: int,
    tag_range: TRange,
    direction: str = "h",
    **kwargs: tp.Any,
) -> None:
    """This function adds range_min/max to plotly figure.

    Args:
        fig: plotly graph figure object
        row: row of subplot to add range to
        col: col of subplot to add range to
        tag_range: minimum and maximum value to show on the plot
        direction: direction of lines. Can only be "h" or "v". Defaults to "h".
        **kwargs: all kwargs are passed to `add_hline` / `add_vline`

    Raises:
        ValueError: raises when direction is neither "h" nor "v"
    """
    range_min, range_max = tag_range

    if direction not in {"h", "v"}:
        raise ValueError("direction must be either 'h' or 'v'")

    line_config = dict(line_width=1, line_color="red", line_dash="dash")

    if not np.isnan(range_min) and not np.isinf(range_min):
        annotation_text = "min"
        if direction == "h":
            fig.add_hline(
                y=range_min,
                row=row,
                col=col,
                annotation_text=annotation_text,
                annotation_position="top right",
                **line_config,
                **kwargs,
            )
        else:
            fig.add_vline(
                x=range_min,
                row=row,
                col=col,
                annotation_text=annotation_text,
                annotation_textangle=_DEFAULT_ANNOTATION_TEXT_ANGLE,
                **line_config,
                **kwargs,
            )

    if not np.isnan(range_max) and not np.isinf(range_max):
        annotation_text = "max"
        if direction == "h":
            fig.add_hline(
                y=range_max,
                row=row,
                col=col,
                annotation_text=annotation_text,
                **line_config,
                annotation_position="bottom left",
                **kwargs,
            )
        else:
            fig.add_vline(
                x=range_max,
                row=row,
                col=col,
                annotation_text=annotation_text,
                **line_config,
                annotation_position="top left",
                annotation_textangle=_DEFAULT_ANNOTATION_TEXT_ANGLE,
                **kwargs,
            )


def _add_overview_first_row_plots(
    fig: go.Figure,
    data: dict[str, pd.DataFrame],
    feature: str,
    target: tp.Optional[pd.Series] = None,
) -> None:
    """
    Makes the plots for the first row of the overview plot in ``fig``.

    These are:
    - box plot of the feature values
    - histogram of the feature value
    - scatterplot to explore the relationship between feature and target
        - replace by a watermark saying that target was not provided if this is the case

    Args:
        fig: plotly figure object
        data: dictionary of dataframes containing the data to plot
        feature: name of the feature
        target: series containing the target data
    """
    opacity = _DEFAULT_OPACITY_MULTIPLE if len(data) > 1 else 1
    for pos, (data_name, data_df) in enumerate(data.items()):
        fig.add_trace(
            go.Box(
                y=data_df[feature],
                name=data_name,
                marker_color=COLORS[pos],
                legendgroup=data_name,
                showlegend=len(data) > 1,
            ),
            row=1,
            col=1,
        )
        # todo: use nice distplot
        fig.add_trace(
            go.Histogram(
                x=data_df[feature],
                name=data_name,
                marker_color=COLORS[pos],
                opacity=opacity,
                legendgroup=data_name,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(barmode="overlay")

        if target is not None:
            fig.add_trace(
                go.Scattergl(
                    x=data_df[feature],
                    y=data_df[target],
                    name=data_name,
                    mode="markers",
                    marker_color=COLORS[pos],
                    opacity=opacity,
                    legendgroup=data_name,
                    showlegend=False,
                ),
                row=1,
                col=3,
            )
        else:
            add_watermark(fig, message="No target provided", xref="x3", yref="y3")


def _add_overview_second_row_table(
    fig: go.Figure,
    data: dict[str, pd.DataFrame],
    missing_data: pd.DataFrame,
    feature: str,
) -> None:
    """
    Adds the table to the second row of ``fig``. This table is a table with a summary
    of the stats about the feature

    Args:
        fig: plotly figure object
        data: dictionary of dataframes containing the data to plot
        missing_data: dataframe with missing data information
        feature: name of the feature

    """

    stats = _generate_stats(data, missing_data, feature)
    # todo: update used table in the view
    fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{col}</b>" for col in stats.columns],
                fill_color="LightGray",
                line_color="black",
                align="left",
                font=dict(color="black", size=_DEFAULT_TABLE_FONT_SIZE),
            ),
            cells=dict(
                values=stats.T,
                fill_color="white",
                line_color="black",
                align="left",
                font=dict(color="black", size=_DEFAULT_TABLE_FONT_SIZE),
            ),
        ),
        row=2,
        col=3,
    )


def _add_overview_second_row_plot(
    fig: go.Figure,
    data: dict[str, pd.DataFrame],
    feature: str,
    timestamp: str,
) -> None:
    """
    Makes the plot for the second row of the overview plot and adds it to ``fig``.

    This plot is a scatter plot of the timeseries of the feature values.

    Args:
        fig: plotly figure object
        data: dictionary of dataframes containing the data to plot
        feature: name of the feature
        timestamp: name of the timestamp column

    """

    for pos, (data_name, data_df) in enumerate(data.items()):
        fig.add_trace(
            go.Scatter(
                x=data_df[timestamp],
                y=data_df[feature],
                name=data_name,
                mode="lines",
                marker_color=COLORS[pos],
                opacity=_DEFAULT_OPACITY_MULTIPLE if len(data) > 1 else 1,
                legendgroup=data_name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )


def _calculate_missing_data(
    data: dict[str, pd.DataFrame],
    feature: str,
    timestamp: str,
) -> pd.DataFrame:
    """
    Transforms dataframes in data into a dataframe with the timestamp as index and the
    name of the dataframe as columns. The values are True if the feature is missing and
    False otherwise.

    Args:
        data: dictionary of dataframes containing the data to plot
        feature: name of the feature
        timestamp: name of the timestamp column

    Returns:
        dataframe with missing data information

    """

    timestamp_list = [
        individual_timestamp
        for data_df in data.values()
        for individual_timestamp in pd.to_datetime(data_df[timestamp])
    ]
    timestamp_list = sorted(set(timestamp_list))
    freq = np.diff(timestamp_list).min()
    missing_data = pd.DataFrame(
        {timestamp: pd.date_range(timestamp_list[0], timestamp_list[-1], freq=freq)},
    )
    for data_name in list(data.keys()):
        missing_data_name = (
            data[data_name][[timestamp, feature]]
            .rename({feature: data_name}, axis=1)
        )
        missing_data_name[timestamp] = pd.to_datetime(missing_data_name[timestamp])
        missing_data = missing_data.merge(
            missing_data_name,
            on=timestamp,
            how="outer",
        )
    missing_data = missing_data.set_index(timestamp)
    missing_data = missing_data.sort_index()

    return missing_data.isna()


def _add_overview_third_row_plot(
    fig: go.Figure,
    missing_data: pd.DataFrame,
    feature_compact_name: str,
) -> None:
    """
    Makes the plot for the third row of the overview plot and adds it to ``fig``.

    This plot is a heatmap that shows the missing values.

    Args:
        fig: plotly figure object
        missing_data: dataframe with missing data information
        feature_compact_name: name of the feature to show in text

    """
    missing_data = missing_data.copy()
    if len(missing_data.columns) == 1:
        missing_data.columns = [feature_compact_name]
    # Each dataframe in data will be plotted with a different color. To do so, the
    # dataframe to be plotted has a value of 0 if missing data is present, and the
    # position of the dataset in the datasets in data that have nan values if not.
    data_scales = missing_data.copy().max().cumsum()
    data_plot = missing_data * data_scales

    if data_scales.max() == 0:
        # Heatmaps cannot have only one color as colorscale, so we add dummy colors that
        # will not be used
        color_scale = ["red", _DEFAULT_NAN_VALUES_BACKGROUND_COLOR, "green"]
    else:
        # We only want to include as colors the ones attached to the datasets that
        # do have missing values
        data_scales = pd.concat([data_scales.iloc[:1], data_scales.diff().iloc[1:]])
        color_scale = [
            COLORS[pos] for pos, scale in enumerate(data_scales) if scale > 0
        ]
        color_scale = [_DEFAULT_NAN_VALUES_BACKGROUND_COLOR] + color_scale

    fig.add_trace(
        go.Heatmap(
            z=data_plot.T,
            x=data_plot.index,
            y=data_plot.columns,
            customdata=np.where(data_plot.T > 0, "Missing", "Not missing"),
            colorscale=color_scale,
            showscale=False,
            hovertemplate="%{x}<br>%{y}<br>%{customdata}<extra></extra>",
        ),
        row=3,
        col=1,
    )


def _update_overview_layout(
    fig: go.Figure,
    title: str,
    feature: str,
    feature_compact_name: str,
    target_compact_name: str,
    num_lines_features: int,
    num_lines_target: int,
) -> None:
    """
    Updates the layout of the feature overview figure (``fig``).

    Args:
        fig: plotly figure object
        title: title of the figure
        feature: name of the feature
        feature_compact_name: name of the feature to show in text
        target_compact_name: name of the target to show in text
        num_lines_features: number of lines the feature name has
        num_lines_target: number of lines the target name has

    """
    height = np.round(
        _DEFAULT_FIGURE_HEIGHT
        * (1 + num_lines_features * _DEFAULT_EXTRA_PERCENTAGE_HEIGHT_PER_LINE),
    )
    width = np.round(
        _DEFAULT_FIGURE_WIDTH
        * (1 + num_lines_target * _DEFAULT_EXTRA_PERCENTAGE_WIDTH_PER_LINE),
    )
    fig.update_layout(
        title=title if title else f"{feature} - Overview",
        title_x=_DEFAULT_X_TITLE_NORMALIZED_POSITION,
        template="plotly",
        showlegend=True,
        height=height,
        width=width,
        xaxis=dict(title=None, showticklabels=False),
        yaxis=dict(title=feature_compact_name),
        yaxis2=dict(title="Frequency"),
        xaxis2=dict(
            title=None,
            matches="y1",
        ),
        yaxis3=dict(title=target_compact_name),
        xaxis3=dict(
            title=None,
            matches="y1",
        ),
        yaxis4=dict(
            title=feature_compact_name,
            matches="y1",
        ),
        xaxis4=dict(matches="x5"),
        font=dict(size=_DEFAULT_GRAPH_FONT_SIZE),
    )
    num_lines = len(re.findall(pattern="<br>", string=feature_compact_name))
    for subplot in (2, 3):
        fig.add_annotation(
            x=0.5,
            y=0,
            xref=f'x{subplot} domain',
            yref=f'y{subplot} domain',
            text=feature_compact_name,
            font_size=_DEFAULT_GRAPH_FONT_SIZE + 2,
            showarrow=False,
            yshift=(
                _DEFAULT_SUBPLOT_TITLE_Y_SHIFT
                + _DEFAULT_EXTRA_TITLE_SHIFT_PER_LINE * num_lines
            ),
        )
    fig.update_xaxes(showgrid=True)  # triggers recursive update
    fig.update_yaxes(showgrid=True)  # triggers recursive update


def _add_ranges(fig: go.Figure, tag_range: TRange, target: tp.Optional[str]) -> None:
    """
    Adds ranges to the feature overview plots (modifies ``fig``)

    Args:
        fig: plotly figure object
        tag_range: range of the feature
        target: name of the target variable
    """
    _add_range(fig, row=1, col=1, tag_range=tag_range, direction="h")
    _add_range(fig, row=1, col=2, tag_range=tag_range, direction="v")
    _add_range(fig, row=2, col=1, tag_range=tag_range, direction="h")
    if target:
        _add_range(fig, row=1, col=3, tag_range=tag_range, direction="v")
