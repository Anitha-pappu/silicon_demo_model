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
import seaborn as sns

from reporting.charts.feature_overview import (
    TData,
    TDict,
    TRange,
    plot_feature_overview,
    transform_data_to_dict,
)
from reporting.charts.primitives import plot_correlation, plot_focused_pairplot
from reporting.charts.utils import check_data, wrap_string
from reporting.rendering.types import TRenderingAgnosticDict

TRangeOrDictWithRange = tp.Union[TRange, tp.Dict[str, TRange]]

def plot_pair_plot(data, features, target, title='Pair Plot'):
    # Create the pair plot using Seaborn
    pair_plot = sns.pairplot(data, hue=target)
    # pair_plot.fig.suptitle(title, y=1.02)  # Adjust the title position
    return pair_plot


def plot_feature_overviews(
    data: TData,
    features: tp.Union[str, tp.Iterable[str]],
    timestamp: str,
    tag_ranges: tp.Optional[TRangeOrDictWithRange] = None,
    target: tp.Optional[str] = None,
    title: tp.Optional[str] = None,
    add_common_plots: bool = False,
    labels_length_limit: int = 20,
    fig_params: tp.Optional[TDict] = None,
    layout_params: tp.Optional[TDict] = None,
) -> tp.Union[go.Figure, TRenderingAgnosticDict]:
    """
    Create plots a collection of plots representing chosen feature "overview".
    This includes a boxplot and histogram to understand the distribution of values,
    a scatter-plot vs the `target` variable, and a time-series plot of the `feature` and
    `target`.

    Args:
        data: data to plot. It can be either a dataframe or a list or dictionary of them
        features: feature or features to show
        tag_ranges: tag's or tags' ranges (min, max) to show
        timestamp: column name of the timestamp associated with the feature
        target: column name of the target variable
        title: title of the chart
        add_common_plots: if True, adds correlation matrix and scatter plots with target
        labels_length_limit: limits feature name to `name[:feature_name_limit]...`
            in case it's too long
        fig_params: kwargs for plotly chart function
        layout_params: dictionary containing keys and values for updating
            the plotly fig layout

    Returns:
        dictionary containing plotly feature overview chart
    """

    features = _transform_features(features)
    data_df = data
    data = _validate_transform_data(
        data, timestamp, features, target, labels_length_limit,
    )
    tag_ranges = _validate_tag_ranges(features, tag_ranges)

    figs: TRenderingAgnosticDict = {
        feature: plot_feature_overview(
            data=data,
            feature=feature,
            timestamp=timestamp,
            target=target,
            tag_range=tag_ranges.get(feature),
            title=title,
            labels_length_limit=labels_length_limit,
            fig_params=fig_params,
            layout_params=layout_params,
        )
        for feature in features
    }
    if len(features) == 1:
        return figs[features[0]]

    if add_common_plots:
        _add_common_plots(figs, data, features, target, labels_length_limit)

    
    figs['pair_plots'] = plot_pair_plot(data_df[features[:7]+[target]],features,target)

    return figs


def _transform_features(features: tp.Union[str, tp.Iterable[str]]) -> tp.List[str]:
    """
    Transforms features to a list.

    Args:
        features: feature or features to show

    Returns:
        list of features to show
    """
    if isinstance(features, str):
        features = [features]
    return list(features)


def _validate_tag_ranges(
    features: tp.List[str],
    tag_ranges: tp.Optional[TRangeOrDictWithRange] = None,
) -> tp.Dict[str, TRange]:
    """
    Ensures that tag ranges (if provided) are a tuple only when exactly one feature is
    plotted.

    Args:
        features: list of features to show
        tag_ranges: tag's or tags' ranges (min, max) to show

    Returns:
        dictionary containing tag ranges for each feature

    Raises:
        ValueError: if `tag_ranges` is a tuple and more than one feature is plotted
    """
    if tag_ranges is None:
        tag_ranges = {}
    elif isinstance(tag_ranges, tuple) and len(features) == 1:
        tag_ranges = {features[0]: tag_ranges}
    elif isinstance(tag_ranges, tuple):
        raise ValueError("`tag_ranges` can be `tuple` only when one feature is passed")
    return tag_ranges


def _validate_transform_data(
    data: TData,
    timestamp: str,
    features: list[str],
    target: str,
    labels_length_limit: int,
) -> dict[str, pd.DataFrame]:
    """
    Transforms data to dictionary and validates it.

    Args:
        data: dataframe holding data to plot
        timestamp: column name of the timestamp associated with the feature
        features: list of features to show
        target: column name of the target variable
        labels_length_limit: limits feature name in case it's too long

    Returns:
        dictionary containing data to plot

    """
    data = transform_data_to_dict(data, labels_length_limit)
    for data_name, data_values in data.items():
        timestamp_in_index = (
            timestamp not in data_values.columns
            and timestamp in data_values.index.names
        )
        if timestamp_in_index:
            data[data_name] = data_values.reset_index()
        check_data(data_values, timestamp, *features, target)
    return data


def _add_common_plots(
    figs: TRenderingAgnosticDict,
    data: TData,
    features: list[str],
    target: str,
    labels_length_limit: int,
) -> None:
    """
    Adds correlation matrix and scatter plots with target to the dictionary of figures.

    Args:
        figs: dictionary containing plotly feature overview chart
        data: dictionary containing data to plot
        features: list of features to show
        target: column name of the target variable
        labels_length_limit: limits feature name in case it's too long
    """
    short_data = {
        data_name: data_df.rename(
            {col: wrap_string(col, labels_length_limit) for col in data_df.columns},
            axis=1,
        ) for data_name, data_df in data.items()
    }
    short_features = [wrap_string(feature, labels_length_limit) for feature in features]
    short_target = (
        wrap_string(target, labels_length_limit) if target is not None else None
    )
    features_plus_target = (
        short_features + [short_target] if target is not None else short_features
    )
    figs["Correlation matrix"] = {
        data_name: plot_correlation(
            data=data_df,
            rows=features_plus_target,
            columns=features_plus_target,
            mask="lower",
        )
        for data_name, data_df in short_data.items()
    }
    if target is not None:
        figs["Scatter plots with target"] = {
            data_name: plot_focused_pairplot(
                data=data_df,
                target_column=short_target,
                feature_columns=short_features,
            )
            for data_name, data_df in short_data.items()
        }
