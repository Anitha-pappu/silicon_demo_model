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
Data split comparison functions.
"""

import logging
import typing as tp

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go

from modeling.types import Vector
from reporting.charts.primitives import plot_histogram
from reporting.charts.utils import (
    apply_chart_style,
    calculate_optimal_bin_width,
    check_data,
)

logger = logging.getLogger(__name__)


TITLE = "Train/Test Distributions Comparison"

SPLIT_LABEL_TEST = "Test"
SPLIT_LABEL_TRAIN = "Train"

COLOR_TRAIN = "rgb(31, 119, 180)"
COLOR_TEST = "rgb(255, 127, 14)"

DEFAULT_HISTNORM = "probability density"

FEATURE_COMPARISON_LEGEND_OFFSET_X = 1.0
FEATURE_COMPARISON_LEGEND_OFFSET_Y = 1.15

NUMERIC_FEATURES_UNIQUE_VALUE_THRESHOLD = 5


def plot_feature_comparison_for_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tp.Optional[tp.List[str]] = None,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
    fig_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
    layout_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> tp.Dict[str, go.Figure]:
    """
    Creates plotly histograms comparing train and test
    distributions of the provided figures.

    Args:
        train: training dataset
        test: testing dataset
        features: columns to plot in comparison
        height: plot's height
        width: plot's width
        fig_params: params passed to `reporting.plot_histogram`
        layout_params: params passed to `reporting.plot_histogram`

    Returns:
        Dictionary of plotly figures.
    """
    features_: tp.Iterable[str] = (
        set(train.columns).union(test.columns)
        if features is None
        else features
    )

    check_data(train, *features_)
    check_data(test, *features_)

    numeric_features, categorical_features = (
        _split_features_into_numeric_and_categorical(train, test, features_)
    )
    figs = dict(
        **_get_distplot_for_numeric_features(
            train, test, numeric_features, height, width, fig_params, layout_params,
        ),
        **_get_histogram_for_categorical_features(
            train, test, categorical_features, height, width, fig_params, layout_params,
        ),
    )
    for fig in figs.values():
        fig.update_layout(
            legend=dict(
                title_text=None,
                orientation="h",
                x=FEATURE_COMPARISON_LEGEND_OFFSET_X,
                y=FEATURE_COMPARISON_LEGEND_OFFSET_Y,
                xanchor="right",
                yanchor="top",
            ),
        )
    return figs


def _get_histogram_for_categorical_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_features: tp.List[str],
    height: tp.Optional[int],
    width: tp.Optional[int],
    fig_params: tp.Optional[tp.Dict[str, tp.Any]],
    layout_params: tp.Optional[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, go.Figure]:
    data = pd.concat([train, test], ignore_index=True)
    split_column = "split"
    data[split_column] = (
        ([SPLIT_LABEL_TRAIN for _ in range(len(train))])
        + ([SPLIT_LABEL_TEST for _ in range(len(test))])
    )
    figs = {
        feature: plot_histogram(
            data=data,
            x=feature,
            title=TITLE,
            color=split_column,
            color_discrete_map=[COLOR_TRAIN, COLOR_TEST],
            histnorm=DEFAULT_HISTNORM,
            xaxis_title=feature,
            height=height,
            width=width,
            fig_params=fig_params,
            layout_params=layout_params,
        )
        for feature in categorical_features
    }
    for fig in figs.values():
        fig.update_traces(hovertemplate="%{x}")
    return figs


def _get_distplot_for_numeric_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: tp.List[str],
    height: tp.Optional[int],
    width: tp.Optional[int],
    fig_params: tp.Optional[tp.Dict[str, tp.Any]],
    layout_params: tp.Optional[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, go.Figure]:
    return {
        feature: plot_distplot_comparison_for_numeric_features(
            train[feature],
            test[feature],
            feature,
            height=height,
            width=width,
            fig_params=fig_params,
            layout_params=layout_params,
        )
        for feature in numeric_features
    }


def plot_distplot_comparison_for_numeric_features(
    first_feature: tp.Iterable[float],
    second_feature: tp.Iterable[float],
    x_axis_title: tp.Optional[str] = None,
    show_rug: bool = True,
    height: tp.Optional[int] = None,
    width: tp.Optional[int] = None,
    fig_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
    layout_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> go.Figure:
    first_feature_ = pd.Series(first_feature).dropna()
    second_feature_ = pd.Series(second_feature).dropna()
    # this plotly interface doesn't evaluate optimal bin size automatically
    optimal_bin_size = min(
        calculate_optimal_bin_width(first_feature_),
        calculate_optimal_bin_width(second_feature_),
    )
    fig = ff.create_distplot(
        hist_data=[first_feature_, second_feature_],
        group_labels=[SPLIT_LABEL_TRAIN, SPLIT_LABEL_TEST],
        bin_size=optimal_bin_size,
        colors=[COLOR_TRAIN, COLOR_TEST],
        histnorm=DEFAULT_HISTNORM,
        show_rug=show_rug,
    )
    if x_axis_title is None:
        x_axis_title = first_feature_.name or second_feature_.name

    apply_chart_style(
        fig=fig,
        height=height,
        width=width,
        title=TITLE,
        xaxis_title=x_axis_title,
        fig_params=fig_params,
        layout_params=layout_params,
    )
    return fig


def is_data_effectively_numeric(
    data: Vector,
    unique_values_threshold: int = NUMERIC_FEATURES_UNIQUE_VALUE_THRESHOLD,
) -> bool:
    """ Determine if (1-dimensional) data are effectively numeric.

    Notes:
        This function uses a robust and conservative method to identify "effectively
         numeric" features. A feature is considered effectively numeric if and only if
         all the following criteria are met:
        - data can be converted to float
        - data have more than ``unique_values_threshold`` distinct values
    """
    try:
        data.astype(float)
    except ValueError:
        return False
    data_as_series = pd.Series(data)  # To ensure the `.nuniqe` in the next line works
    return tp.cast(bool, data_as_series.nunique() > unique_values_threshold)


# TODO: Update the logic so that random split does not affect whether a feature is
#  considered numeric or categorical.
#  Likely best to just test train and test data together, since they are both provided
#  to the function anyway.
def _split_features_into_numeric_and_categorical(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    features: tp.Iterable[str],
    unique_values_threshold: int = NUMERIC_FEATURES_UNIQUE_VALUE_THRESHOLD,
) -> tp.Tuple[tp.List[str], tp.List[str]]:
    """ Return two lists, one with the numeric features, and one with the categorical
    ones

    Notes:
        This function uses a robust and conservative method to identify "truly numeric"
         features.
        A feature is considered "truly numeric" if and only if both ``data_train`` and
        ``data_test`` are "truly numeric". Determining wheter each of these is "truly
         numeric" is done by using the function ``_are_data_truly_numeric``, passing the
         parameter ``unique_values_threshold`` as parameter to to that function.
        All other features are considered categoric.

        ``unique_values_threshold`` defaults to
         ``NUMERIC_FEATURES_UNIQUE_VALUE_THRESHOLD``
    """
    numeric_features = []
    non_numeric_features = []
    for feature in features:
        train_data_are_truly_numeric = is_data_effectively_numeric(
            data_train[feature], unique_values_threshold=unique_values_threshold,
        )
        test_data_are_truly_numeric = is_data_effectively_numeric(
            data_test[feature], unique_values_threshold=unique_values_threshold,
        )
        if train_data_are_truly_numeric and test_data_are_truly_numeric:
            numeric_features.append(feature)
        else:
            non_numeric_features.append(feature)

    return numeric_features, non_numeric_features
