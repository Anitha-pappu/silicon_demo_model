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
from colorsys import hls_to_rgb, rgb_to_hls

import matplotlib.colors as mc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from reporting.charts.utils import apply_chart_style
from reporting.config import COLORS

_BASELINE_COLOR = "#FFC107"
_ACTUAL_COLOR = "#D81B60"
_TEXT_TITLE_SIZE = 16
_TEXT_LEFT_MARGIN = 50
_MAX_CHARACTERS_PER_TEXT_LINE = 150
_P_VALUE_THRESHOLD = 0.05
_PLOT_HEIGHT = 400
_PLOT_WIDTH = 1000
_TITLE_SIZE = 20
_TEXT_SIZE = 12
_LINE_WIDTH = 3
_OPACITY = 0.8
_ANNOTATION_XSHIFT = 5
_OFFSET_LINE_TEXT = 0.02
_SCALE_DARK = 0.3
_SCALE_LIGHT = 1.3
_HISTNORM: tp.Literal[
    "", "percent", "probability", "density", "probability density",
] = "probability density"
_GROUP_ALL_DATA = "all_data"
_BASELINE_HISTOGRAM_NAME = "baseline error"
_BASELINE_LINE_NAME = "expected baseline error mean"
_UPLIFTS_HISTOGRAM_NAME = "uplifts"
_UPLIFTS_LINE_NAME = "uplifts mean"


def plot_uplifts_and_baseline_error_histogram(
    data: pd.DataFrame,
    baseline_historical_data: pd.DataFrame,
    bin_width: float,
) -> go.Figure:
    """
    Generates histogram of uplifts and baseline error.

    Args:
        data: Dataframe with baseline, optimal and actual values
        baseline_historical_data: Dataframe with baseline predictions and actual values
            over a historical dataset to assess its bias
        bin_width: Width of the bins for the histogram

    Returns:
        Histogram of uplifts and baseline error

    """
    data_plot_uplift = data[data["value_type"].isin(["actual_uplift"])].copy()
    histogram = go.Figure()
    histogram = _add_baseline_histogram_traces(
        histogram,
        baseline_historical_data,
        bin_width,
    )
    histogram = _add_uplift_histogram_traces(
        histogram,
        data_plot_uplift,
        bin_width,
    )
    max_height = _get_max_bin_height(histogram, bin_width, _HISTNORM)
    histogram = _add_baseline_line_traces(
        histogram,
        baseline_historical_data,
        max_height,
    )
    histogram = _add_uplift_line_traces(
        histogram,
        data_plot_uplift,
        max_height,
    )
    apply_chart_style(
        fig=histogram,
        height=_PLOT_HEIGHT,
        width=_PLOT_WIDTH,
        title="Uplift and baseline error distribution",
        xaxis_title="Value",
        yaxis_title="Distribution",
        layout_params={
            "title_font_size": _TITLE_SIZE,
            "font_size": _TEXT_SIZE,
            "barmode": "overlay",
        },
    )

    return histogram


def plot_uplift_histogram(
    data: pd.DataFrame,
    bin_width: float,
) -> go.Figure:
    """
    Generates histogram of uplifts.

    Args:
        data: Dataframe with baseline, optimal and actual values
        bin_width: Width of the bins for the histogram

    Returns:
        Histogram of uplifts

    """

    histogram = go.Figure()
    histogram = _add_uplift_histogram_traces(
        histogram,
        data[data["value_type"] == "actual_uplift"],
        bin_width,
    )
    max_height = _get_max_bin_height(histogram, bin_width, _HISTNORM)
    histogram = _add_uplift_line_traces(
        histogram,
        data[data["value_type"] == "actual_uplift"],
        max_height,
    )
    apply_chart_style(
        fig=histogram,
        height=_PLOT_HEIGHT,
        width=_PLOT_WIDTH,
        title="Uplift distribution",
        xaxis_title="Uplifts value",
        yaxis_title="Uplifts distribution",
        layout_params={
            "title_font_size": _TITLE_SIZE,
            "font_size": _TEXT_SIZE,
            "barmode": "overlay",
        },
    )

    return histogram


def plot_baseline_bias_histogram(
    baseline_historical_data: pd.DataFrame,
    bin_width: float,
) -> go.Figure:
    """
    Generates histogram of baseline bias.

    Args:
        baseline_historical_data: Dataframe with baseline predictions and actual values
            over a historical dataset to assess its bias
        bin_width: Width of the bins for the histogram

    Returns:
        Histogram of baseline bias

    """
    histogram = go.Figure()
    histogram = _add_baseline_histogram_traces(
        histogram,
        baseline_historical_data,
        bin_width,
    )
    max_height = _get_max_bin_height(histogram, bin_width, _HISTNORM)
    histogram = _add_baseline_line_traces(
        histogram,
        baseline_historical_data,
        max_height,
    )

    apply_chart_style(
        fig=histogram,
        height=_PLOT_HEIGHT,
        width=_PLOT_WIDTH,
        title="Baseline errors distribution",
        xaxis_title="Error value",
        yaxis_title="Error distribution",
        layout_params={
            "title_font_size": _TITLE_SIZE,
            "font_size": _TEXT_SIZE,
            "barmode": "overlay",
        },
    )

    return histogram


def _add_baseline_histogram_traces(
    histogram: go.Figure,
    baseline_historical_data: pd.DataFrame,
    bin_width: float,
) -> go.Figure:
    """
    Adds histogram traces for baseline error.

    Args:
        histogram: Histogram figure
        baseline_historical_data: Dataframe with baseline predictions and actual values
            over a historical dataset to assess its bias
        bin_width: Width of the bins for the histogram

    Returns:
        Histogram figure with baseline error traces

    """
    histogram.add_trace(go.Histogram(
        x=baseline_historical_data["error"],
        legendgroup=_GROUP_ALL_DATA,
        histnorm=_HISTNORM,
        legendgrouptitle={"text": _GROUP_ALL_DATA},
        name=_BASELINE_HISTOGRAM_NAME,
        xbins={"size": bin_width},
        marker={"color": _BASELINE_COLOR},
        opacity=_OPACITY,
    ))
    if "group" in baseline_historical_data.columns:
        colors_group = [_scale_color(color, _SCALE_LIGHT) for color in COLORS]
        for group in baseline_historical_data["group"].drop_duplicates():
            histogram.add_trace(go.Histogram(
                x=baseline_historical_data[
                    baseline_historical_data["group"] == group
                ]["error"],
                legendgroup=group,
                histnorm=_HISTNORM,
                legendgrouptitle={"text": group},
                name=_BASELINE_HISTOGRAM_NAME,
                xbins={"size": bin_width},
                marker={"color": colors_group.pop(0)},
                opacity=_OPACITY,
                visible="legendonly",
            ))

    return histogram


def _add_baseline_line_traces(
    histogram: go.Figure,
    baseline_historical_data: pd.DataFrame,
    max_height: float,
) -> go.Figure:
    """
    Adds line traces for baseline error.

    Args:
        histogram: Histogram figure
        baseline_historical_data: Dataframe with baseline predictions and actual values
            over a historical dataset to assess its bias
        max_height: Max height of the histogram bins

    Returns:
        Histogram figure with baseline error line traces

    """
    histogram.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, max_height + _OFFSET_LINE_TEXT],
        text=["", "<b>0</b>"],
        textfont={
            "color": _BASELINE_COLOR,
            "size": _TEXT_SIZE,
        },
        mode="lines+text",
        legendgroup=_GROUP_ALL_DATA,
        line=dict(
            color=_BASELINE_COLOR,
            width=_LINE_WIDTH,
            dash="dash",
        ),
        name=_BASELINE_LINE_NAME,
    ))
    if "group" in baseline_historical_data.columns:
        colors_group = [_scale_color(color, _SCALE_LIGHT) for color in COLORS]
        for group in baseline_historical_data["group"].drop_duplicates():
            color = colors_group.pop(0)
            histogram.add_trace(go.Scatter(
                x=[0, 0],
                y=[0, max_height + _OFFSET_LINE_TEXT],
                text=["", "<b>0</b>"],
                textfont={
                    "color": color,
                    "size": _TEXT_SIZE,
                },
                mode="lines+text",
                legendgroup=group,
                line={
                    "color": color,
                    "width": _LINE_WIDTH,
                    "dash": "dash",
                },
                name=_BASELINE_LINE_NAME,
                visible="legendonly",
            ))

    return histogram


def _add_uplift_histogram_traces(
    histogram: go.Figure,
    data_plot_uplift: pd.DataFrame,
    bin_width: float,
) -> go.Figure:
    """
    Adds histogram traces for uplifts.

    Args:
        histogram: Histogram figure
        data_plot_uplift: Dataframe with uplifts
        bin_width: Width of the bins for the histogram

    Returns:
        Histogram figure with uplifts traces

    """
    histogram.add_trace(go.Histogram(
        x=data_plot_uplift["value"],
        legendgroup=_GROUP_ALL_DATA,
        histnorm=_HISTNORM,
        legendgrouptitle={"text": _GROUP_ALL_DATA},
        name=_UPLIFTS_HISTOGRAM_NAME,
        xbins={"size": bin_width},
        marker={"color": _ACTUAL_COLOR},
        opacity=_OPACITY,
    ))
    if "group" in data_plot_uplift.columns:
        colors_group = [_scale_color(color, _SCALE_DARK) for color in COLORS]
        for group in data_plot_uplift["group"].drop_duplicates():
            histogram.add_trace(go.Histogram(
                x=data_plot_uplift[
                    data_plot_uplift["group"] == group
                ]["value"],
                legendgroup=group,
                histnorm=_HISTNORM,
                legendgrouptitle={"text": group},
                name=_UPLIFTS_HISTOGRAM_NAME,
                xbins={"size": bin_width},
                marker={"color": colors_group.pop(0)},
                opacity=_OPACITY,
                visible="legendonly",
            ))

    return histogram


def _add_uplift_line_traces(
    histogram: go.Figure,
    data_plot_uplift: pd.DataFrame,
    max_height: float,
) -> go.Figure:
    """
    Adds line traces for uplifts.

    Args:
        histogram: Histogram figure
        data_plot_uplift: Dataframe with uplifts
        max_height: Max height of the histogram bins

    Returns:
        Histogram figure with uplifts line traces

    """
    uplift_mean = np.round(data_plot_uplift["value"].mean(), 2)
    histogram.add_trace(go.Scatter(
        x=[uplift_mean, uplift_mean],
        y=[0, max_height + _OFFSET_LINE_TEXT],
        text=["", f"<b>{uplift_mean}</b>"],
        textfont={
            "color": _ACTUAL_COLOR,
            "size": _TEXT_SIZE,
        },
        mode="lines+text",
        legendgroup=_GROUP_ALL_DATA,
        line=dict(
            color=_ACTUAL_COLOR,
            width=_LINE_WIDTH,
            dash="dash",
        ),
        name=_UPLIFTS_LINE_NAME,
    ))
    if "group" in data_plot_uplift.columns:
        colors_group = [_scale_color(color, _SCALE_DARK) for color in COLORS]
        for group in data_plot_uplift["group"].drop_duplicates():
            color = colors_group.pop(0)
            uplift_mean = np.round(
                data_plot_uplift[data_plot_uplift["group"] == group]["value"].mean(),
                2,
            )
            histogram.add_trace(go.Scatter(
                x=[uplift_mean, uplift_mean],
                y=[0, max_height + _OFFSET_LINE_TEXT],
                text=["", f"<b>{uplift_mean}</b>"],
                textfont={
                    "color": color,
                    "size": _TEXT_SIZE,
                },
                mode="lines+text",
                legendgroup=group,
                line={
                    "color": color,
                    "width": _LINE_WIDTH,
                    "dash": "dash",
                },
                name=_UPLIFTS_LINE_NAME,
                visible="legendonly",
            ))

    return histogram


def _get_max_bin_height(fig: go.Figure, bin_width: float, histnorm: str) -> float:
    """
    Calculates the maximum height of the bins in the histogram.

    Args:
        fig: Figure with only histogram traces
        bin_width: Width of the bins for the histogram
        histnorm: Normalization mode for the histogram

    Returns:
        Maximum height of the bins in the histogram

    """
    max_vals: list[float] = []
    for data in fig.data:
        x_data = data.x
        plot_bins = np.concatenate((
            np.arange(start=0, stop=max(x_data) + bin_width, step=bin_width),
            np.arange(start=0, stop=min(x_data) - bin_width, step=-bin_width),
        ))
        plot_bins = np.sort(np.unique(plot_bins))
        counts, bins = np.histogram(list(x_data), bins=plot_bins)
        if histnorm in {"percent", "probability", "probability density"}:
            counts = counts / sum(counts)
        max_vals += [max(counts)]
    if histnorm in {"", "probability"}:
        return max(max_vals)
    elif histnorm == "percent":
        return max(max_vals) * 100
    elif histnorm in {"density", "probability density"}:
        return max(max_vals) / bin_width
    raise ValueError(
        "histnorm should be one of the following: "
        "'', 'percent', 'probability', 'density', 'probability density'.",
    )


def _scale_color(hex_color: str, scale: float) -> str:
    """
    Scales a hex color by the given scale.

    Args:
        hex_color: Hex color to scale
        scale: Scale to use

    Returns:
        Scaled hex color

    """
    rgb_color = mc.to_rgb(hex_color)
    hls_h, hls_l, hls_s = rgb_to_hls(*rgb_color)
    manipulated_rgb_color = hls_to_rgb(hls_h, min(1.0, hls_l * scale), s=hls_s)
    return mc.to_hex(manipulated_rgb_color)


def get_text_for_test_results(
    data_test_results: pd.DataFrame,
    null_hypothesis: str,
    alternative_hypothesis: str,
    rejection_comment: str,
    no_rejection_comment: str,
    result_summary_rejection: str,
    result_summary_no_rejection: str,
) -> dict[str, str]:
    """
    Generates explanations for the test results.

    Args:
        data_test_results: Dataframe with the results of the test
        null_hypothesis: Null hypothesis
        alternative_hypothesis: Alternative hypothesis
        rejection_comment: Comment to add in case of rejection
        no_rejection_comment: Comment to add in case of no rejection
        result_summary_rejection: Summary of the results in case of rejection
        result_summary_no_rejection: Summary of the results in case of no rejection

    Returns:
        Text with the explanation of the test results

    """
    base_text = (
        "Hypothesis testing has been done with the following hypothesis:\n"
        f"    - H0: {null_hypothesis}.\n"
        f"    - H1: {alternative_hypothesis}.\n\n"
    )
    p_value_all_data = data_test_results[
        data_test_results["group"] == _GROUP_ALL_DATA
    ]["p_value"].values[0]
    if p_value_all_data < _P_VALUE_THRESHOLD:
        result_summary = result_summary_rejection
        text_test_results = (
            f"Using a significance of {(1 - _P_VALUE_THRESHOLD)} (p-value threshold of "
            f"{_P_VALUE_THRESHOLD}), the null hypothesis is rejected in favor of the "
            f"alternative. The hypothesis of the "
            f"{null_hypothesis.replace(' is ', ' being ').lower()} "
            f"is rejected in favor of the hypothesis of the "
            f"{alternative_hypothesis.replace(' is ', ' being ').lower()}.\n\n"
            f"{rejection_comment}."
        )
    else:
        result_summary = result_summary_no_rejection
        text_test_results = (
            f"Using a significance of {(1 - _P_VALUE_THRESHOLD)} (p-value threshold of "
            f"{_P_VALUE_THRESHOLD}), the null hypothesis is not rejected in favor of "
            f"the alternative. The hypothesis of the "
            f"{null_hypothesis.replace(' is ', ' being ').lower()} is not "
            f"rejected in favor of the hypothesis of the "
            f"{alternative_hypothesis.replace(' is ', ' being ').lower()}.\n\n"
            f"{no_rejection_comment}."
        )

    if len(data_test_results) > 1:
        text_groups = (
            "\n\nThe test was also performed for each group. The results are shown "
            "below. The overall results are the ones that should be used to prove "
            "impact. However, a view per groups can help to understand differences in "
            "behavior."
        )
    else:
        text_groups = ""

    return {
        "summary": result_summary,
        "deep_dive": f"{base_text}{text_test_results}{text_groups}",
    }
