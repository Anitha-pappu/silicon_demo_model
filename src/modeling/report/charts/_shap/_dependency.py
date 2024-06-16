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

import logging
import math
import typing as tp

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modeling.models.model_base import ShapExplanation

from .._histogram_utils import add_histogram_trace  # noqa: WPS436
from ._shap_lib_utils import potential_interactions  # noqa: WPS436
from ._utils import (  # noqa: WPS436,WPS450
    FeatureShapExplanation,
    _crop_feature_values,
    extract_explanations_for_given_features,
    sort_features,
)

_P = tp.TypeVar("_P", bound=npt.NBitBase)  # noqa: WPS111
_TNumericNDArray = npt.NDArray[np.number[_P]]

_TRange = tp.Tuple[float, float]
_TITLE_PREFIX = "%TITLE%"
_AUTO_COLOR_BY_ARG = "_auto"
_LINE_COLOR_NON_COLORED = "rgb(43, 75, 110)"
_COLOR_SCALE = ("rgb(56,138,243)", "rgb(234,51,86)")

LAYOUT_MARGIN_L = 130
LAYOUT_MARGIN_R = 130
LAYOUT_MARGIN_B = 80
LAYOUT_LEGEND_OFFSET_X = 1
LAYOUT_LEGEND_OFFSET_Y = 1.02

logger = logging.getLogger(__name__)

_RELATIVE_OFFSET_RANGE_MULTIPLIER = 0.05


def _get_y_axis_range(
    shap_values: _TNumericNDArray[_P],
    relative_offset_range_multiplier: float = _RELATIVE_OFFSET_RANGE_MULTIPLIER,
) -> _TRange:
    shap_values_range = shap_values.max() - shap_values.min()
    offset = relative_offset_range_multiplier * shap_values_range
    y_axis_range = (shap_values.min() - offset, shap_values.max() + offset)
    return y_axis_range  # noqa: WPS331  # Naming makes meaning clearer


def plot_shap_dependency(  # noqa: WPS210
    features: tp.List[str],
    shap_explanation: ShapExplanation,
    order_by: _TNumericNDArray[_P] | None = None,
    color_by: str | None = _AUTO_COLOR_BY_ARG,
    max_features_to_display: int | None = 20,
    n_columns: int = 2,
    subplot_height: int = 320,
    subplot_width: int = 370,
    horizontal_spacing_per_row: float = 0.8,
    vertical_spacing_per_column: float = 0.45,
) -> go.Figure:
    """
    Plots shap values of a feature against its values

    Args:
        features: feature names to draw dependency for
        shap_explanation: shap explanation
        order_by: array to DESCENDING sort features by;
            sorts by mean abs shap value by default
        n_columns: number of columns to show
        color_by: feature name of shap values to use for coloring scatter points by;
            `color_by` has to be present in `shap_explanation`;
            if `None` is provided, no coloring applied;
            if `"_auto"` is passed, we'll color each scatter by the feature that
            it has the most interaction with
            (note that we use shap lib approximation for those interactions);
        max_features_to_display: max features to show in explanation; show all if None
        subplot_height: each figure's subplot height
        subplot_width: each figure's subplot width
        horizontal_spacing_per_row: horizontal spacing between subplots
            in normalized plot coordinates
            (the value is divided by number of plots since the total plot size grows)
        vertical_spacing_per_column: vertical spacing between subplots
            in normalized plot coordinates
            (the value is divided by number of plots since the total plot size grows)
    """
    features = sort_features(features, order_by, shap_explanation, descending=True)
    if max_features_to_display is not None and len(features) > max_features_to_display:
        logger.info(
            f"Too many features provided. "
            f"Only first {max_features_to_display = } will be plot.",
        )
        features = features[:max_features_to_display]

    by_feature_explanation = extract_explanations_for_given_features(
        features, shap_explanation,
    )
    n_features = len(features)
    n_rows = math.ceil(n_features / n_columns)
    fig = make_subplots(
        cols=n_columns,
        rows=n_rows,
        figure=go.Figure(
            layout=_get_layout(n_rows * subplot_height, n_columns * subplot_width),
        ),
        horizontal_spacing=horizontal_spacing_per_row / n_columns,
        vertical_spacing=vertical_spacing_per_column / n_rows,
        subplot_titles=[f"{_TITLE_PREFIX}{feat}" for feat in features],
        specs=[
            [{"secondary_y": True} for _ in range(n_columns)]
            for _ in range(n_rows)
        ],
    )

    explanation: FeatureShapExplanation
    for plot_index, (feature, explanation) in enumerate(by_feature_explanation.items()):
        row = plot_index // n_columns + 1
        column = plot_index % n_columns + 1
        coloring = _get_coloring(
            fig,
            color_by,
            explanation,
            shap_explanation,
            row,
            column,
        )
        y_axis_range = _get_y_axis_range(explanation.values)
        _add_scatter(
            fig=fig,
            feature_values=explanation.data,
            feature_shaps=explanation.values,
            coloring=coloring,
            row=row,
            column=column,
        )
        # print('Feature',feature)
        add_histogram_trace(fig, explanation.data, row, column, visible=True)
        _update_subplot_layout(fig, feature, y_axis_range, row, column)
    return fig


def _update_subplot_layout(
    fig: go.Figure,
    feature: str,
    y_axis_range: _TRange,
    row: int,
    column: int,
    title_offset: float = 0.005,
) -> None:
    fig.update_xaxes(
        title="feature values",
        showline=True,
        linecolor="black",
        ticks="outside",
        hoverformat=".2f",
        row=row,
        col=column,
    )
    fig.update_yaxes(
        title="SHAP values",
        showline=True,
        linecolor="black",
        ticks="outside",
        range=y_axis_range,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="lightgrey",
        secondary_y=False,
        row=row,
        col=column,
    )
    subplot_title = next(
        fig.select_annotations(selector={"text": f"{_TITLE_PREFIX}{feature}"}),
    )
    subplot_text_without_prefix = subplot_title.text.lstrip(_TITLE_PREFIX)
    subplot_title.text = f"<b>{subplot_text_without_prefix}</b>"
    subplot_title.y += title_offset


def _add_scatter(
    fig: go.Figure,
    feature_values: _TNumericNDArray[_P],
    feature_shaps: _TNumericNDArray[_P],
    coloring: tp.Dict[str, tp.Any],
    row: int,
    column: int,
) -> None:
    fig.add_trace(
        # todo: there's a bug with scattergl
        #  when number of plots is ~20 scatters are shifted up, away from axes.
        #  we can mitigate this by using scatter in case of n_features > XX.
        #  this will also require removal of color bar border,
        #  because that exists by default for go.Scatter
        go.Scattergl(
            x=feature_values,
            y=feature_shaps,
            mode="markers",
            name="feature value<br>shap value",
            hovertemplate="%{x:.2f}<br>%{y:.2f}",
            marker=coloring,
            showlegend=False,
        ),
        row=row,
        col=column,
    )


def _get_layout(
    height: int, width: int,
) -> go.Layout:
    return go.Layout(
        height=height,
        width=width,
        bargap=0,
        plot_bgcolor="rgba(0,0,0,0)",
        title="SHAP Values Dependence Plot",
        legend=dict(
            orientation="h",
            x=LAYOUT_LEGEND_OFFSET_X,
            y=LAYOUT_LEGEND_OFFSET_Y,
            xanchor="left",
            yanchor="bottom",
        ),
        margin=dict(l=LAYOUT_MARGIN_L, r=LAYOUT_MARGIN_R, b=LAYOUT_MARGIN_B),
        hovermode="x",
    )


def _get_color_bar(
    fig: go.Figure, row: int, column: int, color_by: str | None,
) -> dict[str, tp.Any]:
    """Returns color bar config"""
    _, xaxis_domain_end = next(fig.select_xaxes(row=row, col=column)).domain
    yaxis_domain = next(fig.select_yaxes(row=row, col=column)).domain
    yaxis_domain_center = sum(yaxis_domain) / 2
    yaxis_domain_length = yaxis_domain[1] - yaxis_domain[0]
    x_offset = 0.01
    return dict(
        title=color_by,
        titleside="right",
        tickmode="array",
        ticks="outside",
        thickness=5,
        len=yaxis_domain_length,
        x=xaxis_domain_end + x_offset,
        y=yaxis_domain_center,
    )


def _get_coloring(
    fig: go.Figure,
    color_by: str | None,
    feature_shap_explanation: FeatureShapExplanation,
    shap_explanation: ShapExplanation,
    row: int,
    column: int,
) -> tp.Dict[str, tp.Any]:
    if color_by is None:
        return dict(color=_LINE_COLOR_NON_COLORED)
    elif color_by == _AUTO_COLOR_BY_ARG:
        most_interacting_feature_index = potential_interactions(
            feature_shap_explanation, shap_explanation,
        )[0]
        color_by = shap_explanation.feature_names[most_interacting_feature_index]
    else:
        # validate requested `color_by` is present
        if color_by not in shap_explanation.feature_names:
            raise ValueError(
                f"Couldn't find requested shap values to color_by ({color_by})",
            )
    explanation_color_by = extract_explanations_for_given_features(
        [color_by],  # type: ignore  #(incorrect mypy parsing)
        shap_explanation,
    )[color_by]  # type: ignore  #(incorrect mypy parsing)
    normalized_feature_values = _crop_feature_values(
        explanation_color_by.data,
    )
    marker = dict(
        color=normalized_feature_values,
        colorscale=_COLOR_SCALE,
        opacity=1,
        colorbar=_get_color_bar(fig, row, column, color_by),
    )
    return marker  # noqa: WPS331  # Naming makes meaning clearer
