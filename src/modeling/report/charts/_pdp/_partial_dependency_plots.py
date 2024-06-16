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
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modeling.api import SupportsModel

from .._histogram_utils import add_histogram_trace  # noqa: WPS436
from ._grid import create_pdp_grid_from_data  # noqa: WPS436
from ._model_predictions import (  # noqa: WPS436,WPS450
    _create_model_predictions_for_grid,
)

_TRange = tp.Tuple[float, float]
_P = tp.TypeVar("_P", bound=npt.NBitBase)  # noqa: WPS111
_TNumericNDArray = npt.NDArray[np.number[_P]]
_TFeatureImportanceDict = tp.Dict[str, float]

TGridOptions = tp.Literal["quantiles", "uniform", "quantiles+uniform"]
TAxisRangeOptions = tp.Literal["average", "all"]

LAYOUT_MARGIN_L = 130
LAYOUT_MARGIN_R = 10
LAYOUT_MARGIN_B = 80
LAYOUT_LEGEND_OFFSET_X = 0.85
LAYOUT_LEGEND_OFFSET_Y = 1.02

_RELATIVE_OFFSET_RANGE_MULTIPLIER = 0.05

_RGBA_COLOR_GREY = "rgb(189,189,189)"
_RGBA_COLOR_BLUE = "rgb(2,0,121)"
_RGBA_COLOR_TRANSPARENT = "rgba(0,0,0,0)"

logger = logging.getLogger(__name__)


def plot_partial_dependency_plots(  # noqa: WPS210,WPS211
    model: SupportsModel,
    data_for_pdp_grid_calculations: pd.DataFrame,
    ordered_features: tp.Iterable[str] | None = None,
    feature_importance: _TFeatureImportanceDict | None = None,
    n_point_in_grid: int = 20,
    grid_calculation_strategy: TGridOptions = "quantiles+uniform",
    n_samples_to_calculate_predictions: int = 100,
    max_features_to_display: int | None = 20,
    y_axis_range_mode: TAxisRangeOptions = "average",
    y_axis_tick_values_precision: str = ".2f",
    n_columns: int = 2,
    subplot_height: int = 400,
    subplot_width: int = 530,
    horizontal_spacing_per_row: float = 0.2,
    vertical_spacing_per_column: float = 0.4,
    random_state: int | None = 42,
) -> go.Figure:
    """
    Plot interactive partial dependency plots with
    histograms for the provided model, data, and features.
    Works with any kind of model that implements the ``predict()`` method.

    Args:
        model: model to be used for predictions
        data_for_pdp_grid_calculations: data to calculate
         grid for partial dependency plots
        ordered_features: order of features to be displayed in the plots
        feature_importance: feature importance to be used for
         ordering features if order is not provided
        n_point_in_grid: number of points in the grid
        grid_calculation_strategy: calculation strategy for the grid.
         Supported strategies: ``'quantiles'``, ``'uniform'``, ``'quantiles+uniform'``
        n_samples_to_calculate_predictions: Sample count from the data
         to calculate and plot predictions
        max_features_to_display: maximum number of features to be displayed
        y_axis_range_mode: mode to calculate y-axis range.
         Supported values: ``'average'``, ``'all'``.
        y_axis_tick_values_precision: number of digits after
         decimal point for y-axis tick values
        n_columns: number of columns of the subplot in the figure
        subplot_height: height of the subplot in the figure
        subplot_width: width of the subplot in the figure
        horizontal_spacing_per_row: space between subplots in the same row
        vertical_spacing_per_column: space between subplots in the same column
        random_state: random state for sampling from the dataset

    Returns:
        Figure with subplots representing partial dependency plots
    """
    features = _get_ordered_features(
        model,
        ordered_features,
        feature_importance,
        descending=True,
    )
    if max_features_to_display is not None and len(features) > max_features_to_display:
        logger.info(
            f"Too many features provided. "
            f"Only first {max_features_to_display = } will be plot.",
        )
        features = features[:max_features_to_display]
    n_features = len(features)
    n_rows = math.ceil(n_features / n_columns)
    fig = make_subplots(
        cols=n_columns,
        rows=n_rows,
        figure=go.Figure(
            layout=_get_pdp_layout(n_rows * subplot_height, n_columns * subplot_width),
        ),
        horizontal_spacing=horizontal_spacing_per_row / n_columns,
        vertical_spacing=vertical_spacing_per_column / n_rows,
        subplot_titles=list(features),
        specs=[
            [{"secondary_y": True} for _ in range(n_columns)]
            for _ in range(n_rows)
        ],
    )
    for plot_index, feature in enumerate(features):
        row = plot_index // n_columns + 1
        column = plot_index % n_columns + 1
        feature_grid_ = create_pdp_grid_from_data(
            feature,
            data_for_pdp_grid_calculations,
            points_count=n_point_in_grid,
            calculation_strategy=grid_calculation_strategy,
        )
        feature_grid = np.array(sorted(feature_grid_))
        y_axis_values = _create_model_predictions_for_grid(
            model,
            feature,
            feature_grid,
            data=data_for_pdp_grid_calculations,
            n_sample_to_calculate_predictions=n_samples_to_calculate_predictions,
            random_state=random_state,
        )
        y_axis_range = _get_y_axis_range_for_pdp(y_axis_values, y_axis_range_mode)
        x_axis_range = _get_x_axis_range_for_pdp(feature_grid)
        _add_line_plot(
            fig=fig,
            grid_values=feature_grid,
            model_predictions=y_axis_values,
            row=row,
            column=column,
        )
        _add_average_line_plot(
            fig,
            feature_grid,
            y_axis_values,
            row,
            column,
        )
        add_histogram_trace(
            fig,
            data_for_pdp_grid_calculations[feature],
            row,
            column,
            visible=True,
        )
        _update_subplot_layout_for_pdp(
            fig,
            feature,
            feature_grid,
            y_axis_range,
            x_axis_range,
            y_axis_tick_values_precision,
            row,
            column,
        )
    return fig


def _get_ordered_features(
    model: SupportsModel,
    ordered_features: tp.Iterable[str] | None = None,
    feature_importance_for_order: _TFeatureImportanceDict | None = None,
    descending: bool = True,
) -> tp.List[str]:
    """
    Get list of ordered features.
    If specific order is not provided, features are ranked by importance.
    """
    if ordered_features is not None:
        provided_features = set(ordered_features)
        model_features = set(model.features_in)
        if not provided_features.issubset(model_features):
            not_model_features = provided_features - model_features
            raise ValueError(
                "Provided features `ordered_features` are not used"
                f" in the model: {not_model_features}",
            )
        return list(ordered_features)
    if feature_importance_for_order is not None:
        features_, order_by = zip(*feature_importance_for_order.items())
        features: list[str] = np.array(features_)[np.argsort(order_by)].tolist()
        if descending:
            features.reverse()
        return features
    return model.features_in


def _get_pdp_layout(
    height: int, width: int,
) -> go.Layout:
    """
    Get layout for partial dependency plots.
    """
    return go.Layout(
        height=height,
        width=width,
        title="Partial Dependency Plots",
        bargap=0,
        plot_bgcolor=_RGBA_COLOR_TRANSPARENT,
        legend=dict(
            orientation="h",
            x=LAYOUT_LEGEND_OFFSET_X,
            y=LAYOUT_LEGEND_OFFSET_Y,
            xanchor="left",
            yanchor="bottom",
        ),
        margin=dict(l=LAYOUT_MARGIN_L, r=LAYOUT_MARGIN_R, b=LAYOUT_MARGIN_B),
    )


def _create_tick_text_based_on_provided_values(
    tickvals: _TNumericNDArray[_P],
    num_ticks_to_show: int = 4,
    floating_point_precision: float = 0.2,
) -> tp.List[str]:
    """
    Creates ticks names based on the provided values.
    There will be `num_ticks_to_show` ticks in total with a
    ``floating_point_precision`` floating numbers precision.
    """
    show_tick_every = len(tickvals) // num_ticks_to_show
    ticktext = []
    for tick_index, tickval in enumerate(tickvals):
        show_tick = not tick_index % show_tick_every
        if show_tick:
            ticktext.append(f"{tickval:{floating_point_precision}f}")
        else:
            ticktext.append("")
    return ticktext


def _update_subplot_layout_for_pdp(
    fig: go.Figure,
    feature: str,
    x_axis_values: _TNumericNDArray[_P],
    y_axis_range: _TRange,
    x_axis_range: _TRange,
    y_axis_tick_values_precision: str,
    row: int,
    column: int,
    title_offset: float = 0.005,
) -> None:
    """
    Update layout for the subplot with partial dependency plot.
    """
    fig.update_xaxes(
        title="feature values",
        showline=True,
        linecolor="black",
        tickmode='array',
        tickvals=x_axis_values,
        ticktext=_create_tick_text_based_on_provided_values(x_axis_values),
        ticks="inside",
        tickwidth=3,
        hoverformat=".2f",
        range=x_axis_range,
        row=row,
        col=column,
    )
    fig.update_yaxes(
        title="model prediction",
        showline=True,
        linecolor="black",
        ticks="outside",
        hoverformat=".2f",
        range=y_axis_range,
        tickformat=y_axis_tick_values_precision,
        secondary_y=False,
        row=row,
        col=column,
    )
    subplot_title = next(
        fig.select_annotations(selector={"text": f"{feature}"}),
    )
    subplot_title.text = f"<b>{subplot_title.text}</b>"
    subplot_title.y += title_offset


def _add_line_plot(
    fig: go.Figure,
    grid_values: _TNumericNDArray[_P],
    model_predictions: tp.List[_TNumericNDArray[_P]],
    row: int,
    column: int,
) -> None:
    """
    Add line plot on a figure for every model prediction.
    Name is chosen in that way to be shown in hover.
    """
    for prediction in model_predictions:
        fig.add_trace(
            go.Scattergl(
                x=grid_values,
                y=prediction,
                mode="lines",
                line={"color": _RGBA_COLOR_GREY, "width": 0.5},
                showlegend=False,
                name="feature value<br>model prediction",
                hovertemplate="%{x:.2f}<br>%{y:.2f}",
            ),
            row=row,
            col=column,
        )


def _add_average_line_plot(
    fig: go.Figure,
    feature_grid: _TNumericNDArray[_P],
    model_predictions: tp.List[_TNumericNDArray[_P]],
    row: int,
    column: int,
) -> None:
    """
    Calculate an average prediction and add as a trace on a figure.
    """
    only_predictions = np.array(model_predictions)
    average_predictions = only_predictions.mean(axis=0)
    is_first = row == 1 and column == 1
    fig.add_trace(
        go.Scattergl(
            x=feature_grid,
            y=average_predictions,
            mode="lines",
            line={"color": _RGBA_COLOR_BLUE, "width": 3},
            showlegend=is_first,
            name="average",
            hovertemplate="%{x:.2f}<br>%{y:.2f}",
            legendgroup="average",
        ),
        row=row,
        col=column,
    )


def _get_y_axis_range_for_pdp(
    y_axis_values: tp.List[_TNumericNDArray[_P]],
    y_axis_range_mode: tp.Literal["average", "all"] = "average",
    relative_offset_range_multiplier: float = _RELATIVE_OFFSET_RANGE_MULTIPLIER,
) -> _TRange:
    """
    By default, the range is chosen in that way to center the average prediction line.
    Alternatively, the range can be chosen to include all predictions.
    """
    if y_axis_range_mode == "all":
        y_axis_array = np.concatenate(y_axis_values, axis=0)
        min_y_axis = y_axis_array.min()
        max_y_axis = y_axis_array.max()
    elif y_axis_range_mode == "average":
        average_y_axis_value = np.stack(y_axis_values, axis=1).mean(axis=1)
        min_y_axis = average_y_axis_value.min()
        max_y_axis = average_y_axis_value.max()
    else:
        raise ValueError(
            "Supported values for y_axis_range_mode:"
            f" 'all', 'average', passed {y_axis_range_mode=}",
        )
    offset = relative_offset_range_multiplier * (max_y_axis - min_y_axis)
    y_axis_range = (
        min_y_axis - offset,
        max_y_axis + offset,
    )
    return y_axis_range  # noqa: WPS331  # Naming makes meaning clearer


def _get_x_axis_range_for_pdp(feature_grid: _TNumericNDArray[_P]) -> _TRange:
    return min(feature_grid), max(feature_grid)
