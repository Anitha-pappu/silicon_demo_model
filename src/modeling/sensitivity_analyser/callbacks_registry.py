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

from dash import Input, Output, ctx, html
from dash.exceptions import PreventUpdate
from jupyter_dash import JupyterDash

from .components.sidebar.slider_utils import create_marks_for_slider
from .configs import LayoutConfig
from .model_predictions import ModelPrediction

CALLBACK_ID_BUTTON = "id-button"
CALLBACK_ID_RESET_BUTTON = "id-reset-button"
CALLBACK_ID_INFO_ICON = "id-info-icon"
CALLBACK_ID_CONDITION_ANNOTATION = "id-condition-annotation-{condition}"
CALLBACK_ID_CONDITION_SLIDER = "id-condition-slider-{condition}"
CALLBACK_ID_GRAPH = "id-graph"
CALLBACK_ID_STORE = "id-store"

TData = tp.Dict[str, tp.List[tp.List[float]]]
TIndices = tp.List[int]
TPointsToUseCount = int
TExtendDataCallbackType = tp.Tuple[TData, TIndices, TPointsToUseCount]


def register_update_store_main_button_callback(
    app: JupyterDash,
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
) -> None:
    """
    Registries for callback that updates data in `dcc.Store` component
    when main button is pressed. Data in `dcc.Store` is updated
    with the current position of controls
    """
    @app.callback(  # type: ignore
        Output(CALLBACK_ID_STORE, "data"),
        Input(CALLBACK_ID_BUTTON, "n_clicks"),
        *[
            Input(CALLBACK_ID_CONDITION_SLIDER.format(condition=condition), "value")
            for condition in features_to_manipulate
        ],
    )
    def wrapper(n_clicks: int, *conditions_values: int) -> tp.Dict[str, int]:
        if CALLBACK_ID_BUTTON != ctx.triggered_id and ctx.triggered_id is not None:
            raise PreventUpdate()
        return dict(zip(features_to_manipulate, conditions_values))


def register_update_store_reset_button_callback(
    app: JupyterDash,
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
    initial_slider_values: tp.Mapping[str, float],
) -> None:
    """
    Registries for callback that updates data in `dcc.Store` component
    when reset button is pressed. Data in `dcc.Store` is updated
    with the initial slider values that come from user input
    """
    @app.callback(  # type: ignore
        Output(CALLBACK_ID_STORE, "data", allow_duplicate=True),
        Input(CALLBACK_ID_RESET_BUTTON, "n_clicks"),
        prevent_initial_call=True,
    )
    def wrapper(n_clicks: int) -> tp.Dict[str, int]:
        return {
            condition: condition_grid.index(initial_slider_values[condition])
            for condition, condition_grid in features_to_manipulate.items()
        }


def register_update_slider_annotation_callback(
    app: JupyterDash,
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
    layout_config: LayoutConfig,
) -> None:
    """
    Registries for callback that updates the slider annotation on top of the sliders
    every time the slider position is changed.
    """
    @app.callback(  # type: ignore
        *[
            Output(
                # suppressing WPS441 since false positive
                CALLBACK_ID_CONDITION_ANNOTATION.format(condition=condition),  # noqa: WPS441,E501
                "children",
            )
            for condition in features_to_manipulate
        ],
        *[
            Input(
                # suppressing WPS441 since false positive
                CALLBACK_ID_CONDITION_SLIDER.format(condition=condition),  # noqa: WPS441,E501
                "value",
            )
            for condition in features_to_manipulate
        ],
    )
    def wrapper(*condition_values: int) -> tp.List[html.P]:
        components_to_return = []
        for condition, condition_value in zip(features_to_manipulate, condition_values):
            condition_name = (
                layout_config.visualization_mapping[condition]
                if condition in layout_config.visualization_mapping
                else condition
            )
            slider_annotation = features_to_manipulate[condition][condition_value]
            slider_annotation_round = (
                layout_config.slider_annotation_round[condition]
                if condition in layout_config.slider_annotation_round else 4
            )
            slider_annotation = round(slider_annotation, slider_annotation_round)
            components_to_return.append(
                html.P([
                    html.Span(f"{condition_name}: ", style={"margin-left": "45px"}),
                    html.Span(slider_annotation),
                ]),
            )
        return components_to_return


def register_update_slider_marks_callback(
    app: JupyterDash,
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
    initial_slider_values: tp.Mapping[str, float],
    layout_config: LayoutConfig,
) -> None:
    """
    Registers for callback that updates slider marks every time
    data in the `dcc.Store` is updated. This, correspondingly,
    happens when main button is pressed.
    """
    @app.callback(  # type: ignore
        [
            # suppressing WPS441 since false positive
            Output(CALLBACK_ID_CONDITION_SLIDER.format(condition=condition), "marks")  # noqa: WPS441,E501
            for condition in features_to_manipulate
        ],
        Input(CALLBACK_ID_STORE, "data"),
    )
    def wrapper(store_data: tp.Dict[str, int]) -> tp.List[tp.Dict[int, tp.Any]]:
        sliders_to_return = []
        for condition, grid_values in features_to_manipulate.items():
            sliders_to_return.append(
                create_marks_for_slider(
                    grid_values,
                    initial_slider_values,
                    layout_config,
                    condition,
                    store_data,
                ),
            )
        return sliders_to_return


def register_update_trace_from_slider_callback(
    app: JupyterDash,
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
    model_predictions: tp.Mapping[str, ModelPrediction],
) -> None:
    """
    Registries for callback that updates current (blue)
    traces on the sensitivity charts
    when slider positions are changed.
    """
    @app.callback(  # type: ignore
        Output(CALLBACK_ID_GRAPH, "extendData"),
        *[
            Input(CALLBACK_ID_CONDITION_SLIDER.format(condition=condition), "value")
            for condition in features_to_manipulate
        ],
    )
    def wrapper(*condition_values: int) -> TExtendDataCallbackType:
        points_to_extent: TData = {"x": [], "y": []}
        trace_indices = []
        max_points_to_show = 0
        for plot_parameter_index, plot_parameter in enumerate(model_predictions):
            prediction = model_predictions[plot_parameter]
            points_to_extent["x"].append(
                prediction.get_model_feature_values_at_index(condition_values),
            )
            points_to_extent["y"].append(
                prediction.get_model_predictions_at_index(condition_values),
            )
            trace_indices.append(plot_parameter_index + 2 * len(model_predictions))
            max_points_to_show = prediction.calculate_prediction_size_at_index(
                condition_values,
            )
        return points_to_extent, trace_indices, max_points_to_show


def register_update_trace_from_button_callback(
    app: JupyterDash,
    model_predictions: tp.Mapping[str, ModelPrediction],
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
) -> None:
    """
    Registries for callback that updates saved (black)
    traces on the sensitivity charts
    when store data in `dcc.Store` is changed
    (this happens when main button is pressed)
    """
    @app.callback(  # type: ignore
        Output(CALLBACK_ID_GRAPH, "extendData", allow_duplicate=True),
        Input(CALLBACK_ID_STORE, "data"),
        prevent_initial_call=True,
    )
    def wrapper(store_data: tp.Dict[str, int]) -> TExtendDataCallbackType:
        points_to_extent: TData = {"x": [], "y": []}
        trace_indices = []
        max_points_to_show = 0
        for plot_parameter_index, plot_parameter in enumerate(model_predictions):
            condition_index = tuple(
                store_data[condition]
                for condition in features_to_manipulate
            )
            prediction = model_predictions[plot_parameter]
            points_to_extent["x"].append(
                prediction.get_model_feature_values_at_index(condition_index),
            )
            points_to_extent["y"].append(
                prediction.get_model_predictions_at_index(condition_index),
            )
            trace_indices.append(plot_parameter_index + len(model_predictions))
            max_points_to_show = prediction.calculate_prediction_size_at_index(
                condition_index,
            )
        return points_to_extent, trace_indices, max_points_to_show
