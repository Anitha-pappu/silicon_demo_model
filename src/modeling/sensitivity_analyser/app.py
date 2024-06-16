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

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from jupyter_dash import JupyterDash

from ..models import ModelBase
from .callbacks_registry import (
    CALLBACK_ID_STORE,
    register_update_slider_annotation_callback,
    register_update_slider_marks_callback,
    register_update_store_main_button_callback,
    register_update_store_reset_button_callback,
    register_update_trace_from_button_callback,
    register_update_trace_from_slider_callback,
)
from .components.content.content import create_content
from .components.sidebar.sidebar import create_sidebar
from .configs import (
    FeaturesToManipulateConfig,
    FeaturesToPlotConfig,
    LayoutConfig,
    TParameters,
    TRawConfig,
)
from .grid import verify_provided_grid
from .model_predictions import (
    ModelPrediction,
    make_model_predictions_for_features_to_plot,
)


def get_sensitivity_analyzer_app(
    features_to_plot: TParameters,
    features_to_manipulate: TParameters,
    model: ModelBase,
    static_feature_values: tp.Optional[tp.Mapping[str, float]] = None,
    initial_slider_values: tp.Optional[tp.Mapping[str, float]] = None,
    data: tp.Optional[pd.DataFrame] = None,
    features_to_plot_grid_config: TRawConfig = None,
    features_to_manipulate_grid_config: TRawConfig = None,
    layout_config: TRawConfig = None,
) -> JupyterDash:
    """
    Produce sensitivity analyzer dash application
    for understanding the model behaviour

    Args:
        features_to_plot: model features for building the sensitivity analyzer plots
        model: instance of the trained model
        features_to_manipulate: model input parameters for
         interactive manipulation through slider in Dash application
        static_feature_values: values for other model features
         that are required for making a prediction with model
         that are not present in `features_to_plot` or `manageable_conditions`
        initial_slider_values: initial parameters for saved slider position
        data: historical data used for grid calculation and histogram bins calculation
        features_to_plot_grid_config: configuration dict for
         grid calculation for ``features_to_plot``;
         ``modeling.sensitivity_analyzer.configs.FeaturesToPlotConfig``
         is used to parse the provided dict
        features_to_manipulate_grid_config: configuration dict for
         grid calculation for ``manageable_conditions``;
         ``modeling.sensitivity_analyzer.configs.ManageableConditionsConfig``
         is used to parse the provided dict
        layout_config: configuration dict for application and charts layout:
         ``modeling.sensitivity_analyzer.configs.LayoutConfig``
         is used to parse the provided dict

    Returns:
        Sensitivity analyzer Dash application
    """
    features_to_plot_grid_dataclass = FeaturesToPlotConfig(
        **features_to_plot_grid_config if features_to_plot_grid_config else {},
    )
    features_to_manipulate_dataclass = FeaturesToManipulateConfig(
        **features_to_manipulate_grid_config
        if features_to_manipulate_grid_config else {},
    )
    layout_dataclass = LayoutConfig(
        **layout_config if layout_config else {},
    )
    _validate_user_inputs(
        features_to_plot,
        features_to_manipulate,
        model,
        data,
        layout_dataclass,
        static_feature_values,
    )
    initial_slider_values = initial_slider_values or {}
    static_feature_values = static_feature_values or {}
    features_to_plot = verify_provided_grid(
        features_to_plot,
        data,
        features_to_plot_grid_dataclass,
        initial_slider_values,
    )
    features_to_manipulate = verify_provided_grid(
        features_to_manipulate,
        data,
        features_to_manipulate_dataclass,
        initial_slider_values,
    )
    model_predictions = make_model_predictions_for_features_to_plot(
        features_to_plot,
        features_to_manipulate,
        model,
        static_feature_values,
    )
    return _create_app(
        features_to_manipulate,
        initial_slider_values,
        model_predictions,
        data,
        model,
        layout_dataclass,
    )


def _create_app(
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
    initial_slider_values: tp.Mapping[str, float],
    model_predictions: tp.Mapping[str, ModelPrediction],
    data: pd.DataFrame,
    model: ModelBase,
    layout_config: LayoutConfig,
) -> JupyterDash:
    """
    Set up Dash application, when user inputs
    are validated and model predictions are stored
    in the corresponding data structure ``ModelPrediction``
    """
    app = JupyterDash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
    app.layout = html.Div([
        dcc.Store(id=CALLBACK_ID_STORE),
        create_sidebar(
            features_to_manipulate,
            initial_slider_values,
            layout_config,
        ),
        create_content(
            model_predictions,
            data,
            model,
            layout_config,
        ),
    ])
    register_update_store_main_button_callback(app, features_to_manipulate)
    register_update_store_reset_button_callback(
        app,
        features_to_manipulate,
        initial_slider_values,
    )
    register_update_slider_annotation_callback(
        app,
        features_to_manipulate,
        layout_config,
    )
    register_update_slider_marks_callback(
        app,
        features_to_manipulate,
        initial_slider_values,
        layout_config,
    )
    register_update_trace_from_slider_callback(
        app,
        features_to_manipulate,
        model_predictions,
    )
    register_update_trace_from_button_callback(
        app,
        model_predictions,
        features_to_manipulate,
    )
    return app


def _validate_user_inputs(
    features_to_plot: TParameters,
    features_to_manipulate: TParameters,
    model: ModelBase,
    data: tp.Optional[pd.DataFrame],
    layout_config: LayoutConfig,
    static_feature_values: tp.Optional[tp.Mapping[str, float]] = None,
) -> None:
    if data is None:
        _validate_data_is_needed_for_grid_composition(
            features_to_plot,
            features_to_manipulate,
        )
        _validate_data_is_needed_for_histogram(layout_config)
    _validate_static_feature_values_are_required(
        model,
        static_feature_values,
        features_to_manipulate,
    )


# WPS238 Splitting this logic
# into separate functions will reduce readability
def _validate_data_is_needed_for_grid_composition(  # noqa: WPS238
    features_to_plot: TParameters,
    features_to_manipulate: TParameters,
) -> None:
    error_message = (
        "Grid for parameters {no_grid_features} is not provided."
        " Please either provide data or fill the"
        " grid for the parameters listed above."
    )
    if isinstance(features_to_plot, list):
        raise ValueError(error_message.format(no_grid_features=features_to_plot))
    if isinstance(features_to_manipulate, list):
        raise ValueError(
            error_message.format(no_grid_features=features_to_manipulate),
        )
    features_to_plot_without_grid = [
        feature_name
        for feature_name, feature_grid in features_to_plot.items()
        if feature_grid is None
    ]
    if features_to_plot_without_grid:
        raise ValueError(
            error_message.format(no_grid_features=features_to_plot_without_grid),
        )
    features_to_manipulate_without_grid = [
        feature_name
        for feature_name, feature_grid in features_to_manipulate.items()
        if feature_grid is None
    ]
    if features_to_manipulate_without_grid:
        raise ValueError(
            error_message.format(no_grid_features=features_to_manipulate_without_grid),
        )


def _validate_data_is_needed_for_histogram(
    layout_config: LayoutConfig,
) -> None:
    if layout_config.show_histogram:
        raise ValueError(
            "Histogram layout on the background of the sensitivity charts requires"
            " historical data. Please either disable histogram calculation using"
            " `layout_config` parameter or provide data.",
        )


def _validate_static_feature_values_are_required(
    model: ModelBase,
    static_feature_values: tp.Optional[tp.Mapping[str, float]],
    features_to_manipulate: TParameters,
) -> None:
    required_static_features = (
        set(model.features_in) - set(features_to_manipulate)
    )
    if required_static_features:
        if static_feature_values is None:
            raise ValueError(
                "Missing `static_feature_values` argument to calculate prediction."
                f" Provide values for {required_static_features} using"
                " `static_feature_values` input argument",
            )
        missing_static_values = required_static_features - set(static_feature_values)
        if missing_static_values:
            raise ValueError(
                f"Missing static feature values for {missing_static_values}."
                " Please provide using `static_feature_values` input argument.",
            )
