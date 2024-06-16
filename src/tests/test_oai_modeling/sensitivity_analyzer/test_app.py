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
from jupyter_dash import JupyterDash

from modeling import SklearnModel
from modeling.sensitivity_analyser import get_sensitivity_analyzer_app


class TestEndToEndApplicationRun(object):
    """
    Test application can be created with different combination of input parameters.
    """
    def test_end_to_end_run_on_sample_data_with_single_values_in_grid(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        manageable_conditions = {
            '% Iron Feed': [52],
        }
        features_to_plot = {
            '% Silica Feed': [52],
        }
        app = get_sensitivity_analyzer_app(
            features_to_plot,
            manageable_conditions,
            trained_sklearn_model,
            static_feature_values=simple_data.iloc[-1].to_dict(),
            data=simple_data,
        )
        assert isinstance(app, JupyterDash)

    def test_end_to_end_automatic_grid_computation_features_to_plot(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        manageable_conditions = {'% Iron Feed': [52, 50, 51, 53]}
        features_to_plot = {'% Silica Feed': None}
        app = get_sensitivity_analyzer_app(
            features_to_plot,
            manageable_conditions,
            trained_sklearn_model,
            static_feature_values=simple_data.iloc[-1].to_dict(),
            data=simple_data,
        )
        assert isinstance(app, JupyterDash)

    def test_end_to_end_automatic_grid_computation_manageable_condition(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        manageable_conditions = {'% Iron Feed': None}
        features_to_plot = {'% Silica Feed': [52, 50, 51, 53]}
        app = get_sensitivity_analyzer_app(
            features_to_plot,
            manageable_conditions,
            trained_sklearn_model,
            static_feature_values=simple_data.iloc[-1].to_dict(),
            data=simple_data,
        )
        assert isinstance(app, JupyterDash)

    def test_end_to_end_automatic_grid_computation_from_list(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        manageable_conditions = ['% Iron Feed']
        features_to_plot = ['% Silica Feed']
        app = get_sensitivity_analyzer_app(
            features_to_plot,
            manageable_conditions,
            trained_sklearn_model,
            static_feature_values=simple_data.iloc[-1].to_dict(),
            data=simple_data,
        )
        assert isinstance(app, JupyterDash)

    def test_end_to_end_manageable_condition_equals_to_feature_to_plot(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        manageable_conditions = ['% Iron Feed']
        features_to_plot = ['% Iron Feed']
        app = get_sensitivity_analyzer_app(
            features_to_plot,
            manageable_conditions,
            trained_sklearn_model,
            static_feature_values=simple_data.iloc[-1].to_dict(),
            data=simple_data,
        )
        assert isinstance(app, JupyterDash)

    def test_end_to_end_multiple_conditions_and_features(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        manageable_conditions = ['% Iron Feed', '% Silica Feed']
        features_to_plot = ['% Iron Feed', '% Silica Feed']
        app = get_sensitivity_analyzer_app(
            features_to_plot,
            manageable_conditions,
            trained_sklearn_model,
            static_feature_values=simple_data.iloc[-1].to_dict(),
            data=simple_data,
        )
        assert isinstance(app, JupyterDash)


class TestEndToEndApplicationRunWithLayoutCustomization(object):
    def test_initial_saved_slider_position(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        manageable_conditions = ['% Iron Feed', '% Silica Feed']
        features_to_plot = ['% Iron Feed', '% Silica Feed']
        app = get_sensitivity_analyzer_app(
            features_to_plot,
            manageable_conditions,
            trained_sklearn_model,
            static_feature_values=simple_data.iloc[-1].to_dict(),
            initial_slider_values=simple_data.iloc[-1].to_dict(),
            data=simple_data,
        )
        assert isinstance(app, JupyterDash)

    def test_features_to_plot_grid_config(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        manageable_conditions = ['% Iron Feed', '% Silica Feed']
        features_to_plot = ['% Iron Feed', '% Silica Feed']
        app = get_sensitivity_analyzer_app(
            features_to_plot,
            manageable_conditions,
            trained_sklearn_model,
            static_feature_values=simple_data.iloc[-1].to_dict(),
            initial_slider_values=simple_data.iloc[-1].to_dict(),
            data=simple_data,
            features_to_plot_grid_config={
                "points_count": 10,
                "calculation_strategy": "uniform",
            },
        )
        assert isinstance(app, JupyterDash)

    def test_manageable_conditions_grid_config(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        manageable_conditions = ['% Iron Feed', '% Silica Feed']
        features_to_plot = ['% Iron Feed', '% Silica Feed']
        app = get_sensitivity_analyzer_app(
            features_to_plot,
            manageable_conditions,
            trained_sklearn_model,
            static_feature_values=simple_data.iloc[-1].to_dict(),
            initial_slider_values=simple_data.iloc[-1].to_dict(),
            data=simple_data,
            features_to_manipulate_grid_config={
                "points_count": 10,
                "calculation_strategy": "quantiles",
            },
        )
        assert isinstance(app, JupyterDash)

    def test_show_histogram(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        manageable_conditions = ['% Iron Feed', '% Silica Feed']
        features_to_plot = ['% Iron Feed', '% Silica Feed']
        app = get_sensitivity_analyzer_app(
            features_to_plot,
            manageable_conditions,
            trained_sklearn_model,
            static_feature_values=simple_data.iloc[-1].to_dict(),
            initial_slider_values=simple_data.iloc[-1].to_dict(),
            data=simple_data,
            layout_config={"show_histogram": False},
        )
        assert isinstance(app, JupyterDash)
