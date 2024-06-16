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

import warnings

from kedro.pipeline import Pipeline, node, pipeline

from modeling import (
    calculate_feature_importance,
    calculate_metrics,
    calculate_model_predictions,
    calculate_shap_feature_importance,
    create_model,
    create_model_factory,
    create_splitter,
    create_tuner,
    cross_validate,
    drop_nan_rows,
    split_data,
    train_model,
    tune_model,
)
from modeling.kedro_utils import convert_metrics_to_nested_mlflow_dict
from modeling.report import get_modeling_overview
from optimus_core.utils import partial_wrapper
from reporting.rendering import generate_html_report

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_modeling_steps() -> Pipeline:
    return Pipeline(
        [
            node(
                drop_nan_rows,
                inputs={
                    "data": "model_input_data",
                    "target_column": "params:train.target_column",
                    "model_features": "params:train.model_features",
                },
                outputs="data_dropna",
                name="drop_nan_rows",
            ),
            node(
                create_splitter,
                inputs={
                    "split_method": "params:split.split_method",
                    "splitting_parameters": "params:split.split_parameters",
                },
                outputs="splitter",
                name="create_splitter",
            ),
            node(
                split_data,
                inputs={
                    "data": "data_dropna",
                    "splitter": "splitter",
                },
                outputs=["train_data", "test_data"],
                name="split_data",
            ),
            node(
                create_model_factory,
                inputs={
                    "model_factory_type": "params:train.factory_class_name",
                    "model_init_config": "params:train.init",
                    "features": "params:train.model_features",
                    "target": "params:train.target_column",
                },
                outputs="model_factory",
            ),
            node(
                create_model,
                inputs={"model_factory": "model_factory"},
                outputs="model",
            ),
            node(
                create_tuner,
                inputs={
                    "model_factory": "model_factory",
                    "model_tuner_type": "params:tune.class_name",
                    "tuner_config": "params:tune.tuner",
                },
                outputs="model_tuner",
            ),
            node(
                tune_model,
                inputs={
                    "model_tuner": "model_tuner",
                    "hyperparameters_config": "params:tune.hyperparameters",
                    "data": "train_data",
                },
                outputs="tuned_model",
            ),
            node(
                train_model,
                inputs={
                    "model": "tuned_model",
                    "data": "train_data",
                },
                outputs="trained_model",
                name="train_model",
            ),
            node(
                partial_wrapper(
                    calculate_model_predictions,
                    keep_input=True,
                    add_error=True,
                ),
                inputs={
                    "data": "train_data",
                    "model": "trained_model",
                    "target_column": "params:train.target_column",
                },
                outputs="train_data_with_predictions",
                name="train_data_with_predictions",
            ),
            node(
                partial_wrapper(
                    calculate_model_predictions,
                    keep_input=True,
                    add_error=True,
                ),
                inputs={
                    "data": "test_data",
                    "model": "trained_model",
                    "target_column": "params:train.target_column",
                },
                outputs="test_data_with_predictions",
                name="test_data_with_predictions",
            ),
            node(
                calculate_metrics,
                inputs={
                    "data": "train_data",
                    "model": "trained_model",
                },
                outputs="train_metrics",
                name="create_train_metrics",
            ),
            node(
                calculate_metrics,
                inputs={
                    "data": "test_data",
                    "model": "trained_model",
                },
                outputs="test_metrics",
                name="create_test_metrics",
            ),
            node(
                cross_validate,
                inputs={
                    "model": "trained_model",
                    "data": "data_dropna",
                    "cv_strategy_config": "params:cross_validate.cv_strategy_config",
                },
                outputs="cross_validation_scores",
                name="cross_validate",
            ),
            # This node converts ``test_metrics`` to nested `Dict` format that can be
            # mapped to ``kedro_mlflow.io.metrics.MlflowMetricsDataSet`` in catalog.
            # Can be safely deleted if MLflow is not used.
            node(
                convert_metrics_to_nested_mlflow_dict,
                inputs="test_metrics",
                outputs="test_metrics_for_mlflow",
                name="convert_metrics_to_nested_mlflow_dict",
            ),
            node(
                calculate_feature_importance,
                inputs={
                    "data": "train_data",
                    "model": "trained_model",
                },
                outputs="feature_importance",
                name="feature_importance",
            ),
            node(
                calculate_shap_feature_importance,
                inputs={
                    "data": "train_data",
                    "shap_producer": "trained_model",
                },
                outputs="shap_feature_importance",
                name="shap_feature_importance",
            ),
        ],
    ).tag("modeling")


def get_silica_model_steps() -> Pipeline:
    return Pipeline(
        nodes=[
            pipeline(
                pipe=get_modeling_steps(),
                inputs={"model_input_data": "model_input_data"},
                outputs={
                    "trained_model": "trained_model",
                    "train_data": "train_data",
                    "test_data": "test_data",
                    "train_data_with_predictions": "train_data_with_predictions",
                    "test_data_with_predictions": "test_data_with_predictions",
                    "train_metrics": "train_metrics",
                    "test_metrics": "test_metrics",
                    "cross_validation_scores": "cross_validation_scores",
                    "test_metrics_for_mlflow": "test_metrics_for_mlflow",
                    "feature_importance": "feature_importance",
                    "shap_feature_importance": "shap_feature_importance",
                },
                namespace="silicon_model",
            ),
        ],
    )


def get_baseline_model_steps() -> Pipeline:
    # In this tutorial, we assume that the results of the optimization from the previous
    # section have been implemented in the plant and that the input data from those
    # timestamps are the readings after such implementation. In a real situation, it is
    # expected that the modeling test results are not implemented.
    # Data used for each node might be different from this based on that.
    return Pipeline(
        nodes=[
            pipeline(
                pipe=get_modeling_steps(),
                inputs={"model_input_data": "train_data"},
                outputs={
                    "trained_model": "baseline_trained_model",
                    "train_data": "baseline_train_data",
                    "test_data": "baseline_test_data",
                    "train_data_with_predictions":
                        "baseline_train_data_with_baseline_predictions",
                    "test_data_with_predictions":
                        "baseline_test_data_with_baseline_predictions",
                    "train_metrics": "baseline_train_metrics",
                    "test_metrics": "baseline_test_metrics",
                    "cross_validation_scores": "baseline_cross_validation_scores",
                    "test_metrics_for_mlflow": "baseline_test_metrics_for_mlflow",
                    "feature_importance": "baseline_feature_importance",
                    "shap_feature_importance": "baseline_shap_feature_importance",
                },
                namespace="baseline_model",
            ),
            node(
                partial_wrapper(
                    calculate_model_predictions,
                    keep_input=True,
                ),
                inputs={
                    "data": "test_data",
                    "model": "baseline_trained_model",
                },
                outputs="test_data_with_baseline_predictions",
                name="test_data_with_baseline_predictions",
            ),
        ],
    )


def _build_reporting_for_modeling_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                get_modeling_overview,
                inputs={
                    "model": "trained_model",
                    "timestamp_column": "params:split.split_parameters.datetime_column",
                    "train_data": "train_data",
                    "test_data": "test_data",
                },
                outputs="model_performance_report",
                name="generate_performance_figures",
            ),
            node(
                generate_html_report,
                inputs={
                    "report_structure": "model_performance_report",
                    "render_path": "params:model_performance_report.render_path",
                    "report_meta_data": "params:model_performance_report.report_meta_data",
                },
                outputs=None,
                name="generate_model_performance_report",
            ),
        ],
    )


def get_post_modeling_reporting_steps() -> Pipeline:
    return Pipeline(
        nodes=[
            pipeline(
                pipe=_build_reporting_for_modeling_pipeline(),
                inputs={
                    "trained_model": "trained_model",
                    "train_data": "train_data",
                    "test_data": "test_data",
                },
                namespace="silicon_model",
            ),
        ],
    ).tag(["modeling", "reporting"])
