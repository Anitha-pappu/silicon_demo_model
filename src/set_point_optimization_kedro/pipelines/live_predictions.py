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

from kedro.pipeline import Pipeline, node

from modeling import (
    calculate_model_prediction_bounds,
    calculate_model_predictions,
)
from optimus_core.utils import partial_wrapper
from preprocessing import create_detectors_dict, detect_data_anomaly
from recommend import (
    get_value_after_recs_counterfactual,
    get_value_after_recs_impact,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_recent_live_data() -> Pipeline:
    """
    Get the most recent data from the model_input_data pipeline.
    This is a mock-up to be used for live inference.
    In real life, this would be a node that reads from a database.
    One thing to notice is that the fillna node is added to the pipeline
    to ensure that the data is always filled for recommendation. However,
    how to deal with missing data in recommendation is a business decision
    and should be discussed with the client.
    """
    return Pipeline(
        nodes=[
            node(
                # Create a mock-up of the most recent raw data
                func=lambda data: data.iloc[-100:].set_index("timestamp"),
                inputs={
                    "data": "post_enforce_schema",
                },
                outputs="live_monitoring_data",
            ),
            node(
                func=lambda data: data.fillna(method="ffill"),
                inputs={
                    "data": "model_input_data",
                },
                outputs="model_input_data_filled",
            ),
            node(
                func=lambda data: data.iloc[[-1]],
                inputs={
                    "data": "model_input_data_filled",
                },
                outputs="test_data",  # This is actually the live data
            ),
        ],
    ).tag("recent_live_data")


def get_live_prediction_and_monitoring_steps() -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                create_detectors_dict,
                inputs={
                    "anomaly_parameters": "params:live_monitoring.detectors",
                    "tags_to_monitor": "params:live_monitoring.tags_to_monitor",
                },
                outputs="detectors",
                name="create_detectors",
            ),
            node(
                detect_data_anomaly,
                inputs={
                    "data": "live_monitoring_data",
                    "anomaly_detectors": "detectors",
                },
                outputs="anomalies_table",
                name="create_data_anomaly_table",
            ),
            node(
                partial_wrapper(
                    calculate_model_predictions,
                    keep_input=True,
                ),
                inputs={
                    "data": "test_data",
                    "model": "trained_model",
                    "target_column": "params:silica_model.train.target_column",
                },
                outputs="test_data_with_predictions",
                name="test_data_with_predictions",
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
            node(
                calculate_model_prediction_bounds,
                inputs={
                    "data": "test_data",  # This is actually the live data
                    "model": "trained_model",
                    "model_metrics": "test_metrics",
                    "error_metric": "params:model_monitoring.error_metric",
                    "error_multiplier": "params:model_monitoring.error_multiplier",
                },
                outputs="model_prediction_bounds",
                name="create_model_performance_tracking_metrics",
            ),
            node(
                get_value_after_recs_counterfactual,
                inputs={
                    "counterfactual_type": "params:impact.counterfactual_type",
                    "solutions": "solutions",
                    "data": "test_data",
                    "actual_value": "params:impact.target_col",
                    "datetime_column": "params:impact.datetime_col",
                },
                outputs="value_after_recs_opt",
                name="get_value_after_recs_opt",
            ),
            node(
                get_value_after_recs_impact,
                inputs={
                    "data": "test_data",
                    "value_after_recs": "params:impact.target_col",
                    "datetime_column": "params:impact.datetime_col",
                },
                outputs="value_after_recs_act",
                name="get_value_after_recs_act",
            ),
        ],
    ).tag("live_prediction")
