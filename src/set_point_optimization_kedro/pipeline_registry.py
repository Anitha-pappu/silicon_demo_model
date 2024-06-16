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


"""Usecase registry."""

import typing as tp
import warnings

from kedro.pipeline import Pipeline

from .pipelines import (
    create_feature_report,
    get_baseline_model_steps,
    get_export_steps,
    get_feature_factory_steps,
    get_impact_steps,
    get_live_prediction_and_monitoring_steps,
    get_post_modeling_reporting_steps,
    get_preprocessing_steps,
    get_recent_live_data,
    get_recommend_report_steps,
    get_recommend_steps,
    get_silica_model_steps,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def register_pipelines() -> tp.Dict[str, Pipeline]:  # noqa: WPS210
    """Create the usecase's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    preprocessing_steps = get_preprocessing_steps()
    feature_factory_steps = get_feature_factory_steps()
    modeling_steps = get_silica_model_steps()
    reporting_steps = get_post_modeling_reporting_steps()
    recommend_steps = get_recommend_steps()
    recommend_report_steps = get_recommend_report_steps()
    baseline_steps = get_baseline_model_steps()
    impact_steps = get_impact_steps()
    export_steps = get_export_steps()

    default_pipeline = (
        preprocessing_steps
        + feature_factory_steps
        + modeling_steps
        + reporting_steps
        # + recommend_steps
        # + recommend_report_steps
        # + baseline_steps
        # + impact_steps
    )

    inference_pipeline = (
        preprocessing_steps
        + feature_factory_steps
        + get_recent_live_data()
        + get_live_prediction_and_monitoring_steps()
        + recommend_steps
        + export_steps
    )

    return {
        "__default__": default_pipeline,
        "live_inference": inference_pipeline,
        "feature_report": create_feature_report(),
    }
