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


import pytest

from modeling import SklearnPipelineFactory, SklearnPipelineTuner

N_FEATURES = 10


@pytest.fixture(scope="function")
def sklearn_pipeline_init_config_preserve_columns_wrapper():
    return {
        "transformers": [
            {
                "name": "scaler",
                "class_name": "sklearn.preprocessing.StandardScaler",
                "kwargs": {},
                "wrapper": "preserve_columns",
            },
        ],
        "estimator": {
            "class_name": "sklearn.linear_model.Lasso",
            "kwargs": {
                "random_state": 0,
                "fit_intercept": True,
                "alpha": 0.1,
            },
        },
    }


@pytest.fixture(scope="function")
def tuner_init_config():
    return {
        #  `ModelTunerBase` inheritor class that
        #  performs model hyperparameters tuning
        "class_name": "sklearn.model_selection.GridSearchCV",
        "kwargs": {
            "n_jobs": -1,
            "refit": "mae",
            "param_grid": {
                "estimator__alpha": [0.01, 0.1, 1],
            },
            "scoring": {
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
        },
        "hyperparameters": None,
    }


def test_sklearn_pipeline_tuner_produces_pipeline_with_features_out(
    regression_data,
    sklearn_pipeline_init_config_preserve_columns_wrapper,
    tuner_init_config,
):
    sklearn_pipeline_factory = SklearnPipelineFactory(
        sklearn_pipeline_init_config_preserve_columns_wrapper,
        features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        target="Target",
    )
    tuned_pipeline = SklearnPipelineTuner(
        sklearn_pipeline_factory,
        tuner_init_config,
    ).tune(regression_data)
    assert tuned_pipeline.features_out == tuned_pipeline.features_in
