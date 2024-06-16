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
"""
Tests for model training and tuning functions.
"""

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from modeling import (
    SklearnModel,
    SklearnModelFactory,
    SklearnModelTuner,
    SklearnPipeline,
    create_tuner,
    train_model,
    tune_model,
)
from modeling.models.sklearn_pipeline.transformer import add_transformers
from modeling.utils import drop_nan_rows
from optimus_core import SelectColumns


class TestAddTransformers(object):
    def test_transformers(
        self,
            estimator,
            simple_data_tag_dict,
            silica_model_features,
            mixed_transformers,
    ):
        actual = add_transformers(estimator, transformers=mixed_transformers)
        assert len(actual) == 3
        assert actual.steps[0][0] == "min_max"
        assert actual.steps[1][0] == "select_best_15"
        assert actual.steps[2][0] == "estimator"

    def test_custom_estimator_name_transformers(
        self, estimator, silica_model_features, mixed_transformers,
    ):
        actual = add_transformers(
            estimator,
            transformers=mixed_transformers,
            estimator_step_name="my_custom_estimator_step_name",
        )
        assert len(actual) == 3
        assert actual.steps[0][0] == "min_max"
        assert actual.steps[1][0] == "select_best_15"
        assert actual.steps[2][0] == "my_custom_estimator_step_name"


class VerboseLR(LinearRegression):
    def fit(self, *args, verbose=True, **kwargs):
        if verbose:
            raise ValueError("Verbose must be set to false.")
        return super().fit(*args, **kwargs)


@pytest.fixture
def tuner_config():
    return {
        "class_name": "sklearn.model_selection.GridSearchCV",
        "kwargs": {
            "param_grid": {"fit_intercept": [True, False]},
            "n_jobs": 1,
            "scoring": {
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            "refit": "mae",
        },
    }


@pytest.fixture
def tuner_pipeline():
    return GridSearchCV(None, {"estimator__fit_intercept": [True, False]}, n_jobs=1)


class TestTuneModel(object):
    def test_cv_results_sklearn(self, simple_data, silica_model_features, tuner_config):
        init_config = {
            "estimator": {
                "class_name": "sklearn.linear_model.LinearRegression",
                "kwargs": None,
            },
        }
        sklearn_model_builder = SklearnModelFactory(
            model_init_config=init_config,
            features_in=silica_model_features,
            target="% Silica Concentrate",
        )
        tuner = create_tuner(
            model_factory=sklearn_model_builder,
            model_tuner_type=SklearnModelTuner,
            tuner_config=tuner_config,
        )
        model = tune_model(
            model_tuner=tuner,
            data=simple_data,
            hyperparameters_config=None,
        )
        assert isinstance(model.estimator, LinearRegression)


class TestTrainModel(object):
    def test_sklearn_pipeline(
        self, simple_data, silica_model_features,
    ):
        pipeline = Pipeline(
            [
                ("select_columns", SelectColumns(items=silica_model_features)),
                ("estimator", LinearRegression()),
            ],
        )
        sklearn_pipeline = SklearnPipeline(
            pipeline,
            features_in=silica_model_features,
            target="% Silica Concentrate",
        )
        trained_sklearn_pipeline = train_model(sklearn_pipeline, simple_data)
        assert trained_sklearn_pipeline is sklearn_pipeline
        assert isinstance(sklearn_pipeline.estimator, LinearRegression)
        assert sklearn_pipeline.features_out == silica_model_features

    def test_sklearn_model(self, simple_data, silica_model_features):
        estimator = LinearRegression()
        sklearn_model = SklearnModel(
            estimator,
            features_in=silica_model_features,
            target="% Silica Concentrate",
        )
        trained_sklearn_model = train_model(sklearn_model, simple_data)
        assert trained_sklearn_model is sklearn_model
        assert isinstance(sklearn_model.estimator, LinearRegression)

    def test_sklearn_train_kwargs(self, simple_data, silica_model_features):
        estimator = VerboseLR()
        sklearn_model = SklearnModel(
            estimator,
            features_in=silica_model_features,
            target="% Silica Concentrate",
        )
        with pytest.raises(ValueError):
            trained_sklearn_model = train_model(sklearn_model, simple_data)
        trained_sklearn_model = train_model(sklearn_model, simple_data, verbose=False)
        assert trained_sklearn_model is sklearn_model
        assert isinstance(sklearn_model.estimator, VerboseLR)


class TestDropNanRows(object):
    def test_drops_nans(self, simple_data, simple_data_tag_dict):
        nan_data = simple_data.copy()

        # Not a feature, won't be dropped.
        date = nan_data["date"]
        date[0] = pd.NA
        nan_data["date"] = date

        # Feature column, will be dropped.
        iron_feed = nan_data["% Iron Feed"]
        iron_feed[1] = pd.NA
        nan_data["% Iron Feed"] = iron_feed

        # Target column, will be dropped.
        silica_conc = nan_data["% Silica Concentrate"]
        silica_conc[2] = pd.NA
        nan_data["% Silica Concentrate"] = silica_conc

        actual = drop_nan_rows(
            nan_data,
            td=simple_data_tag_dict,
            td_features_column="silica_model_feature",
            target_column="% Silica Concentrate",
        )

        assert len(actual) == len(simple_data) - 2
        assert actual.isna().sum().sum() == 1
