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
Tests for functions that evaluate the result of modeling.
"""

import logging
import warnings

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from modeling import (
    SklearnModel,
    SklearnPipeline,
    calculate_feature_importance,
    calculate_metrics,
)
from modeling.models.functional import verify_selected_controls
from optimus_core.transformer import SelectColumns


class TestVerifySelectedControls(object):
    def test_no_warning_no_controls(
        self, simple_data, simple_data_tag_dict, silica_model_features,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sklearn_model = SklearnModel(
                LinearRegression(),
                features_in=silica_model_features,
                target="% Silica Concentrate",
            )
            sklearn_model.fit(simple_data)
            verify_selected_controls(
                sklearn_model, td=simple_data_tag_dict,
            )

    def test_no_warning_selects_controls(self, simple_data, simple_data_tag_dict):
        pipeline = Pipeline(
            [
                (
                    "select_columns",
                    SelectColumns(["% Silica Feed", "% Iron Concentrate"]),
                ),
                ("estimator", LinearRegression()),
            ],
        )
        sklearn_pipeline = SklearnPipeline(
            pipeline,
            features_in=["% Silica Feed", "% Iron Concentrate"],
            target="% Silica Concentrate",
        )
        sklearn_pipeline.fit(simple_data)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            verify_selected_controls(sklearn_pipeline, simple_data_tag_dict)

    def test_warning_selects_no_controls(self, simple_data, simple_data_tag_dict):
        pipeline = Pipeline(
            [
                ("estimator", LinearRegression()),
            ],
        )
        sklearn_pipeline = SklearnPipeline(
            pipeline,
            features_in=["% Iron Feed", "% Silica Feed"],
            target="% Silica Concentrate",
        )
        sklearn_pipeline.fit(simple_data)

        with pytest.warns(RuntimeWarning):
            verify_selected_controls(sklearn_pipeline, simple_data_tag_dict)


class TestCalculateMetrics(object):
    def test_functional_api_does_not_contradict_with_model_method(
        self,
        simple_data: pd.DataFrame,
        trained_sklearn_model: SklearnModel,
    ) -> None:
        model = trained_sklearn_model
        metrics_via_function = calculate_metrics(simple_data, model)
        metrics_via_method = model.evaluate_metrics(simple_data)
        assert metrics_via_function == pytest.approx(metrics_via_method)


class ImportanceLR(LinearRegression):
    @property
    def feature_importances_(self):  # noqa: WPS120
        return self.coef_


class TestFeatureImportance(object):
    def test_uses_feature_importances(self, simple_data):
        features = ["% Iron Feed", "% Silica Feed"]
        target_column = "% Silica Concentrate"

        data, target = simple_data[features], simple_data[target_column]

        model = ImportanceLR().fit(data, target)
        sklearn_model = SklearnModel(model, features, target)

        actual = calculate_feature_importance(simple_data, sklearn_model)

        assert all(actual["feature_importance"] == sorted(model.coef_)[::-1])

    def test_uses_pipeline(self, simple_data):
        pipeline = Pipeline(
            [
                ("select_columns", SelectColumns(["% Iron Feed", "% Silica Feed"])),
                ("estimator", ImportanceLR()),
            ],
        )

        target = "% Silica Concentrate"

        sklearn_pipeline = SklearnPipeline(
            pipeline,
            features_in=["% Iron Feed", "% Silica Feed"],
            target=target,
        )
        sklearn_pipeline.fit(simple_data)

        actual = calculate_feature_importance(simple_data, sklearn_pipeline)

        assert all(
            actual["feature_importance"]
            == sorted(pipeline.named_steps["estimator"].coef_)[::-1],
        )

    def test_uses_permutation_importance(self, simple_data, caplog):
        pipeline = Pipeline(
            [
                ("select_columns", SelectColumns(["% Iron Feed", "% Silica Feed"])),
                ("estimator", LinearRegression()),
            ],
        )

        target = "% Silica Concentrate"
        sklearn_pipeline = SklearnPipeline(
            pipeline,
            features_in=["% Iron Feed", "% Silica Feed"],
            target=target,
        )
        sklearn_pipeline.fit(simple_data)

        with caplog.at_level(logging.INFO):
            actual = calculate_feature_importance(simple_data, sklearn_pipeline)

            assert "permutation_importance" in caplog.text
            assert "feature_importance" in actual.columns
