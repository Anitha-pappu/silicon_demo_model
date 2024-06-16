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


import numpy as np
import pandas as pd
import pytest
import tensorflow
from _pytest.fixtures import FixtureRequest  # noqa: WPS436

from modeling.models.keras_model import KerasModel

_N_COLS_DATASET = 10
_N_FEATURES_IN_MODEL = 5
_N_ROWS_DATASET = 25


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame(
        data=np.random.randn(_N_ROWS_DATASET, _N_COLS_DATASET),
        columns=[f"Feature_{i}" for i in range(_N_COLS_DATASET)],
    )


@pytest.fixture(scope="module", params=[16, 32])
def keras_model(request: FixtureRequest, data: pd.DataFrame) -> KerasModel:
    keras_model = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.Normalization(axis=-1),
            tensorflow.keras.layers.Dense(units=request.param, activation="tanh"),
            tensorflow.keras.layers.Dense(units=1),
        ],
    )
    keras_model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=[
            tensorflow.keras.metrics.MeanAbsoluteError(),
            tensorflow.keras.metrics.MeanSquaredError(),
            tensorflow.keras.metrics.MeanAbsolutePercentageError(),
        ],
    )
    return KerasModel(
        keras_model=keras_model,
        features_in=[f"Feature_{i}" for i in range(_N_FEATURES_IN_MODEL)],
        target="Feature_0",
    ).fit(data)


class TestProduceShapValuesForKerasModel(object):
    @pytest.mark.parametrize("algorithm", ("deep", "auto", "exact"))
    def test_data_from_explanation_is_the_same_as_input_data(
        self, keras_model: KerasModel, data: pd.DataFrame, algorithm: str,
    ) -> None:
        explanation = keras_model.produce_shap_explanation(data, algorithm=algorithm)
        np.testing.assert_equal(
            explanation.data, data[keras_model.features_in].values,
        )
        assert explanation.data.shape == explanation.values.shape

    @pytest.mark.parametrize("algorithm", ("deep", "auto", "exact"))
    def test_shap_values_sum_into_prediction(
        self, keras_model: KerasModel, algorithm: str, data: pd.DataFrame,
    ) -> None:
        explanation = keras_model.produce_shap_explanation(data)
        np.testing.assert_allclose(
            keras_model.predict(data),
            explanation.base_values + explanation.values.sum(axis=1),
            rtol=1e-6,
            atol=1e-6,
        )


class TestGetShapFeatureImportanceForKerasModel(object):
    def test_get_shap_feature_importance_produces_dict_same_len_as_input(
        self, keras_model: KerasModel, data: pd.DataFrame,
    ) -> None:
        shap_feature_importance = keras_model.get_shap_feature_importance(data)
        assert (
            len(shap_feature_importance)
            == len(data[keras_model.features_in].columns)
        )

    def test_columns_not_used_as_features_have_low_feature_importance(
        self, keras_model: KerasModel, data: pd.DataFrame,
    ) -> None:
        shap_feature_importance = keras_model.get_shap_feature_importance(data)
        for feature_name, importance in shap_feature_importance.items():
            if feature_name not in keras_model.features_in:
                np.testing.assert_allclose(importance, 0)


class TestEvaluatesMetricsForKerasModel(object):
    def test_evaluation_metrics_has_correct_types(
        self,
        keras_model: KerasModel,
        data: pd.DataFrame,
    ) -> None:
        evaluation_metrics = keras_model.evaluate_metrics(data)
        for metric_name, metric_value in evaluation_metrics.items():
            assert isinstance(metric_name, str)
            assert isinstance(metric_value, float)
