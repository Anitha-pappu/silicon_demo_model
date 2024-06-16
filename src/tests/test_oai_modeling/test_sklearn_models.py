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

import logging
import typing as tp

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from typing_extensions import Self

from modeling import SklearnModel, SklearnPipeline
from modeling.splitters import BySequentialSplitter
from modeling.types import Matrix, Vector
from optimus_core import SelectColumns, SklearnTransform
from optimus_core.transformer.feature_selection import SkLearnSelector

N_FEATURES = 10
N_SAMPLES = 250
N_INFORMATIVE = 3
N_NOISE = 5


class SklearnIncompatibleEstimator(object):
    """
    A predictor class that looks like sklearn compatible model,
    however lacks `.set_params` method which is essential
    """
    def predict(self, data: Matrix, **kwargs: tp.Any) -> Vector:
        """
        Sklearn compatible definition of .predict method
        """

    def fit(self, data: Matrix, **kwargs: tp.Any) -> Self:
        """
        Sklearn compatible definition of .fit method
        """
        return self


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame(
        data=np.random.randn(N_SAMPLES, N_FEATURES),
        columns=[f"Column_{i}" for i in range(N_FEATURES)],
    )


@pytest.fixture(scope="module")
def sklearn_incompatible_estimator():
    return SklearnIncompatibleEstimator()


@pytest.fixture(scope="function")
def two_steps_sklearn_pipeline():
    return sklearn.pipeline.Pipeline([
        ("scaler", SklearnTransform(StandardScaler())),
        ("estimator", LinearRegression()),
    ])


@pytest.fixture(scope="function")
def two_steps_sklearn_pipeline_without_wrappers():
    return sklearn.pipeline.Pipeline(
        [
            ("scaler", StandardScaler()),
            ("estimator", LinearRegression()),
        ],
    )


@pytest.fixture(scope="function")
def select_features_sklearn_pipeline():
    return sklearn.pipeline.Pipeline(
        [
            (
                "select_required_columns",
                SelectColumns(items=[f"Column_{i + 1}" for i in range(N_FEATURES)]),
            ),
            (
                "select_best_3",
                SkLearnSelector(SelectKBest(k=N_INFORMATIVE, score_func=f_regression)),
            ),
            ("estimator", LinearRegression()),
        ],
    )


@pytest.fixture(scope="function")
def estimator():
    return LinearRegression()


@pytest.fixture(scope="module")
def fitted_linear_sklearn_model(data: pd.DataFrame):
    return SklearnModel(
        estimator=LinearRegression(),
        features_in=[f"Column_{i + 1}" for i in range(N_INFORMATIVE)],
        target="Column_7",
    ).fit(data)


@pytest.fixture(scope="module")
def fitted_sklearn_pipeline(data: pd.DataFrame):
    pipeline = sklearn.pipeline.Pipeline(
        [
            ("scaler", SklearnTransform(StandardScaler())),
            ("estimator", LinearRegression()),
        ],
    )
    return SklearnPipeline(
        estimator=pipeline,
        features_in=[f"Column_{i + 1}" for i in range(N_INFORMATIVE)],
        target="Column_7",
    ).fit(data)


class TestInitialization(object):
    def test_estimator_validation(self, sklearn_incompatible_estimator):
        with pytest.raises(RuntimeError):
            model = SklearnModel(  # noqa:F841
                sklearn_incompatible_estimator,
                target="Target",
                features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
            )

    def test_initialize_with_fitted_pipeline(
        self, two_steps_sklearn_pipeline, regression_data,
    ):
        two_steps_sklearn_pipeline.fit(
            regression_data[[f"Column_{i + 1}" for i in range(N_FEATURES)]],
            regression_data["Target"],
        )
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        assert model.features_in == model.features_out
        assert set(model.get_feature_importance(regression_data).keys()) == set(
            model.features_out,
        )

    def test_initialize_with_fitted_estimator(self, estimator, regression_data):
        estimator.fit(
            regression_data[[f"Column_{i + 1}" for i in range(N_FEATURES)]],
            regression_data["Target"],
        )
        model = SklearnModel(
            estimator,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        assert model.features_in == model.features_out
        assert set(model.get_feature_importance(regression_data).keys()) == set(
            model.features_out,
        )


class TestAttributes(object):
    def test_sklearn_pipeline_attributes(self, two_steps_sklearn_pipeline):
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        assert model.target == "Target"
        assert model.features_in == [f"Column_{i + 1}" for i in range(N_FEATURES)]
        assert isinstance(model.get_pipeline(), sklearn.pipeline.Pipeline)
        assert isinstance(model.estimator, sklearn.base.BaseEstimator)

    def test_sklearn_estimator_attributes(self, estimator):
        model = SklearnModel(
            estimator,
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
            target="Target",
        )
        assert model.target == "Target"
        assert model.features_in == [f"Column_{i + 1}" for i in range(N_FEATURES)]
        assert isinstance(model.estimator, sklearn.base.BaseEstimator)


class TestRaisesNotFittedError(object):
    def test_sklearn_model_raises_not_fitted_error(self, regression_data, estimator):
        model = SklearnModel(
            estimator,
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
            target="Target",
        )
        with pytest.raises(sklearn.exceptions.NotFittedError):
            model.get_feature_importance(regression_data)

        with pytest.raises(sklearn.exceptions.NotFittedError):
            model.predict(regression_data)

    def test_sklearn_pipeline_features_out_raises_not_fitted_error(
        self, regression_data, two_steps_sklearn_pipeline,
    ):
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        with pytest.raises(sklearn.exceptions.NotFittedError):
            model.get_feature_importance(regression_data)

        with pytest.raises(sklearn.exceptions.NotFittedError):
            model.predict(regression_data)

        with pytest.raises(sklearn.exceptions.NotFittedError):
            model.features_out


class TestFitPredict(object):
    def test_sklearn_pipeline_fit_predict(
        self, regression_data, two_steps_sklearn_pipeline,
    ):
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        model.fit(regression_data)
        assert list(model.features_out) == [
            f"Column_{i + 1}" for i in range(N_FEATURES)
        ]
        assert (
            set(model.get_feature_importance(data=regression_data))
            == {f"Column_{i + 1}" for i in range(N_FEATURES)}
        )
        assert np.allclose(
            model.predict(regression_data[model.features_in]),
            model.predict(regression_data),
        )
        assert (
            r2_score(regression_data["Target"], model.predict(regression_data)) > 0.99
        )

    def test_sklearn_model_fit_predict(self, regression_data, estimator):
        model = SklearnModel(
            estimator,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        model.fit(regression_data)
        assert list(model.features_in) == [f"Column_{i + 1}" for i in range(N_FEATURES)]
        assert np.allclose(
            model.predict(regression_data[model.features_in]),
            model.predict(regression_data),
        )
        assert (
            r2_score(regression_data["Target"], model.predict(regression_data)) > 0.99
        )

    def test_sklearn_pipeline_predict_but_not_all_columns_are_present(
        self, regression_data, two_steps_sklearn_pipeline,
    ):
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        model.fit(regression_data)
        with pytest.raises(ValueError):
            model.predict(regression_data[model.features_in[:-1]])

    def test_sklearn_model_predict_but_not_all_columns_are_present(
        self, regression_data, estimator,
    ):
        model = SklearnModel(
            estimator,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        model.fit(regression_data)
        with pytest.raises(ValueError):
            model.predict(regression_data[model.features_in[:-1]])

    def test_sklearn_pipeline_fit_but_not_all_columns_are_present(
        self, regression_data, two_steps_sklearn_pipeline,
    ):
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        with pytest.raises(ValueError):
            model.fit(regression_data[model.features_in[:-1]])

    def test_sklearn_model_fit_but_not_all_columns_are_present(
        self, regression_data, estimator,
    ):
        model = SklearnModel(
            estimator,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        with pytest.raises(ValueError):
            model.fit(regression_data[model.features_in[:-1]])

    def test_sklearn_pipeline_feature_selection(
        self, regression_data, select_features_sklearn_pipeline,
    ):
        model = SklearnPipeline(
            select_features_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        model.fit(regression_data)
        assert len(model.features_out) == N_INFORMATIVE
        assert set(model.features_out).issubset(model.features_in)
        assert (
            r2_score(regression_data["Target"], model.predict(regression_data)) > 0.99
        )
        assert np.allclose(
            model.estimator.predict(regression_data[model.features_out]),
            model.predict(regression_data),
        )

    def test_warnings_when_features_out_not_specified(
        self, regression_data, two_steps_sklearn_pipeline, caplog,
    ):
        two_steps_sklearn_pipeline.fit(
            regression_data[[f"Column_{i + 1}" for i in range(N_FEATURES)]],
            regression_data["Target"],
        )
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        with caplog.at_level(logging.WARNING):
            assert model.features_in == model.features_out
            assert (
                "`features_out` property was not set during initialization"
                in caplog.text
            )

    def test_info_when_features_out_are_getting_set(
        self, regression_data, two_steps_sklearn_pipeline, caplog,
    ):
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        with caplog.at_level(logging.INFO):
            model.fit(regression_data)
            assert "`features_out` attribute is not specified" in caplog.text


class TestRaiseWhenColumnsMismatch(object):
    def test_sklearn_pipeline_raises_error_when_factual_columns_dont_match(
        self, regression_data, two_steps_sklearn_pipeline,
    ):
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
            features_out=[f"Column_{i + 1}" for i in range(N_FEATURES - 1)],
        )
        with pytest.raises(RuntimeError):
            model.fit(regression_data)


class TestPipelineDoesntKeepColumns(object):
    def test_warning_when_pipeline_doesnt_keep_columns(
        self, regression_data, two_steps_sklearn_pipeline_without_wrappers, caplog,
    ):
        model = SklearnPipeline(
            two_steps_sklearn_pipeline_without_wrappers,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        with caplog.at_level(logging.WARNING):
            model.fit(regression_data)
            assert model.features_in != model.features_out
            assert "transformer" in caplog.text


class TestProduceShapValuesForSklearnModel(object):
    def test_data_from_explanation_is_the_same_as_input_data(
        self, fitted_linear_sklearn_model: SklearnModel, data: pd.DataFrame,
    ) -> None:
        explanation = fitted_linear_sklearn_model.produce_shap_explanation(data)
        np.testing.assert_equal(
            explanation.data, data[fitted_linear_sklearn_model.features_in].values,
        )
        assert explanation.data.shape == explanation.values.shape

    @pytest.mark.parametrize("algorithm", ("linear", "auto", "exact", "permutation"))
    def test_shap_values_sum_into_prediction(
        self,
        fitted_linear_sklearn_model: SklearnModel,
        algorithm: str,
        data: pd.DataFrame,
    ) -> None:
        explanation = fitted_linear_sklearn_model.produce_shap_explanation(data)
        np.testing.assert_allclose(
            fitted_linear_sklearn_model.predict(data),
            explanation.base_values + explanation.values.sum(axis=1),
            rtol=1e-6,
            atol=1e-6,
        )


class TestGetShapFeatureImportanceForSklearnModel(object):
    def test_get_shap_feature_importance_produces_dict_same_len_as_input(
        self, fitted_linear_sklearn_model: SklearnModel, data: pd.DataFrame,
    ) -> None:
        shap_importance = fitted_linear_sklearn_model.get_shap_feature_importance(
            data,
        )
        assert (
            len(shap_importance)
            == len(data[fitted_linear_sklearn_model.features_in].columns)
        )

    @pytest.mark.parametrize("algorithm", ("linear", "auto", "exact", "permutation"))
    def test_columns_not_used_as_features_have_low_feature_importance(
        self,
        fitted_linear_sklearn_model: SklearnModel,
        algorithm: str,
        data: pd.DataFrame,
    ) -> None:
        shap_importance = fitted_linear_sklearn_model.get_shap_feature_importance(
            data,
        )
        for feature_name, importance in shap_importance.items():
            if feature_name not in fitted_linear_sklearn_model.features_in:
                np.testing.assert_allclose(importance, 0)


class TestProduceShapValuesForSklearnPipeline(object):
    def test_data_from_explanation_is_the_same_as_input_data(
        self, fitted_sklearn_pipeline: SklearnPipeline, data: pd.DataFrame,
    ) -> None:
        explanation = fitted_sklearn_pipeline.produce_shap_explanation(data)
        np.testing.assert_equal(
            explanation.data,
            data[fitted_sklearn_pipeline.features_in].values,
        )
        assert explanation.data.shape == explanation.values.shape

    @pytest.mark.parametrize("algorithm", ("auto", "exact", "permutation"))
    def test_shap_values_sum_into_prediction(
        self,
        fitted_sklearn_pipeline: SklearnPipeline,
        algorithm: str,
        data: pd.DataFrame,
    ) -> None:
        explanation = fitted_sklearn_pipeline.produce_shap_explanation(data)
        np.testing.assert_allclose(
            fitted_sklearn_pipeline.predict(data),
            explanation.base_values + explanation.values.sum(axis=1),
            rtol=1e-6,
            atol=1e-6,
        )


class TestGetShapFeatureImportanceForSklearnPipeline(object):
    def test_get_shap_feature_importance_produces_dict_same_len_as_input(
        self, fitted_sklearn_pipeline: SklearnPipeline, data: pd.DataFrame,
    ) -> None:
        shap_feature_importance = fitted_sklearn_pipeline.get_shap_feature_importance(
            data,
        )
        assert (
            len(shap_feature_importance)
            == len(data[fitted_sklearn_pipeline.features_in].columns)
        )

    @pytest.mark.parametrize("algorithm", ("auto", "exact", "permutation"))
    def test_columns_not_used_as_features_have_low_feature_importance(
        self,
        fitted_sklearn_pipeline: SklearnPipeline,
        algorithm: str,
        data: pd.DataFrame,
    ) -> None:
        shap_feature_importance = fitted_sklearn_pipeline.get_shap_feature_importance(
            data,
        )
        for feature_name, importance in shap_feature_importance.items():
            if feature_name not in fitted_sklearn_pipeline.features_in:
                np.testing.assert_allclose(importance, 0)


class TestEvaluatesMetricsForSklearnModel(object):
    def test_evaluation_metrics_has_correct_types(
        self,
        estimator: LinearRegression,
        regression_data: pd.DataFrame,
    ) -> None:
        model = SklearnModel(
            estimator,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        model.fit(regression_data)
        evaluation_metrics = model.evaluate_metrics(regression_data)
        for metric_name, metric_value in evaluation_metrics.items():
            assert isinstance(metric_name, str)
            assert isinstance(metric_value, float)

    def test_evaluation_metrics_are_close_to_ideal(
        self,
        estimator: LinearRegression,
        regression_data: pd.DataFrame,
    ) -> None:
        model = SklearnModel(
            estimator,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        model.fit(regression_data)
        evaluation_metrics = model.evaluate_metrics(regression_data)
        assert evaluation_metrics["mape"] < 0.1
        assert evaluation_metrics["r_squared"] > 0.99
        assert evaluation_metrics["var_score"] > 0.99


class TestEvaluatesMetricsForSklearnPipeline(object):
    def test_evaluation_metrics_has_correct_types(
        self,
        two_steps_sklearn_pipeline: sklearn.pipeline.Pipeline,
        regression_data: pd.DataFrame,
    ) -> None:
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        model.fit(regression_data)
        evaluation_metrics = model.evaluate_metrics(regression_data)
        for metric_name, metric_value in evaluation_metrics.items():
            assert isinstance(metric_name, str)
            assert isinstance(metric_value, float)

    def test_evaluation_metrics_are_close_to_ideal(
        self,
        two_steps_sklearn_pipeline: sklearn.pipeline.Pipeline,
        regression_data: pd.DataFrame,
    ) -> None:
        model = SklearnPipeline(
            two_steps_sklearn_pipeline,
            target="Target",
            features_in=[f"Column_{i + 1}" for i in range(N_FEATURES)],
        )
        model.fit(regression_data)
        evaluation_metrics = model.evaluate_metrics(regression_data)
        assert evaluation_metrics["mape"] < 0.1
        assert evaluation_metrics["r_squared"] > 0.99
        assert evaluation_metrics["var_score"] > 0.99


class TestRobustAndNonRobustShapExplanationForSklearnModel(object):
    """ This test class tests that the robust shap explanation works, and that the
     non-robust shap explanation also works for cases where it is supposed to work
     (for linear models).
    """

    @pytest.fixture(scope="class")
    def main_data(self, n: int = 100):
        df = pd.DataFrame({
            "sensor1": np.linspace(0, 10, n),
            "sensor2": np.random.rand(n),
        })
        assign_dict = {
            "output": np.square(df["sensor1"]) * df["sensor2"],
            "timestamp": pd.date_range(start="2020-01-01", periods=n, freq="1T"),
        }
        df = df.assign(**assign_dict)
        return df

    @pytest.fixture(scope="class")
    def train_test_data(self, main_data):
        splitter = BySequentialSplitter(
            datetime_column="timestamp",
            block_freq="20T",
            train_freq="15T",
        )
        train_df, test_df = splitter.split(main_data)
        return train_df, test_df

    @pytest.mark.parametrize("estimator", [
        LinearRegression(),
        RandomForestRegressor(),
        SVR(),
    ])
    def test_robust_shap_explanation_works(self, estimator, train_test_data):
        train_data, test_data = train_test_data
        model = SklearnModel(
            estimator=estimator,
            features_in=["sensor1", "sensor2"],
            target="output",
        )
        model.fit(train_data)
        data_with_model_input_features = train_data[model.features_in]
        model.produce_shap_explanation(data_with_model_input_features, robust=True)

    @pytest.mark.parametrize("estimator", [
        LinearRegression(),
    ])
    def test_non_robust_shap_explanation_works_for_linear_model(
        self,
        estimator,
        train_test_data,
    ):
        train_data, test_data = train_test_data
        model = SklearnModel(
            estimator=estimator,
            features_in=["sensor1", "sensor2"],
            target="output",
        )
        model.fit(train_data)
        data_with_model_input_features = train_data[model.features_in]
        model.produce_shap_explanation(data_with_model_input_features, robust=False)


class TestRobustAndNonRobustShapExplanationForSklearnPipeline(object):  # noqa: WPS118
    """ This test class tests that the shap explanation works for ``SklearnPipeline``,
     and that it works even without the need for a specific ``robust`` option (as is
     instead the case for ``SklearnModel``).
    """

    @pytest.fixture(scope="class")
    def main_data(self, num: int = 100):
        df = pd.DataFrame({
            "sensor1": np.linspace(0, 10, num),
            "sensor2": np.random.rand(num),
        })
        assign_dict = {
            "output": np.square(df["sensor1"]) * df["sensor2"],
            "timestamp": pd.date_range(start="2020-01-01", periods=num, freq="1T"),
        }
        df = df.assign(**assign_dict)
        return df

    @pytest.fixture(scope="class")
    def train_test_data(self, main_data):
        splitter = BySequentialSplitter(
            datetime_column="timestamp",
            block_freq="20T",
            train_freq="15T",
        )
        train_df, test_df = splitter.split(main_data)
        return train_df, test_df

    @pytest.mark.parametrize("sklearn_model", [
        LinearRegression(),
        RandomForestRegressor(),
        SVR(),
    ])
    def test_robust_shap_explanation_works(self, sklearn_model, train_test_data):
        """
        The implementation of ``produce_shap_explanation`` in SklearnPipeline is by
         default a robust one, since using arbitrary transformations in the pipeline
         requires a robust and model-agnostic way of calculating shap values by default.
        So a ``robust`` option for ``SklearnPipeline`` is not needed (while it is
         needed for ``SklearnModel``).
        Confirming that this is true by testing with these sklearn models, which might
         fail in ``SklearnModel`` without the robust option, but should work in
         ``SklearnPipeline``:
        - LinearRegression(): this works without troubles
        - RandomForestRegressor(): this can raise additivity check erros is not handled
         robustly
        - SVR(): this requres the ``.predict`` method and "permutation". In the case of
         ``SklearnPipeline``, this should work with "auto" too, since for an
         ``sklearn.Pipeline`` shap chooses "permutation" when the algorithm is "auto"
          """
        train_data, test_data = train_test_data
        sklearn_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("sklearn_model", sklearn_model),
        ])
        model = SklearnPipeline(
            estimator=sklearn_pipe,
            features_in=["sensor1", "sensor2"],
            target="output",
        )
        model.fit(train_data)
        data_with_model_input_features = train_data[model.features_in]
        model.produce_shap_explanation(data_with_model_input_features)
