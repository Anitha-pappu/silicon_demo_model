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
Tests for config loading functions.
"""
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import BaseEstimator
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LinearRegression
from sklearn.metrics._scorer import _BaseScorer  # noqa: WPS436,WPS450
from sklearn.model_selection import LeavePOut, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from modeling.models.sklearn_model.factory import load_estimator
from modeling.models.sklearn_model.target_transformer import (
    InvalidTransformerFuncError,
    TargetTransformerInitConfig,
    add_target_transformer,
)
from modeling.models.sklearn_model.tuner import (
    initialize_sklearn_hyperparameters_tuner,
)
from modeling.models.sklearn_pipeline.transformer import load_transformer
from optimus_core.transformer import SklearnTransform
from optimus_core.transformer.feature_selection import SkLearnSelector


class DummyClass(object):
    """
    Dummy class that has no `.fit` method
    """


class DummyFitter(object):
    def fit(self):
        """
        Definition of `.fit` method for dummy class
        """


class TestLoadEstimator(object):
    def test_raises_bad_class(self):
        with pytest.raises(AttributeError, match="fit"):
            load_estimator("tests.test_load.DummyClass")

        with pytest.raises(AttributeError, match="predict"):
            load_estimator("tests.test_load.DummyFitter")

    def test_returns_class(self):
        estimator = load_estimator(
            "sklearn.linear_model.LinearRegression",
            model_kwargs={"fit_intercept": False},
        )

        assert type(estimator).__name__ == "LinearRegression"
        assert not estimator.fit_intercept


@pytest.fixture
def regression_problem():
    """Create regression data"""
    X, y = make_regression(
        n_samples=100, n_features=4, n_informative=3, random_state=0, noise=2.0,
    )

    return train_test_split(
        pd.DataFrame(X, columns=["tag_a", "tag_b", "tag_c", "tag_e"]),
        pd.Series(y, name="tag_d"),
        test_size=0.3,
        random_state=0,
    )


@pytest.fixture
def params_target_transformer_function():
    """Target transformer parameters"""
    return yaml.safe_load(
        dedent(
            """
            func: numpy.log1p
            inverse_func: numpy.expm1
            """,
        ),
    )


@pytest.fixture
def params_target_transformer_scaler():
    """Target transformer parameters"""
    return {
        'transformer': {
            'class_name': 'sklearn.preprocessing.MinMaxScaler',
            'kwargs': {'feature_range': (0, 2)},
        },
    }


@pytest.fixture
def estimator() -> BaseEstimator:
    return LinearRegression()


class TestAddTargetTransformer(object):
    def test_transformer(
        self, estimator, regression_problem, params_target_transformer_scaler,
    ):
        """Test that `add_target_transformer` makes an estimator return same predictions
        as if this estimator was trained on transformed target
        and then predictions were inverse transformed manually.

        This case tests target transformation via `transformer` - meaning
        a class with `fit_transform` and `inverse_transform` methods.
        """

        X_train, X_test, y_train, y_test = regression_problem
        expected_transformer = MinMaxScaler(feature_range=(0, 2))
        expected_estimator = LinearRegression()
        y_train = y_train.to_numpy().reshape(-1, 1)

        # Fitting estimator to transformed target.
        expected_estimator.fit(X_train, expected_transformer.fit_transform(y_train))
        # Predicts on X_test and inverse_transforms to normal scale.
        expected_prediction = expected_transformer.inverse_transform(
            expected_estimator.predict(X_test),
        )
        target_transformer_init_config = TargetTransformerInitConfig(
            **params_target_transformer_scaler,
        )
        estimator = add_target_transformer(
            estimator, target_transformer_init_config.transformer,
        )
        estimator.fit(X_train, y_train)
        prediction = estimator.predict(X_test)

        np.testing.assert_allclose(prediction, expected_prediction, rtol=1e-10, atol=0)

    def test_custom_functions(
        self, estimator, regression_problem, params_target_transformer_function,
    ):
        """Similarly to case above, tests that `add_target_transformer` makes
        an estimator return same predictions as if this estimator was trained
        on transformed target and then predictions were inverse transformed manually.

        However this one assumes that transformer is initialized via
        `func` and `inverse_func` as opposed to a class from transformer init config.
        """

        X_train, X_test, y_train, _ = regression_problem
        y_train = np.exp(y_train)

        expected_estimator = LinearRegression()
        expected_estimator.fit(X_train, np.log1p(y_train))
        expected_prediction = np.expm1(expected_estimator.predict(X_test))
        target_transformer_init_config = TargetTransformerInitConfig(
            **params_target_transformer_function,
        )
        estimator = add_target_transformer(
            estimator,
            func=target_transformer_init_config.func,
            inverse_func=target_transformer_init_config.inverse_func,
        )
        estimator.fit(X_train, y_train)
        prediction = estimator.predict(X_test)

        np.testing.assert_allclose(prediction, expected_prediction, rtol=1e-10, atol=0)

    @staticmethod
    @pytest.mark.parametrize("config,should_raise", [
        # Those should be initialized successfully.
        (
            TargetTransformerInitConfig(
                func="numpy.log1p",
                inverse_func="numpy.expm1",
            ),
            False,
        ),
        (
            TargetTransformerInitConfig(
                func="numpy.exp",
                inverse_func="numpy.log",
            ),
            False,
        ),
        (
            TargetTransformerInitConfig(
                func="numpy.square",
                inverse_func="numpy.sqrt",
            ),
            False,
        ),
        # Those should fail.
        (
            TargetTransformerInitConfig(
                func="numpy.sum",
                inverse_func="numpy.product",
            ),
            True,
        ),
        (
            TargetTransformerInitConfig(
                func="numpy.sum",
                inverse_func="numpy.square",
            ),
            True,
        ),
        (
            TargetTransformerInitConfig(
                func="numpy.log",
                inverse_func="numpy.product",
            ),
            True,
        ),
    ])
    def test_pair_of_func_and_inverse_func_raises_error_when_expected(
        estimator: BaseEstimator,
        config: TargetTransformerInitConfig,
        should_raise: bool,
    ) -> None:
        """Test if a given combination of func and inverse_func
        raises appropriate error when needed."""
        if should_raise:
            with pytest.raises(
                InvalidTransformerFuncError,
                match="func or inverse func supplied aren't valid",
            ):
                add_target_transformer(
                    estimator,
                    func=config.func,
                    inverse_func=config.inverse_func,
                )
        else:
            add_target_transformer(
                estimator,
                func=config.func,
                inverse_func=config.inverse_func,
            )


class TestLoadTransformer(object):
    @pytest.mark.parametrize("pop", ["class_name", "kwargs", "name", "wrapper"])
    def test_raises_bad_config(self, pop):
        valid_config = {
            "class_name": "sklearn.preprocessing.StandardScaler",
            "kwargs": {},
            "name": "standard_scaler",
            "wrapper": None,
        }

        valid_config.pop(pop)

        with pytest.raises(KeyError, match=f"{pop}"):
            load_transformer(valid_config)

    def test_returns_object(self):
        expected = ("transform", MinMaxScaler())

        with pytest.warns(RuntimeWarning):
            actual = load_transformer(expected)

        assert expected == actual

    @pytest.mark.parametrize(
        "config,expected_class,expected_object",
        [
            (
                {
                    "class_name": "sklearn.preprocessing.StandardScaler",
                    "kwargs": {"with_mean": False},
                    "name": "standard_scaler",
                    "wrapper": "preserve_columns",
                },
                SklearnTransform,
                StandardScaler(with_mean=False),
            ),
            (
                {
                    "class_name": "sklearn.feature_selection.SelectKBest",
                    "kwargs": {"k": 15, "score_func": "sklearn.feature_selection.chi2"},
                    "name": "select_best_15",
                    "wrapper": "select_columns",
                },
                SkLearnSelector,
                SelectKBest(k=15, score_func=chi2),
            ),
        ],
    )
    def test_config_dicts(self, config, expected_class, expected_object):
        actual_name, actual_object = load_transformer(config)

        assert isinstance(actual_object, expected_class)

        attribute = (
            "selector"
            if config["wrapper"] == "select_columns"
            else "transformer"
        )

        assert (
            type(getattr(actual_object, attribute))  # noqa: WPS516
            == type(expected_object)  # noqa: WPS516
        )
        assert (
            getattr(actual_object, attribute).get_params()
            == expected_object.get_params()
        )
        assert actual_name == config["name"]


@pytest.fixture
def tuner_config():
    return {
        "class_name": "sklearn.model_selection.RandomizedSearchCV",
        "kwargs": {
            "cv": 7,
            "param_distributions": {
                "hyper_param_1": [1, 2, 3, 4],
                "hyper_param_2": [10.0 ** i for i in range(-5, 5)],
            },
            "scoring": {"mae": "neg_mean_absolute_error"},
        },
    }


class TestDistribution(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestLoadTuner(object):
    def test_correct(self, tuner_config, estimator):
        actual = initialize_sklearn_hyperparameters_tuner(tuner_config, estimator)

        assert actual.estimator == estimator
        assert actual.cv == 7

    def test_leave_p_out_cv(self, tuner_config, estimator):
        p = 3

        tuner_config["kwargs"]["cv"] = {
            "class": "sklearn.model_selection.LeavePOut",
            "kwargs": {"p": p},
        }

        actual = initialize_sklearn_hyperparameters_tuner(tuner_config, estimator)

        assert actual.estimator == estimator
        assert isinstance(actual.cv, LeavePOut)
        assert actual.cv.p == p

    def test_param_distributions_loading(self, tuner_config, estimator):
        tuner_config["kwargs"]["param_distributions"]["hyper_param_1"] = str(
            tuner_config["kwargs"]["param_distributions"]["hyper_param_1"],
        )
        tuner_config["kwargs"]["param_distributions"]["hyper_param_3"] = {
            "class": "tests.test_load.TestDistribution",
            "kwargs": {"loc": 2.0, "scale": 3.0},
        }

        actual = initialize_sklearn_hyperparameters_tuner(tuner_config, estimator)

        assert actual.estimator == estimator
        assert actual.param_distributions["hyper_param_1"] == [1, 2, 3, 4]
        assert isinstance(actual.param_distributions["hyper_param_3"], TestDistribution)

    def test_scoring_default_mape(self, tuner_config, estimator):
        del tuner_config["kwargs"]["scoring"]  # noqa: WPS420
        actual = initialize_sklearn_hyperparameters_tuner(tuner_config, estimator)
        assert actual.estimator == estimator
        assert "mape" in actual.scoring
        assert isinstance(actual.scoring["mape"], _BaseScorer)
