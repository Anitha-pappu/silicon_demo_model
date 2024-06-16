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
Tests for cross-validation functionality.
"""
import typing as tp

import numpy as np
import pandas as pd
import pytest
import tensorflow
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    RepeatedKFold,
    ShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from modeling import SklearnModel, SklearnPipeline
from modeling.models._cross_validation import (
    CVFoldInfo,
    CVResultsContainer,
    TCrossValidatableModel,
    TCVStrategyConfig,
    _build_cross_validator,
    _get_unique_keys_from_sequence_of_dicts,
    _parse_cv_strategy_from_config,
    _produce_cv_results_container,
    cross_validate,
)
from modeling.models.keras_model import KerasModel

N_FEATURES = 10
N_SAMPLES = 1000

FEATURE_NAMES = tuple(f"Feature_{i + 1}" for i in range(N_FEATURES))
TARGET_NAME = "Target"


@pytest.fixture(scope="module")
def modeling_dataset() -> pd.DataFrame:
    """
    Fixture dataset to run cross-validation tests on.
    Does not reuse ``regression_data`` fixture available in ``conftest.py`` due to:
        - Here we need a bigger dataset since we're doing a multifold validation.
        - We define it here in a way that tested models yield realistic accuracy values.

    Returns:
        A dataset to be used as a fixture across this module in ``data`` arguments.
    """

    regression_data, regression_target = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_FEATURES // 2,
        effective_rank=2,
        random_state=42,
        noise=1,
    )
    regression_df = pd.DataFrame(regression_data, columns=FEATURE_NAMES)
    target_df = pd.DataFrame(regression_target, columns=[TARGET_NAME])
    return pd.concat(objs=[regression_df, target_df], axis=1)


def _get_keras_model() -> KerasModel:
    """
    Get OAI KerasModel that can be used in parametrized tests.
    Implemented as a function instead of a ``pytest.fixture`` so that it can be used in
    ``pytest.mark.parametrized`` in a native way.
    """
    keras_model = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.Normalization(axis=-1),
            tensorflow.keras.layers.Dense(units=4, activation="tanh"),
            tensorflow.keras.layers.Dense(units=1),
        ],
    )
    keras_model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=[
            tensorflow.keras.metrics.MeanSquaredError(),
            tensorflow.keras.metrics.MeanAbsoluteError(),
        ],
    )
    return KerasModel(
        keras_model=keras_model,
        features_in=FEATURE_NAMES,
        target=TARGET_NAME,
    )


VALID_CV_STRATEGY_CONFIGS = (
    5,
    None,
    {
        "class_name": "sklearn.model_selection.KFold",
        "kwargs": {
            "shuffle": True,
            "random_state": 42,
        },
    },
    {
        "class_name": "sklearn.model_selection.ShuffleSplit",
        "kwargs": {
            "n_splits": 10,
        },
    },
    {
        "class_name": "sklearn.model_selection.TimeSeriesSplit",
        "kwargs": {
            "n_splits": 3,
        },
    },
    {
        "class_name": "sklearn.model_selection.RepeatedKFold",
        "kwargs": {
            "n_splits": 4,
            "n_repeats": 2,
            "random_state": 42,
        },
    },
)

INVALID_CV_STRATEGY_CONFIGS = (  # noqa: WPS317
    "5",
    (1, 2),
    {
        "class_name": "should_be_object_import_path!",
        "kwargs": {},
    },
)

MODELS_TO_TEST = (
    SklearnModel(
        estimator=LinearRegression(),
        features_in=FEATURE_NAMES,
        target=TARGET_NAME,
    ),
    SklearnModel(
        estimator=Lasso(alpha=0.5, fit_intercept=False, random_state=42),
        features_in=FEATURE_NAMES,
        target=TARGET_NAME,
    ),
    SklearnModel(
        estimator=RandomForestRegressor(),
        features_in=FEATURE_NAMES,
        target=TARGET_NAME,
    ),
    SklearnPipeline(
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("estimator", DecisionTreeRegressor()),
            ],
        ),
        features_in=FEATURE_NAMES,
        target=TARGET_NAME,
    ),
    _get_keras_model(),
)


class TestCVResultsContainer(object):
    """Tests for ``CVResultsContainer`` machinery."""

    default_testing_fold_infos = (
        CVFoldInfo(
            index=0,
            train_data=pd.DataFrame(),
            test_data=pd.DataFrame(),
            train_metrics={
                "MAE": 42,
                "R2": 0.9,
            },
            test_metrics={
                "MAE": 28,
                "R2": 0.6,
            },
        ),
        CVFoldInfo(
            index=1,
            train_data=pd.DataFrame(),
            test_data=pd.DataFrame(),
            train_metrics={
                "MAE": 40,
                "R2": 0.8,
            },
            test_metrics={
                "MAE": 26,
                "R2": 0.4,
            },
        ),
    )
    fold_infos_with_duplicating_indices = (
        CVFoldInfo(
            index=42,
            train_data=pd.DataFrame(),
            test_data=pd.DataFrame(),
            train_metrics={},
            test_metrics={},
        ),
        CVFoldInfo(
            index=42,
            train_data=pd.DataFrame(),
            test_data=pd.DataFrame(),
            train_metrics={},
            test_metrics={},
        ),
    )
    default_testing_container = CVResultsContainer(default_testing_fold_infos)
    container_with_inconsistent_names = CVResultsContainer(
        fold_infos=(
            CVFoldInfo(
                index=1,
                train_data=pd.DataFrame(),
                test_data=pd.DataFrame(),
                train_metrics={
                    "MAE": 42,
                    "R2": 0.9,
                    "Train only metric": 99,
                },
                test_metrics={
                    "MAE": 28,
                    "R2": 0.6,
                },
            ),
        ),
    )

    @staticmethod
    @pytest.mark.parametrize("fold_infos,should_raise", [
        (default_testing_fold_infos, False),
        (fold_infos_with_duplicating_indices, True),
    ])
    def test_raises_on_duplicating_indices(
        fold_infos: tp.Sequence[CVFoldInfo],
        should_raise: bool,
    ) -> None:
        """
        Test that attempting to initialize results container with duplicating
        fold indices raises an exception.
        """
        if should_raise:
            with pytest.raises(ValueError, match="duplicating fold indices"):
                CVResultsContainer(fold_infos)
        else:
            CVResultsContainer(fold_infos)

    def test_iter(self) -> None:
        """Test that iterable over results container yields folds."""
        for fold in self.default_testing_container:
            assert isinstance(fold, CVFoldInfo)

    def test_len(self) -> None:
        """Test container length dunder method."""
        container = self.default_testing_container
        assert len(container) == 2

    def test_get_scores_dataframe_with_common_names(self) -> None:
        """
        Test that dataframe with scores of default container (that has common metric
        names in train and test parts) features required properties, such as proper
        index casting, metrics columns assembly and overall values consistency.
        """
        container = self.default_testing_container
        df = container.get_scores_dataframe()

        index_corresponds_to_folds = np.array_equal(df.index, container.fold_indices)
        assert index_corresponds_to_folds

        inferred_train_mae = df["MAE"]["train"]
        expected_train_mae = (42, 40)
        mae_inferred_properly = np.array_equal(inferred_train_mae, expected_train_mae)
        assert mae_inferred_properly

        all_values_are_positive = (df > 0).all().all()
        assert all_values_are_positive

    def test_get_scores_dataframe_with_different_names(self) -> None:
        """
        Test that dataframe with scores of a container that has different metrics
        in train and test sets features required properties.
        """
        container = self.container_with_inconsistent_names
        df = container.get_scores_dataframe()

        inferred_interesting_metric = df["Train only metric"]
        expected_value_included = 99 in inferred_interesting_metric.values
        nans_are_present = inferred_interesting_metric.isna().any().any()
        assert (expected_value_included and nans_are_present)

        inferred_mae = df["MAE"]
        no_empty_maes = not inferred_mae.isna().any().any()
        assert no_empty_maes

    def test_get_data_splits(self) -> None:
        """Test that data splits are inferred properly for a container."""
        container = self.default_testing_container
        splits = container.get_data_splits()
        assert len(splits) == len(container)

    @staticmethod
    @pytest.mark.parametrize("container,expected", [
        (
            default_testing_container,
            ("MAE", "R2"),
        ),
        (
            container_with_inconsistent_names,
            ("MAE", "R2", "Train only metric"),
        ),
    ])
    def test_involved_metric_names(
        container: CVResultsContainer,
        expected: tp.Tuple[str],
    ) -> None:
        """Test that names of metrics involved in container are retrieved properly."""
        assert container._involved_metrics_names == expected


class TestCrossValidate(object):
    """Tests for the only user-facing API: ``cross_validate()``."""

    @staticmethod
    @pytest.mark.parametrize("cross_validatable_model", MODELS_TO_TEST)
    @pytest.mark.parametrize("cv_strategy_config", VALID_CV_STRATEGY_CONFIGS)
    def test_cross_validate_works_with_all_key_models_and_expected_configs(
        cross_validatable_model: TCrossValidatableModel,
        modeling_dataset: pd.DataFrame,
        cv_strategy_config: TCVStrategyConfig,
    ) -> None:
        """
        Test that cross-validation output is generated successfully for
        valid models and expected configs.
        """
        scores = cross_validate(
            model=cross_validatable_model,
            data=modeling_dataset,
            cv_strategy_config=cv_strategy_config,
        )
        has_non_zero_scores = (scores.abs() > 0).any().any()
        assert has_non_zero_scores
        column_we_expect_to_find = scores["mae", "test"]
        assert column_we_expect_to_find.mean() > 0.01
        assert isinstance(column_we_expect_to_find, pd.Series)

    @staticmethod
    @pytest.mark.parametrize("cross_validatable_model", MODELS_TO_TEST)
    @pytest.mark.parametrize("cv_strategy_config", VALID_CV_STRATEGY_CONFIGS)
    def test_cross_validate_retrieves_splits_if_requested(
        cross_validatable_model: TCrossValidatableModel,
        modeling_dataset: pd.DataFrame,
        cv_strategy_config: TCVStrategyConfig,
    ) -> None:
        """
        Test that cross-validation function returns relevant data splits if requested.
        """
        scores, splits = cross_validate(
            model=cross_validatable_model,
            data=modeling_dataset,
            cv_strategy_config=cv_strategy_config,
            return_splits=True,
        )
        for data in splits.values():
            for df in data.values():
                assert df.size > 10

    @staticmethod
    @pytest.mark.parametrize("cross_validatable_model", MODELS_TO_TEST)
    @pytest.mark.parametrize("cv_strategy_config", INVALID_CV_STRATEGY_CONFIGS)
    def test_cross_validate_at_all_key_models_raises_on_unexpected_configs(
        cross_validatable_model: TCrossValidatableModel,
        modeling_dataset: pd.DataFrame,
        cv_strategy_config: TCVStrategyConfig,
    ) -> None:
        """Test that cross-validation API raises error on invalid config."""
        with pytest.raises(expected_exception=(ValueError, TypeError)):
            cross_validate(
                model=cross_validatable_model,
                data=modeling_dataset,
                cv_strategy_config=cv_strategy_config,
            )


@pytest.mark.parametrize("model", MODELS_TO_TEST)
@pytest.mark.parametrize("cross_validator_type", [
    ShuffleSplit,
    KFold,
    TimeSeriesSplit,
    RepeatedKFold,
])
@pytest.mark.parametrize("n_splits", [2, 3, 5])
def test_produce_cv_results_container(
    model: TCrossValidatableModel,
    modeling_dataset: pd.DataFrame,
    cross_validator_type: tp.Type[BaseCrossValidator],
    n_splits: int,
) -> None:
    """Test that a container with CV metrics is produced properly."""
    cross_validator = cross_validator_type(n_splits=n_splits)
    container = _produce_cv_results_container(model, modeling_dataset, cross_validator)

    # Testing that it covers all folds.
    assert len(container) == cross_validator.get_n_splits()

    # Testing that it has positive metrics.
    for attribute_name in ("train_metrics", "test_metrics"):
        for metrics in getattr(container, attribute_name):
            assert any(value > 0.01 for value in metrics.values())

    # Testing that splits are relevant.
    splits = container.get_data_splits()
    for data in splits.values():
        for df in data.values():
            assert df.size > 10


@pytest.mark.parametrize("cv_strategy_config", VALID_CV_STRATEGY_CONFIGS)
def test_build_cross_validator(cv_strategy_config: TCVStrategyConfig) -> None:
    """
    Test that cross-validator builder function works as expected.
    We build a cross-validator from a config and check that it has a ``split()`` method.
    """
    built_object = _build_cross_validator(cv_strategy_config)
    its_split_method = getattr(built_object, "split", None)
    assert callable(its_split_method)


@pytest.mark.parametrize("config,expected_type", [
    (
        None,
        type(None),
    ),
    (
        5,
        int,
    ),
    (
        {
            "class_name": "sklearn.model_selection.ShuffleSplit",
            "kwargs": {
                "n_splits": 10,
            },
        },
        ShuffleSplit,
    ),
])
def test_parse_cv_strategy_from_config(
    config: TCVStrategyConfig,
    expected_type: tp.Type,
) -> None:
    """Test that CV strategy is parsed from config properly."""
    parsed = _parse_cv_strategy_from_config(config)
    assert isinstance(parsed, expected_type)


class TestUtilities(object):
    """Tests for private utility functions of the module."""

    @staticmethod
    @pytest.mark.parametrize("sequence,expected", [
        (
            (
                {
                    "MAE": 42,
                    "R2": 0.9,
                    "Interesting metric": 99,
                },
                {
                    "MAE": 47,
                    "R2": 0.2,
                },
            ),
            {"MAE", "R2", "Interesting metric"},
        ),
    ])
    def test_get_unique_keys_from_sequence_of_dicts(
        sequence: tp.Sequence[tp.Dict[tp.Any, tp.Any]],
        expected: tp.Set[tp.Any],
    ) -> None:
        """Test that unique keys are retrieved from a sequence of dicts properly."""
        retrieved = _get_unique_keys_from_sequence_of_dicts(sequence)
        assert retrieved == expected
