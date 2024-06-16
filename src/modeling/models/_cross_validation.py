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
Cross-validation module.
"""
import logging
import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, check_cv

from ..api.model import TCrossValidatableModel
from ..utils import load_obj

logger = logging.getLogger(__name__)

TCVStrategyConfig = tp.Optional[int | tp.Dict[str, tp.Any]]
TCVStrategy = tp.Optional[int | BaseCrossValidator]
TCVDataSplits = tp.Dict[
    int,
    tp.Dict[tp.Literal["train_data", "test_data"], pd.DataFrame],
]


@dataclass(frozen=True)
class CVFoldInfo(object):
    """
    Simple data structure to store info about a single cross-validation fold.

    Constructor args:
        index: Corresponding to a fold to distinguish it in a sequence.
        train_data: Corresponding train dataframe of this fold.
        test_data: Corresponding test dataframe of this fold.
        train_metrics: Achieved in this fold.
        test_metrics: Achieved in this fold.
    """
    index: int
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    train_metrics: tp.Mapping[str, float]
    test_metrics: tp.Mapping[str, float]


@dataclass(frozen=True)
class CVResultsContainer(object):
    """
    A data structure responsible for storing and representing cross-validation results.

    Constructor args:
        fold_infos: Sequence of fold information objects.
    """
    fold_infos: tp.Sequence[CVFoldInfo]

    def __post_init__(self) -> None:
        """
        Post init method to handle inputs validation, given that the class is frozen.

        Raises:
            ValueError: If attempted to initialize from folds with duplicating indices.
        """
        found_duplicating_indices = (
            len(self.fold_indices) != len(set(self.fold_indices))
        )
        if found_duplicating_indices:
            raise ValueError(
                "Cannot initialize CV results container with duplicating fold indices.",
            )

    def __iter__(self) -> tp.Iterator[tp.Any]:
        """Overridden iterator dunder to simplify iterations over self."""
        return iter(self.fold_infos)

    def __len__(self) -> int:
        """Object length getter."""
        return len(self.fold_infos)

    @property
    def fold_indices(self) -> tp.Tuple[int]:
        """Get indices of folds involved in self."""
        return tuple(fold.index for fold in self)

    @property
    def train_metrics(self) -> tp.Tuple[tp.Dict[str, float]]:
        """Get all train metrics involved in self arranged in sequence."""
        return tuple(fold.train_metrics for fold in self)

    @property
    def test_metrics(self) -> tp.Tuple[tp.Dict[str, float]]:
        """Get all test metrics involved in self arranged in sequence."""
        return tuple(fold.test_metrics for fold in self)

    def get_scores_dataframe(self) -> pd.DataFrame:
        """
        Get dataframe representation of self that could be conveniently visualized
        or saved as artifact.

        Returns:
            A multiindex dataframe with train and test metric values
            for all metrics involved.
        """

        df = pd.DataFrame(index=self.fold_indices)

        for metric_name in self._involved_metrics_names:
            for data_type in ("train", "test"):
                corresponding_array = tuple(
                    metrics.get(metric_name, np.nan)
                    for metrics in getattr(self, f"{data_type}_metrics")
                )
                df[(metric_name, data_type)] = corresponding_array

        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.rename_axis(mapper="Fold", axis="index", inplace=True)

        return df

    def get_data_splits(self) -> TCVDataSplits:
        """
        Get a mapping from fold index to train and test datasets
        corresponding to this index.

        Returns:
            A mapping from integer fold indices to dictionaries with 2 items:
            train and test dataframes accordingly.
        """
        return {
            fold.index: {
                "train_data": fold.train_data,
                "test_data": fold.test_data,
            }
            for fold in self
        }

    @property
    def _involved_metrics_names(self) -> tp.Tuple[str]:
        """Get metric names involved in self. Sorts in alphabetic order."""
        train_metric_names = _get_unique_keys_from_sequence_of_dicts(self.train_metrics)
        test_metric_names = _get_unique_keys_from_sequence_of_dicts(self.test_metrics)
        union = train_metric_names | test_metric_names
        return tuple(sorted(union))


def _produce_cv_results_container(
    model: TCrossValidatableModel,
    data: pd.DataFrame,
    cross_validator: BaseCrossValidator,
    **predict_kwargs: tp.Any,
) -> CVResultsContainer:
    """
    Produce a container with CV results for a given model, data,
    and initialized sklearn ``BaseCrossValidator`` instance.

    Args:
        model: To cross-validate.
        data: To cross-validate on.
        cross_validator: Initialized sklearn cross-validator object capable of providing
            the ``cv.split(data)`` interface.
        **predict_kwargs: To be additionally supplied to model.evaluate_metrics().

    Returns:
        A container with fold-by-fold information such as indices, metrics,
        corresponding data slices.
        It can, for example, be reviewed in dataframe representation.
    """

    fold_infos = []

    for fold_index, (train_indices, test_indices) in enumerate(cross_validator.split(data)):  # noqa: E501
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]
        trained_model = model.fit(train_data)
        fold_info = CVFoldInfo(
            index=fold_index,
            train_data=train_data,
            test_data=test_data,
            train_metrics=trained_model.evaluate_metrics(
                data=train_data, **predict_kwargs,
            ),
            test_metrics=trained_model.evaluate_metrics(
                data=test_data, **predict_kwargs,
            ),
        )
        fold_infos.append(fold_info)

    return CVResultsContainer(fold_infos)


def cross_validate(
    model: TCrossValidatableModel,
    data: pd.DataFrame,
    cv_strategy_config: tp.Optional[TCVStrategyConfig] = None,
    decimals: tp.Optional[int] = 2,
    return_splits: bool = False,
    **predict_kwargs: tp.Any,
) -> pd.DataFrame | tp.Tuple[pd.DataFrame, TCVDataSplits]:
    """
    Cross-validate a model on a given data using a given config for CV strategy.

    Args:
        model: To cross-validate.
        data: To cross-validate on.
        cv_strategy_config: To parse sklearn-compatible strategy from. Can be:
            - ``None``
            - Positive ``int``
            - Config of an instance of sklearn.model_selection.BaseCrossValidator, e.g.:
                {
                    "class": "sklearn.model_selection.ShuffleSplit",
                    "kwargs": {
                        "n_splits": 10,
                    },
                }
        decimals: To round output to.
        return_splits: Whether to return data corresponding to CV splits or not.
        **predict_kwargs: To be additionally supplied to model.evaluate_metrics().

    Returns:
        A dataframe with cross-validation metrics for each fold,
        rounded to 2 decimals by default or whatever specified by user.
    """

    cross_validator = _build_cross_validator(cv_strategy_config)
    cv_results = _produce_cv_results_container(
        model=model,
        data=data,
        cross_validator=cross_validator,
        **predict_kwargs,
    )
    scores_df = cv_results.get_scores_dataframe()
    if decimals:
        scores_df = scores_df.round(decimals)

    if return_splits:
        splits = cv_results.get_data_splits()
        return scores_df, splits

    return scores_df


def _build_cross_validator(cv_strategy_config: TCVStrategyConfig) -> BaseCrossValidator:
    """
    Build a sklearn cross validator object capable of providing the``cv.split(data)``
    interface from user-specified strategy config.

    Args:
        cv_strategy_config: To parse sklearn-compatible strategy from.

    Returns:
        Sklearn cross-validator object.
    """
    cv_strategy = _parse_cv_strategy_from_config(cv_strategy_config)
    cross_validator = check_cv(cv_strategy)
    logger.info(f"Cross-validating using: {cross_validator}")
    return cross_validator


def _parse_cv_strategy_from_config(
    cv_strategy_config: TCVStrategyConfig,
) -> TCVStrategy:
    """
    Parse a cross-validation strategy from supplied config.
    **Strategy** in sklearn terminology is an input compatible with ``cv`` argument
    of their API. Examples of valid strategies are:
        - None
        - Positive integer
        - Inheritors of ``sklearn.model_selection.BaseCrossValidator``, e.g. ``KFold``.

    Args:
        cv_strategy_config: To parse strategy from.

    Returns:
        Sklearn-compatible cross-validation strategy. Means that it's a compatible input
        for ``cv`` arguments of sklearn APIs.
    """
    if (cv_strategy_config is None) or isinstance(cv_strategy_config, int):
        return cv_strategy_config

    return load_obj(cv_strategy_config["class_name"])(**cv_strategy_config["kwargs"])


def _get_unique_keys_from_sequence_of_dicts(
    sequence: tp.Sequence[tp.Dict[tp.Any, tp.Any]],
) -> tp.Set[tp.Any]:
    """
    For a sequence of dictionaries, get a set with all unique keys they contain.

    Args:
        sequence: To get unique keys for.

    Returns:
        A set of unique keys that dictionaries in this sequence contain.
    """
    sets_of_keys = tuple(set(dictionary.keys()) for dictionary in sequence)
    return set.union(*sets_of_keys)
