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


import typing as tp
from itertools import product

import pandas as pd

from modeling import ModelBase


class ModelPrediction(object):
    """
    Class to store model predictions and relevant feature values
    that were used for making the model prediction.

    Notes:
        This is a wrapper around two pd.Series that allows you to access
        model predictions and feature values without explicit usage
        with pandas multi-index.
    """
    def __init__(
        self,
        model_predictions: tp.Iterable[float],
        feature_to_plot_values: tp.Iterable[float],
        index: pd.Index,
    ) -> None:
        self._model_predictions = pd.Series(
            model_predictions,
            index=index,
        ).sort_index(kind="stable")
        self._feature_to_plot_values = pd.Series(
            feature_to_plot_values,
            index=index,
        ).sort_index(kind="stable")

    def get_model_predictions_at_index(
        self,
        index_values: tp.Tuple[int, ...],
    ) -> tp.List[float]:
        model_prediction_at_index = self._model_predictions.loc[index_values]
        if isinstance(model_prediction_at_index, pd.Series):
            return tp.cast(
                tp.List[float],
                self._model_predictions.loc[index_values].to_list(),
            )
        return [model_prediction_at_index]

    def get_model_feature_values_at_index(
        self,
        index_values: tp.Tuple[int, ...],
    ) -> tp.List[float]:
        feature_to_plot_at_index = self._feature_to_plot_values.loc[index_values]
        if isinstance(feature_to_plot_at_index, pd.Series):
            return tp.cast(
                tp.List[float],
                feature_to_plot_at_index.to_list(),
            )
        return [feature_to_plot_at_index]

    @property
    def prediction_min_value(self) -> float:
        return tp.cast(
            float,
            self._model_predictions.min(),
        )

    @property
    def prediction_max_value(self) -> float:
        return tp.cast(
            float,
            self._model_predictions.max(),
        )

    @property
    def feature_to_plot_min_value(self) -> float:
        return tp.cast(
            float,
            self._feature_to_plot_values.min(),
        )

    @property
    def feature_to_plot_max_value(self) -> float:
        return tp.cast(
            float,
            self._feature_to_plot_values.max(),
        )

    def calculate_prediction_size_at_index(
        self,
        index_values: tp.Tuple[int, ...],
    ) -> int:
        return len(self._model_predictions.loc[index_values])


def make_model_predictions_for_features_to_plot(
    features_to_plot: tp.Mapping[str, tp.List[float]],
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
    model: ModelBase,
    static_feature_values: tp.Mapping[str, float],
) -> tp.Dict[str, ModelPrediction]:
    """
    Create a mapping from ``feature_to_plot`` name
    into ``ModelPrediction`` that collects model prediction
    and corresponding feature values.

    Notes:
        pd.Series wrapped with ``ModelPrediction`` has
        multi-index corresponding to different combinations
        of feature values inside ``features_to_manipulate``.

    Examples:
        ModelPrediction.get_model_predictions_at_index((0, 1, 2))
        will give you the model predictions relevant for
        the grid of ``feature_to_plot``
        under consideration that values for ``features_to_manipulate``
        are picked from the grid using the input indices (e.g., 0, 1, 2).
    """
    data = _create_empty_data_with_index(features_to_manipulate)
    model_predictions = {}
    for feature_to_plot, grid_values in features_to_plot.items():
        feature_to_plot_values = _add_feature_to_plot_to_data(
            data,
            feature_to_plot,
            grid_values,
        )
        model_prediction = model.predict(
            _prepare_data_for_model_prediction(
                feature_to_plot_values,
                features_to_manipulate,
                feature_to_plot,
                model,
                static_feature_values,
            ),
        )
        model_predictions[feature_to_plot] = ModelPrediction(
            model_prediction,
            feature_to_plot_values[feature_to_plot],
            feature_to_plot_values.index,
        )
    return model_predictions


def _prepare_data_for_model_prediction(
    feature_to_plot_values: pd.DataFrame,
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
    feature_to_plot: str,
    model: ModelBase,
    static_conditions: tp.Mapping[str, float],
) -> pd.DataFrame:
    feature_to_plot_values = feature_to_plot_values.copy()
    required_columns = (
        set(model.features_in)
        - set(feature_to_plot_values.index.names)
        - {feature_to_plot}
    )
    if not required_columns.issubset(set(static_conditions)):
        raise ValueError(
            f"Model requires features {required_columns} that are not"
            " included either in `static_conditions`,"
            " `features_to_plot` or `manageable_conditions.",
        )
    feature_to_plot_values = feature_to_plot_values.assign(
        **{condition: static_conditions[condition] for condition in required_columns},
    )
    return _reset_index_and_replace_index_with_values(
        feature_to_plot_values,
        features_to_manipulate,
        feature_to_plot,
    )


def _reset_index_and_replace_index_with_values(
    feature_to_plot_values: pd.DataFrame,
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
    feature_to_plot: str,
) -> pd.DataFrame:
    condition_values = []
    for condition, conditions_values in features_to_manipulate.items():
        if feature_to_plot == condition:
            continue
        condition_index_to_value_mapping = dict(enumerate(conditions_values))
        condition_index = feature_to_plot_values.reset_index(level=condition)[condition]
        condition_values.append(
            (
                condition_index
                .reset_index(drop=True)
                .replace(condition_index_to_value_mapping)
            ),
        )
    return pd.concat(
        [*condition_values, feature_to_plot_values.reset_index(drop=True)],
        axis=1,
    )


def _add_feature_to_plot_to_data(
    data: pd.DataFrame,
    feature_to_plot: str,
    grid_values: tp.List[float],
) -> pd.DataFrame:
    data = data.copy()
    return pd.concat(
        [data.assign(**{feature_to_plot: grid_value}) for grid_value in grid_values],
    )


def _create_empty_data_with_index(
    manageable_conditions: tp.Mapping[str, tp.List[float]],
) -> pd.DataFrame:
    conditions = list(manageable_conditions)
    return pd.DataFrame(
        data=product(
            *[
                range(len(condition_value))
                for condition_value in manageable_conditions.values()
            ],
        ),
        columns=conditions,
    ).set_index(conditions)
