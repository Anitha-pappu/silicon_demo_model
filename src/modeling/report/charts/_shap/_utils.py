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
from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from modeling.api import ShapExplanation

_P = tp.TypeVar("_P", bound=npt.NBitBase)  # noqa: WPS111
_TNumericNDArray = npt.NDArray[np.number[_P]]

_INITIAL_PERCENTILE_MIN = 5
_INITIAL_PERCENTILE_MAX = 95
_BACKUP_PERCENTILE_MIN = 1
_BACKUP_PERCENTILE_MAX = 99


@dataclass
class FeatureShapExplanation(object):
    """
    Stores feature's data and shap values

    Attributes:
        values: contains shap values
        data: contains initial feature data
    """

    # Using ``values`` to keep the same interface as shap.Explanations
    values: npt.NDArray[tp.Any]  # noqa: WPS110
    data: npt.NDArray[tp.Any]


def extract_explanations_for_given_features(
    features: tp.List[str], shap_explanation: ShapExplanation,
) -> tp.Dict[str, FeatureShapExplanation]:
    explanation_for_feature = {}
    for feature in features:
        feature_index_in_shap_data = shap_explanation.feature_names.index(feature)
        explanation_for_feature[feature] = FeatureShapExplanation(
            data=encode_array_if_needed(
                shap_explanation.data[:, feature_index_in_shap_data],
            ),
            values=shap_explanation.values[:, feature_index_in_shap_data],
        )
    return explanation_for_feature


def sort_features(
    features: tp.List[str],
    order_by: _TNumericNDArray[_P] | None,
    shap_explanation: ShapExplanation,
    descending: bool,
) -> tp.List[str]:
    if order_by is None:
        order_by = _get_order_by_mean_abs_shaps(features, shap_explanation)
    if len(features) != len(order_by):
        raise ValueError(
            "Please provide same length collections for `order_by` and `features`",
        )
    features = np.array(features)[np.argsort(order_by)].tolist()
    if descending:
        features.reverse()
    return features


def _get_order_by_mean_abs_shaps(
    features: tp.List[str], shap_explanation: ShapExplanation,
) -> _TNumericNDArray[_P]:
    """
    Returns an array where the i-th value corresponds to the average absolute shap value
    for the i-th feature.
    """
    mean_shap_values = np.abs(shap_explanation.values).mean(axis=0)
    order_by = np.zeros_like(features, dtype=np.float64)
    for index, feature in enumerate(features):
        index_in_shaps = shap_explanation.feature_names.index(feature)
        order_by[index] = mean_shap_values[index_in_shaps]
    return order_by  # type: ignore  #(incorrect mypy parsing of np)


def encode_array_if_needed(
    array: npt.NDArray[tp.Any],
    dtype: tp.Any = np.float64,  # todo: make result generic by this arg
) -> npt.NDArray[tp.Any]:
    try:
        return array.astype(dtype)
    except ValueError:
        unique_values = np.unique(array)
        encoding_dict = {string: index for index, string in enumerate(unique_values)}
        return np.array([encoding_dict[string] for string in array], dtype=dtype)


def _crop_feature_values(
    feature_values: _TNumericNDArray[_P],
) -> _TNumericNDArray[_P]:
    """
    Crops feature values to avoid extreme outliers that destroy the color bar.
    """
    feature_values = feature_values.copy()
    # type ignored because NDArray does not support
    # __array__ for some reason
    min_value = np.nanpercentile(
        feature_values,  # type: ignore
        _INITIAL_PERCENTILE_MIN,
    )
    max_value = np.nanpercentile(
        feature_values,  # type: ignore
        _INITIAL_PERCENTILE_MAX,
    )
    if min_value == max_value:
        min_value = np.nanpercentile(
            feature_values,  # type: ignore
            _BACKUP_PERCENTILE_MIN,
        )
        max_value = np.nanpercentile(
            feature_values,  # type: ignore
            _BACKUP_PERCENTILE_MAX,
        )
    if min_value == max_value:
        min_value = np.nanmin(feature_values)
        max_value = np.nanmax(feature_values)
    # Rare numerical precision issues
    if min_value > max_value:
        min_value = max_value
    feature_values[feature_values > max_value] = max_value
    feature_values[feature_values < min_value] = min_value
    return tp.cast(npt.NDArray[tp.Any], feature_values)


def _normalize_feature_value_for_color_bar(
    feature_values: _TNumericNDArray[_P],
) -> _TNumericNDArray[_P]:
    """Crop and normalizes feature values to be in the range [0, 1]"""
    cropped_values = _crop_feature_values(feature_values)
    fractions = (
        (cropped_values - np.nanmin(cropped_values))
        / (np.nanmax(cropped_values) - np.nanmin(cropped_values))
    )
    return tp.cast(npt.NDArray[tp.Any], fractions)
