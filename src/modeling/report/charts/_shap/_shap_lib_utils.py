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
This module contains a refactored COPY of couple utility functions from shap library
"""

import typing as tp

import numpy as np
import numpy.typing as npt

from ._utils import (  # noqa: WPS436
    FeatureShapExplanation,
    ShapExplanation,
    encode_array_if_needed,
)

_THRESHOLD = 1e-8
_MAX_STEP_SIZE = 50
_P = tp.TypeVar("_P", bound=npt.NBitBase)  # noqa: WPS111
_TNumericNDArray = npt.NDArray[np.number[_P]]


def _calculate_single_column_interactions(
    data: _TNumericNDArray[_P],
    row_indices: npt.NDArray[np.intp],
    reference_column_index: int,
    data_column_sorting: npt.NDArray[np.intp],
    ignore_indices: npt.NDArray[np.intp],
    n_rows: int,
    step_size: int,
    sampled_shap_values_column: _TNumericNDArray[_P],
) -> float:
    """ Calculate the interactions for the column with the given
    `reference_column_index`
    """
    reference_column_values = encode_array_if_needed(
        data[row_indices, reference_column_index][data_column_sorting],
        dtype=np.float64,
    )
    if reference_column_index in ignore_indices:
        return 0
    cumulative_correlation = _calculate_cumulative_correlation_by_buckets(
        sampled_shap_values_column, reference_column_values, n_rows, step_size,
    )
    cumulative_correlation_for_nans = _calculate_cumulative_correlation_by_buckets(
        sampled_shap_values_column,
        np.isnan(reference_column_values),
        n_rows,
        step_size,
    )
    return max(cumulative_correlation, cumulative_correlation_for_nans)


def _calculate_columns_interactions(
    data: _TNumericNDArray[_P],
    shap_values_column: FeatureShapExplanation,
    row_indices: npt.NDArray[np.intp],
    data_column_sorting: npt.NDArray[np.intp],
    n_rows: int,
    n_columns: int,
    ignore_indices: npt.NDArray[np.intp],
) -> npt.NDArray[np.intp]:
    """ Calculate the interactions of all columns """
    sampled_shap_values_column = (
        shap_values_column.values[row_indices][data_column_sorting]
    )
    step_size = int(max(min(n_rows // 10, _MAX_STEP_SIZE), 1))
    interactions = []
    for reference_column_index in range(n_columns):
        interactions.append(_calculate_single_column_interactions(
            data,
            row_indices,
            reference_column_index,
            data_column_sorting,
            ignore_indices,
            n_rows,
            step_size,
            sampled_shap_values_column,
        ))
    return np.argsort(-np.abs(interactions))


def _get_data_column_sorting(
    shap_values_column: FeatureShapExplanation,
    row_indices: _TNumericNDArray[_P],
) -> npt.NDArray[np.intp]:
    """ Calculate data column sorting """
    data_column = shap_values_column.data[row_indices]
    return np.argsort(data_column)


def potential_interactions(
    shap_values_column: FeatureShapExplanation,
    shap_values_matrix: ShapExplanation,
) -> npt.NDArray[np.intp]:
    """
    Order other features by how much interaction they seem to have
    with the feature at the given index.

    This just bins the SHAP values for a feature along that feature's value
    and computes the total correlation between each of those buckets.

    For true Shapley interaction index values for SHAP see the interaction_contribs
    option implemented in XGBoost.
    """

    # ignore row_indices that are identical to the column
    ignore_indices = np.where(
        (
            shap_values_matrix.values.T - shap_values_column.values
        ).T.std(0) < _THRESHOLD,
    )[0]

    data = shap_values_matrix.data
    n_rows, n_columns = data.shape
    row_indices = _sample_rows(n_rows)

    data_column_sorting = _get_data_column_sorting(shap_values_column, row_indices)
    return _calculate_columns_interactions(
        data,
        shap_values_column,
        row_indices,
        data_column_sorting,
        n_rows,
        n_columns,
        ignore_indices,
    )


def _calculate_cumulative_correlation_by_buckets(
    shap_values: _TNumericNDArray[_P],
    feature_values: _TNumericNDArray[_P],
    n_rows: int,
    step_size: int,
    threshold: float = _THRESHOLD,
) -> float:
    """
    Splits shap_values and values by buckets and
    calculates total abs correlation for along those buckets
    """

    abs_sum_of_values = np.sum(np.abs(feature_values))
    if abs_sum_of_values < threshold:
        return 0

    total_correlation = 0
    for start_index in range(0, n_rows, step_size):
        end_index = start_index + step_size
        bin_feature_values = feature_values[start_index:end_index]
        bin_shap_values = shap_values[start_index:end_index]
        feature_is_variate = np.std(bin_feature_values) > 0  # type: ignore
        shap_is_variate = np.std(bin_shap_values) > 0  # type: ignore
        if feature_is_variate and shap_is_variate:
            total_correlation += abs(
                np.corrcoef(
                    bin_shap_values, bin_feature_values,  # type: ignore
                )[0, 1],
            )
    return total_correlation


def _sample_rows(
    n_rows: int, max_rows: int = 1000, random_seed: int = 0,
) -> npt.NDArray[np.intp]:
    row_indices = np.arange(n_rows)
    if n_rows > max_rows:
        np.random.default_rng(seed=random_seed).shuffle(row_indices)
        row_indices = row_indices[:max_rows]
    return row_indices
