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

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from preprocessing.tags_config import TagResampleParameters, TagsConfig

SUPPORTED_METHODS = frozenset(("mean", "min", "max", "first", "sum", "last", "median"))

logger = logging.getLogger(__name__)


def resample_data(
    data: pd.DataFrame,
    resample_config: TagsConfig[TagResampleParameters],
    timestamp_col: str,
    errors: tp.Literal["raise", "coerce"] = "coerce",
    default_method: str = "mean",
    default_rule: str = "1H",
    default_offset: str = "0min",
    **kwargs: tp.Any,
) -> pd.DataFrame:
    """Resample data according to configurations in resample_df.

    Args:
        data: input data
        resample_config: TagsConfig with resample configurations for each tag
        timestamp_col: timestamp column name to use as index
        errors: 'raise' or 'coerce', behavior when an invalid method is specified
        default_method: method to use when agg_method is missing from resample_df
        default_rule: rule to use when rule is missing from resample_df
        default_offset: offset to use when offset is missing from resample_df
        **kwargs: additional arguments to pass to pandas resample
    Returns:
        data_resampled: resampled output data
    """

    if not is_datetime(data[timestamp_col]):
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])

    data.set_index(timestamp_col, drop=True, inplace=True)

    result_series = {}

    for tag in data.columns:
        # Get resampling parameters for the tag, or use defaults
        agg_method = _get_valid_agg_method(tag, resample_config, errors, default_method)
        rule = _get_valid_resample_freq(tag, resample_config, default_rule)
        offset = _get_valid_resample_offset(tag, resample_config, default_offset)

        # Apply resampling with extracted or default parameters
        resampled_data = data[tag].resample(
            rule=rule,
            offset=offset,
            **(kwargs or {}),
        ).agg(agg_method)
        result_series[tag] = resampled_data

    data_resampled = pd.DataFrame(result_series)

    return data_resampled.reset_index()


def _get_valid_agg_method(
    tag: str,
    resample_config: TagsConfig[TagResampleParameters],
    errors: tp.Literal["raise", "coerce"],
    default_method: str = "mean",
) -> str:
    """Select valid aggregation method for a tag.

    Selects the aggregation method for a tag from the
    data dictionary. If not defined, raise error or default
    to a default aggregation method.

    Args:
        tag: string of the tag
        resample_config: TagsConfig with resample configurations for each tag
        errors: str {'raise', 'coerce'}, raise errors if tag has no agg_method
            defined in data dictionary or coerce to default
        default_method: method to use when agg_method is missing from td

    Returns:
        data_resampled: resampled output data
    """
    method = resample_config[tag].resample_method
    if method is None:
        method = default_method  # type: ignore

    if isinstance(method, str) and method in SUPPORTED_METHODS:
        return method

    if errors == "raise":
        if pd.isna(method):
            raise ValueError(f"No aggregation method defined for column {tag}")
        raise ValueError(f"Invalid aggregation method defined for column {tag}")
    elif errors == "coerce":
        return default_method
    raise ValueError(f"Unknown errors behavior handling: {errors}")


def _get_valid_resample_freq(
    tag: str,
    resample_config: TagsConfig[TagResampleParameters],
    default_rule: str,
) -> str:
    """Select valid aggregation method for a tag.

    Args:
        tag: string of the tag
        resample_config: TagsConfig with resample configurations for each tag
        default_rule: method to use when rule is missing from td

    Returns:
        rule: rule to use for resampling
    """
    rule = resample_config[tag].resample_freq
    if rule is None:
        rule = default_rule
    return rule


def _get_valid_resample_offset(
    tag: str,
    resample_config: TagsConfig[TagResampleParameters],
    default_offset: str,
) -> str:
    """Select valid aggregation method for a tag.

    Args:
        tag: string of the tag
        resample_df: tag meta schema
        default_offset: method to use when offset is missing from td

    Returns:
        offset: offset to use for resampling
    """
    offset = resample_config[tag].resample_offset
    if offset is None:
        offset = default_offset
    return offset
