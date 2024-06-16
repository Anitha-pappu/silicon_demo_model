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
Preprocessing utils code
"""
import importlib
import logging
import typing as tp

import numpy as np
import pandas as pd

from preprocessing.tags_config import (
    TagMetaParameters,
    TagOutliersParameters,
    TagRawParameters,
    TagsConfig,
    TTagParameters,
)

logger = logging.getLogger(__name__)

TPrefillMethodOptions = tp.Literal["min-max", "IQR", "3-sigma"]


def create_summary_table(
    data: pd.DataFrame,
    tags_meta_config: TagsConfig[TagMetaParameters] | None = None,
    percentile: float = 0.05,
) -> pd.DataFrame:
    """This function create descriptive summary table for input data

    Args:
        data (pd.DataFrame): input data.
        tags_meta_config: TagsConfig with tags meta data
        percentile (float, optional): percentile to use as threshold for oulier check.
             When tag dictionary is missing, (range_min, range_max)
             will be (percentile, 1-percentile). Defaults to 0.05.

    Returns:
        pd.pd.DataFrame: _description_
    """

    pd_desc = data.describe().T
    pd_desc["null_count"] = data.isnull().sum()
    pd_desc["inf_count"] = data[data == np.inf].count()

    outlier_count = count_outlier(data, tags_meta_config, percentile)

    resulting_df = pd_desc.join(outlier_count, how="left")

    resulting_df["percent_below_range_min"] = (
        100 * resulting_df["below_range_min_count"] / resulting_df["count"]
    )

    resulting_df["percent_above_range_max"] = (
        100 * resulting_df["above_range_max_count"] / resulting_df["count"]
    )

    return resulting_df


def preprocessing_output_summary(
    tags_raw_config: TagsConfig[TTagParameters],
    tags_meta_config: TagsConfig[TTagParameters],
    tags_outliers_config: TagsConfig[TTagParameters],
    tags_impute_config: TagsConfig[TTagParameters],
    tags_on_off_config: TagsConfig[TTagParameters],
    tags_resample_config: TagsConfig[TTagParameters],
    outlier_summary: pd.DataFrame = None,
    interpolate_summary: pd.DataFrame = None,
    tag_name_col: str = "tag_name",
) -> pd.DataFrame:
    """Create a descriptive summary table for input data by joining preprocessing
    datasets and TagsConfig dataframes by tag_name.

    Args:
        tags_raw_config: TagsConfig object containing the raw tag information.
        tags_meta_config: TagsConfig object containing the tag meta information.
        tags_outliers_config: TagsConfig object containing the tag outliers parameters.
        tags_impute_config: TagsConfig object containing the tag impute parameters.
        tags_on_off_config: TagsConfig object containing the tag on/off parameters.
        tags_resample_config: TagsConfig object containing the tag resample parameters.
        outlier_summary: Summary table of outliers handling for each tag.
        interpolate_summary: Summary table of interpolated values for each tag.
        tag_name_col (str, optional): column to join datasets. Defaults to "tag_name".

    Returns:
        pd.DataFrame: Preprocessing summary table.
    """

    config_list = [
        tags_raw_config,
        tags_meta_config,
        tags_outliers_config,
        tags_impute_config,
        tags_on_off_config,
        tags_resample_config,
    ]

    config_list = [config for config in config_list if config is not None]
    config_dfs = [config.to_df() for config in config_list]

    summary_list = [outlier_summary, interpolate_summary]
    summary_datasets = [summary for summary in summary_list if summary is not None]

    if not config_dfs and not summary_datasets:
        raise ValueError("Both configs and summary datasets must not be empty")

    datasets = config_dfs + summary_datasets

    # Ensure the join column exists on all DataFrames and is of type str
    datasets = [
        df.assign(**{tag_name_col: df[tag_name_col].astype(str)})
        for df in datasets if tag_name_col in df.columns
    ]

    resulting_df = datasets[0]

    for df in datasets[1:]:
        resulting_df = resulting_df.merge(
            df,
            on=tag_name_col,
            how="left",
            suffixes=('', '_drop'),
        )

        # Drop columns that are produced from the join with suffix "_drop"
        drop_columns = [col for col in resulting_df.columns if col.endswith('_drop')]
        resulting_df.drop(columns=drop_columns, axis=1, inplace=True)

    return resulting_df


def get_drop_summary(
    original_data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the percentage summary of dropped outliers for each tag."""
    n_dropped = (
        (original_data.isnull() | original_data.notnull()) & cleaned_data.isnull()
    )

    perc_dropped = (n_dropped.sum() / original_data.shape[0] * 100).round(2)

    summary_df = pd.DataFrame(
        {
            "tag_name": perc_dropped.index,
            "outlier_percentage": perc_dropped.values,
        },
    )
    # Filter out tags with no outliers dropped
    return summary_df[summary_df["outlier_percentage"] > 0]


def get_clip_summary(
    input_data: pd.DataFrame,
    tag_range: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Calculate the percentage summary of clipped outliers for each tag."""
    summary_data = []

    for tag, (lower, upper) in tag_range.items():
        # Skip tag completely if both bounds are NaN
        if np.isnan(lower) and np.isnan(upper):
            continue

        # Calculate the count of clipped values for this tag
        total_clipped = _calculate_clipped_values(input_data[tag], lower, upper)
        perc_clipped = (total_clipped / input_data[tag].notnull().sum() * 100).round(2)

        if total_clipped > 0:
            summary_data.append({"tag_name": tag, "outlier_percentage": perc_clipped})

    return pd.DataFrame(summary_data, columns=["tag_name", "outlier_percentage"])


def _calculate_clipped_values(
    input_series: pd.Series,
    lower: float,
    upper: float,
) -> int:
    """Calculate the count of values that fall outside the given range."""
    lower_cond = ~np.isnan(lower) & (input_series < lower)
    upper_cond = ~np.isnan(upper) & (input_series > upper)

    return int((lower_cond | upper_cond).sum())


def rename_tags(
    tags_raw_config: TagsConfig[TagRawParameters],
    data_to_rename: pd.DataFrame,
) -> pd.DataFrame:
    """
    Renames tags in data_to_rename according to the mapping in rename_config.

    Args:
        tags_raw_config: A TagsConfig object with RawTagParameters models containing
                       'raw_tag' and 'tag_name' attributes.
        data_to_rename: A DataFrame with columns to be renamed based on
                        rename_config mapping.

    Returns:
        A DataFrame with columns replaced according to 'tag_name' mapping.
    """

    rename_df = tags_raw_config.to_df()
    rename_dict = rename_df.set_index('raw_tag')['tag_name'].to_dict()

    missing_columns = set(data_to_rename.columns) - set(rename_dict.keys())
    if missing_columns:
        logger.warning(
            f"Some columns are not in the rename map "
            f"and will not be renamed: {missing_columns}",
        )

    return data_to_rename.rename(columns=rename_dict)


def update_tag_range(
    tags_outliers_config: TagsConfig[TagOutliersParameters],
    tag_range: tp.Dict[str, tp.Tuple[float, float]],
) -> TagsConfig[TagOutliersParameters]:
    """This function updates the tag range in the TagsConfig object.

    Args:
        tags_outliers_config: TagsConfig object containing TagOutliersParameters models.
        tag_range: Dictionary mapping tag names to their corresponding ranges.

    Returns:
        Updated TagsConfig object with new ranges.
    """
    if not isinstance(tag_range, dict):
        raise TypeError("tag_range must be a dictionary")

    if len(tag_range) == 0:  # NOQA: WPS507
        raise ValueError("tag_range must not be empty")

    # Ensure all tags in tag_range are in the TagsConfig
    if not set(tag_range.keys()).issubset(set(tags_outliers_config.keys())):
        raise ValueError("tag_range must be a subset of tags_config keys")

    # Update the range_min and range_max for each tag in tag_range
    for tag_name, (range_min, range_max) in tag_range.items():
        if tag_name in tags_outliers_config:
            tags_outliers_config[tag_name].range_min = range_min
            tags_outliers_config[tag_name].range_max = range_max

    return tags_outliers_config


def get_tag_range(
    outliers_config: TagsConfig[TagOutliersParameters],
    tags: tp.Optional[tp.Iterable[str]] = None,
) -> tp.Dict[str, tp.Tuple[float, float]]:
    """This function creates dictionary of range for each tag in tag dictionary.

    Args:
        outliers_config (TagsConfig[TagOutliersParameters]): tags outliers config
        tags (Optional[Iterable[str]], optional): list of tags to get range,

    Returns:
        Dict: range map
    """

    if tags is not None:
        _check_are_known(outliers_config, tags)

    range_map = {}

    for tag in outliers_config if tags is None else tags:
        range_map[tag] = (
            outliers_config[tag].range_min,
            outliers_config[tag].range_max,
        )

    return range_map


def create_range_map(
    tags_meta_config: TagsConfig[TagMetaParameters],
) -> dict[str, tuple[float, float]]:
    """This function creates dictionary of range for each tag in tag dictionary.

    Args:
        tags_meta_config (TagsConfig): TagsConfig with tags meta data

    Returns:
        Dict: range map
    """

    range_map: dict[str, tuple[float, float]] = {}

    for tag in tags_meta_config.keys():
        if tags_meta_config[tag].min is None or tags_meta_config[tag].max is None:
            continue
        range_map[tag] = (  # type: ignore
            tags_meta_config[tag].min,
            tags_meta_config[tag].max,
        )

    return range_map


def calculate_tag_range(
    data: pd.DataFrame,
    method: TPrefillMethodOptions,
    list_of_tags: tp.Optional[tp.List[str]] = None,
    within_observed_range: bool = True,
) -> tp.Dict[str, tp.Tuple[float, float]]:
    """
    Get prefilled range of tags by using given method on the data.

    Args:
        data: input data
        method: method to use for filling sensor range, default is None
         - min-max: use min and max value of the data
         - IQR: use interquartile range to calculate the whisker value
         - 3-sigma: use 3-sigma rule to calculate the whisker value
        list_of_tags: list of tags to get range,
         default is None. If None and td is provided, then all numeric
         tags will be used.
        within_observed_range: if True, then the calculated sensor range will be
         within whether the observed data range. This is useful when the data is
         not normally distributed and the whisker value is outside the observed.

    Returns:
        a dictionary contains lower and upper limits for the tags
    """

    if list_of_tags is None:
        list_of_tags = data.select_dtypes("number").columns
        logger.warning("list_of_tags is None, using all numeric tags from the data")

    return {
        col: _calculate_sensor_range(
            data[col],
            method,
            within_observed_range,
        )
        for col in list_of_tags if col in data.columns
    }


def count_outside_threshold(
    series: pd.Series,
    threshold: float,
    direction: str = "lower",
) -> float:
    """This function count number of outliers from given series.

    Args:
        series (pd.Series): series of data to check outliers
        threshold (float): threshold
        direction (str, optional): lower/upper direction
            to count outliers. Defaults to "lower".

    Raises:
        ValueError: raises when wrong direction is given.

    Returns:
        float: number of outliers
    """

    if direction == "lower":
        return float(np.sum(series < threshold))
    if direction == "upper":
        return float(np.sum(series > threshold))
    raise ValueError("direction must be either 'lower' or 'upper'")


def count_outlier(
    data: pd.DataFrame,
    tags_meta_config: TagsConfig[TagMetaParameters] | None = None,
    percentile: float = 0.05,
) -> pd.DataFrame:
    """This function count outliers from given data

    Args:
        data (pd.DataFrame): input data
        tags_meta_config: TagsConfig with tags meta data

        percentile (float, optional): percentile to use as
            threshold for oulier check.
            when tag dictionary is missing, (range_min, range_max)
            will be (percentile, 1-percentile).
            Defaults to 0.05.

    Returns:
        pd.DataFrame: _description_
    """

    numeric_cols = data.select_dtypes("number").columns
    description_df = pd.DataFrame(index=numeric_cols)

    if tags_meta_config is not None:
        range_map = create_range_map(tags_meta_config)
    else:
        range_map = {}

    for tag in numeric_cols:
        (range_min, range_max) = range_map.get(
            tag,
            [
                data[tag].quantile(q=percentile),
                data[tag].quantile(q=(1 - percentile)),
            ],
        )

        description_df.loc[tag, "below_range_min_count"] = count_outside_threshold(
            data[tag],
            threshold=range_min,
            direction="lower",
        )

        description_df.loc[tag, "above_range_max_count"] = count_outside_threshold(
            data[tag],
            threshold=range_max,
            direction="upper",
        )

    return description_df


def load_obj(obj_path: str, default_obj_path: str = "") -> tp.Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path. In the case this is provided, `obj_path`
        must be a single name of the object being imported.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.

    Examples:
        Importing an object::

            >>> load_obj("sklearn.linear_model.Ridge")

        Importing using `default_obj_path`::

            >>> load_obj("Ridge", default_obj_path="sklearn.linear_model")
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    loaded_object = getattr(module_obj, obj_name, None)
    if loaded_object is None:
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`.",
        )
    return loaded_object


def _calculate_sensor_range(
    data: pd.Series,
    method: TPrefillMethodOptions,
    within_observed_range: bool = True,
) -> tp.Tuple[float, float]:
    """
    Calculate sensor range based on given method

    Args:
        data: input data
        method: method to use for calculating sensor range
         - min-max: use min and max value of the data
         - IQR: use interquartile range to calculate the whisker value
         - 3-sigma: use 3-sigma rule to calculate the whisker value
        within_observed_range: if True, then the calculated sensor range will be
         within whether the observed data range. This is useful when the data is
         not normally distributed and the whisker value is outside the observed.

    Returns:
        tuple: sensor range
    """
    if method == "min-max":
        range_min = data.min()
        range_max = data.max()
    elif method == "IQR":
        range_min, range_max = _calculate_iqr_whisker(
            data,
            within_observed_range,
        )
    elif method == "3-sigma":
        range_min, range_max = _calculate_three_sigma_whisker(
            data,
            within_observed_range,
        )
    else:
        raise ValueError("method must be one of 'min-max', 'IQR', '3-sigma'")
    return range_min, range_max


def _calculate_iqr_whisker(
    data: pd.Series,
    within_observed_range: bool = True,
) -> tp.Tuple[float, float]:
    """
    Calculate the whisker value based on IQR
    https://en.wikipedia.org/wiki/Interquartile_range

    Args:
        data: input data
        within_observed_range: if True, then the calculated IQR whisker will be
         within whether the observed data range. This is useful when the data is
         not normally distributed and the whisker value is outside the observed.

    Returns:
        tuple: whisker value
    """
    q1 = data.quantile(0.25)  # NOQA: WPS432
    q3 = data.quantile(0.75)  # NOQA: WPS432
    iqr = q3 - q1
    range_min = q1 - 1.5 * iqr  # NOQA: WPS432
    range_max = q3 + 1.5 * iqr  # NOQA: WPS432
    if within_observed_range:
        range_min = max(range_min, data.min())
        range_max = min(range_max, data.max())
    return range_min, range_max


def _calculate_three_sigma_whisker(
    data: pd.Series,
    within_observed_range: bool = True,
) -> tp.Tuple[float, float]:
    """
    Calculate the whisker value based on 3-sigma rule
    https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule

    Args:
        data: input data
        within_observed_range: if True, then the calculated 3-sigma whisker will be
         within whether the observed data range. This is useful when the data is
         not normally distributed and the whisker value is outside the observed.

    Returns:
        tuple: whisker value
    """
    mean = data.mean()
    std = data.std()
    range_min = mean - 3 * std
    range_max = mean + 3 * std
    if within_observed_range:
        range_min = max(range_min, data.min())
        range_max = min(range_max, data.max())
    return range_min, range_max


def _check_are_known(
    outliers_config: TagsConfig[TagOutliersParameters],
    tags: tp.Iterable[str],
) -> None:
    """
    Check if tag is known

    Raises:
        KeyError: if some tag is missing
    """
    missing_tags = set(tags) - set(outliers_config)
    if missing_tags:
        raise KeyError(
            f"Following tags were not found in tag dictionary: {missing_tags}",
        )
