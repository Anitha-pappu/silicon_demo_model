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
import re
import typing as tp

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from preprocessing.tags_config.tags_config_enums import (
    DataType,
    ImputationRule,
    OutliersRule,
    ResampleMethod,
    TagType,
)

ILLEGAL_TAG_PATTERNS = (
    ("^.*,+.*$", "no commas in tag"),
    (r"^\s.*$", "tag must not start with whitespace character"),
    (r"^.*\s$", "tag must not end with whitespace character"),
)


class BaseTagParameters(BaseModel):
    """
    Base model for tag parameters with common validation.

    Attributes:
        tag_name: The name of the tag.
    """
    tag_name: str

    @field_validator('tag_name')
    def check_illegal_patterns(cls, tag_name: str) -> str:  # noqa: N805
        """
        Validate that the tag name does not contain illegal patterns.
        """
        for pattern, rule in ILLEGAL_TAG_PATTERNS:
            if re.match(pattern, tag_name):
                raise ValueError(f'tag does not adhere to rule `{rule}`')
        return tag_name


class TagRawParameters(BaseTagParameters):
    """
    Model for raw tag parameters.

    Attributes:
        raw_tag: The raw tag identifier.
        description: A description of the tag.
        display_name: A human-readable name for the tag.
    """
    raw_tag: str
    description: str
    display_name: str


class TagMetaParameters(BaseTagParameters):
    """
    Model for metadata parameters of a tag.

    Attributes:
        data_source: The source of the data.
        data_type: The type of data (e.g., numeric, categorical).
        tag_type: The type of tag (e.g., input, output).
        unit: The unit of measurement.
        min: The minimum valid value.
        max: The maximum valid value.
        extract_freq: The frequency of data extraction.
    """
    data_source: str | None = None
    data_type: DataType
    tag_type: TagType
    unit: str | None = None
    min: float | None = Field(allow_inf_nan=True, default=None)
    max: float | None = Field(allow_inf_nan=True, default=None)
    extract_freq: str | None = None

    @field_validator('data_source', 'unit', 'extract_freq', mode="before")
    def change_nan_to_none(cls, str_value: tp.Any) -> tp.Any:  # noqa: N805
        """Converts NaN to None for pandas compatibility."""
        if isinstance(str_value, float) and pd.isna(str_value):
            return None
        return str(str_value)


class TagOutliersParameters(BaseTagParameters):
    """
    Model for outlier handling parameters of a tag.

    Attributes:
        range_min: The minimum range value for outlier detection.
        range_max: The maximum range value for outlier detection.
        special_values: A string representing special values.
        outlier_rules: The rule for handling outliers.
    """
    range_min: float = Field(allow_inf_nan=True)
    range_max: float = Field(allow_inf_nan=True)
    special_values: str | None = None
    outlier_rules: OutliersRule | None = None

    @field_validator('special_values', 'outlier_rules', mode="before")
    def change_nan_to_none(cls, str_value: tp.Any) -> tp.Any:  # noqa: N805
        """Converts NaN to None for pandas compatibility."""
        if isinstance(str_value, float) and pd.isna(str_value):
            return None
        return str(str_value)


class TagImputationParameters(BaseTagParameters):
    """
    Model for imputation parameters of a tag.

    Attributes:
        imputation_rule: The rule for imputing missing values.
    """
    imputation_rule: ImputationRule | None

    @field_validator('imputation_rule', mode="before")
    def change_nan_to_none(cls, str_value: tp.Any) -> tp.Any:  # noqa: N805
        """Converts NaN to None for pandas compatibility."""
        if isinstance(str_value, float) and pd.isna(str_value):
            return None
        return str(str_value)


class TagOnOffDependencyParameters(BaseTagParameters):
    """
    Model for on/off dependency parameters of a tag.

    Attributes:
        on_off_dependencies: A list of tags that the current tag depends on.
    """
    on_off_dependencies: list[str] = []

    @field_validator(
        'on_off_dependencies',
        mode="before",
        check_fields=False,
    )
    def parse_on_off_dependencies(cls, on_off_value: tp.Any) -> list[str]:  # noqa: N805
        """
        Parse and validate on/off dependencies from a comma-separated string.
        """
        if isinstance(on_off_value, list):
            return on_off_value
        if isinstance(on_off_value, str):
            return list(filter(None, [tag.strip() for tag in on_off_value.split(',')]))
        return []


class TagResampleParameters(BaseTagParameters):
    """
    Model for resampling parameters of a tag.

    Attributes:
        resample_method: The method used to resample data (e.g., mean, sum).
        resample_freq: The frequency at which data should be resampled.
        resample_offset: The offset used for resampling time series data.
    """
    resample_method: ResampleMethod | None = None
    resample_freq: str | None = None
    resample_offset: str | None = None

    @field_validator(
        'resample_method', 'resample_freq', 'resample_offset', mode="before",
    )
    def change_nan_to_none(cls, str_value: tp.Any) -> tp.Any:  # noqa: N805
        """Converts NaN to None for pandas compatibility."""
        if isinstance(str_value, float) and pd.isna(str_value):
            return None
        return str(str_value)
