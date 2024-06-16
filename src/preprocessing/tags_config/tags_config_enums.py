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

from enum import Enum


class DataType(str, Enum):  # noqa: WPS600
    """
    Enumeration of data types for tags.
    """
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    NUMERIC_COERCE = "numeric_coerce"


class TagType(str, Enum):  # noqa: WPS600
    """
    Enumeration of tag types.
    """
    INPUT = "input"
    OUTPUT = "output"
    STATE = "state"
    CONTROL = "control"
    ON_OFF = "on_off"


class OutliersRule(str, Enum):  # noqa: WPS600
    """
    Enumeration of rules for handling outliers.
    """
    DROP = "drop"
    CLIP = "clip"


class ImputationRule(str, Enum):  # noqa: WPS600
    """
    Enumeration of imputation rules.
    """
    LINEAR = "linear"
    FFILL = "ffill"


class ResampleMethod(str, Enum):  # noqa: WPS600
    """
    Enumeration of resampling methods.
    """
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    FIRST = "first"
    SUM = "sum"
    LAST = "last"
    MEDIAN = "median"
