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

import os
import warnings
from distutils.util import strtobool

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from ..solution import Solutions
from .meta_data import MetaDataConfig


class NonBooleanEnvVariableError(Exception):
    """Error raised when provided value in env var is not boolean-like."""


def parse_boolean_env_var(env_var_name: str) -> bool:
    env_var_value = os.environ.get(env_var_name, "false")
    try:
        return bool(strtobool(env_var_value.lower()))
    except ValueError:
        raise NonBooleanEnvVariableError(
            f"The value {env_var_value} for "
            f"env var {env_var_name} is not boolean-like.",
        )


def get_timestamp_in_iso_format(
    timestamp_column: pd.Series, iso_format: str,
) -> pd.Series:
    if is_datetime64_any_dtype(timestamp_column):
        timestamp_column_dt = timestamp_column.copy()
    else:
        warnings.warn("Automatically parsing timestamp using `pd.to_datetime`")
        timestamp_column_dt = pd.to_datetime(timestamp_column)

    return timestamp_column_dt.dt.strftime(iso_format).astype(str)


def get_id_mapping(
    meta_config: MetaDataConfig,  # type: ignore
) -> dict[str, str]:
    return {conf.tag: conf.id for conf in meta_config}


def get_run_id(
    solutions: Solutions,
    timestamp_column: str,
    iso_format: str,
) -> dict[str, str]:
    solutions_df = solutions.to_frame()
    solutions_df[(timestamp_column, "initial")] = get_timestamp_in_iso_format(
        solutions_df[(timestamp_column, "initial")], iso_format,
    )
    return {
        row[timestamp_column].iloc[0]: row["run_id"].iloc[0]
        for _, row in solutions_df.iterrows()
    }


def parse_timestamp(
    iso_format: str,
    timestamp: np.datetime64 | pd.Timestamp | str,
) -> str:
    parsed_timestamp: pd.Timestamp
    if is_datetime64_any_dtype(timestamp) or isinstance(timestamp, pd.Timestamp):
        parsed_timestamp = timestamp
    else:
        warnings.warn("Automatically parsing timestamp using `pd.to_datetime`")
        parsed_timestamp = pd.to_datetime(timestamp)
    return str(parsed_timestamp.strftime(iso_format))
