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

import pandas as pd

from .._io_utils import DATA_DIR, load_json_data  # noqa: WPS436


def get_sample_implementation_status_input_data() -> pd.DataFrame:
    """Sample input data for implementation status pipeline"""
    return pd.read_csv(
        DATA_DIR / "sample_implementation_status_input_data.csv",
        parse_dates=["timestamp"],
    )


def get_sample_recommendations_cra() -> list[dict[str, tp.Any]]:
    """Sample recommendations sent to CRA"""
    return tp.cast(
        list[dict[str, tp.Any]],
        load_json_data("sample_recommendations_cra"),
    )


def get_sample_states_cra() -> list[dict[str, tp.Any]]:
    """Sample recommendations sent to CRA"""
    return tp.cast(
        list[dict[str, tp.Any]],
        load_json_data("sample_states_cra"),
    )


def get_sample_runs_cra() -> list[dict[str, tp.Any]]:
    """Sample run information sent to CRA"""
    return tp.cast(list[dict[str, tp.Any]], load_json_data("sample_runs_cra"))
