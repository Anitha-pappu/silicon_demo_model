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
from sklearn.pipeline import Pipeline

from .._io_utils import DATA_DIR, load_pickle_data  # noqa: WPS436


def get_baseline_trained_model() -> Pipeline:
    """Sample baseline model"""
    return tp.cast(Pipeline, load_pickle_data("sample_trained_baseline_model"))


def get_sample_model_errors() -> pd.DataFrame:
    """Sample baseline model errors"""
    return pd.read_csv(
        DATA_DIR / "sample_baseline_model_errors_data.csv", parse_dates=["timestamp"],
    )


def get_sample_actual_values_after_recs() -> pd.DataFrame:
    """Sample actual values after recommendations"""
    return pd.read_csv(
        DATA_DIR / "sample_actual_value_after_recs_data.csv", parse_dates=["timestamp"],
    )


def get_recs_performance() -> pd.DataFrame:
    """Sample recommendations performance"""
    return pd.read_csv(DATA_DIR / "sample_recommendations_performance_data.csv")
