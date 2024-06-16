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

import pandas as pd

from reporting.datasets.io_utils import load_csv


def get_batch_meta_with_features() -> pd.DataFrame:
    """Sample dataset batch analytics. Batch meta with features for batch-level.

    Returns:
        A `pandas.DataFrame` with 100 rows (100 batches) and the following columns:
        `['reactor_start_time', 'reactor_end_time', 'filter_start_time',
        'filter_end_time', 'reactor', 'filter']`
    """
    return load_csv(
        directory="batch_mock_data", file_name="batch_meta_with_features", index_col=0,
    )


def get_sensor_data_batched_phased() -> pd.DataFrame:
    """Sample dataset batch analytics. Batch sensors data labeled by batches and phases.

    Returns:
         A `pandas.DataFrame` with 13304 rows and the following columns:
        `['batch_id', 'time_step', 'datetime', 'filter_infeed', 'filter_trough_lvl',
          'reactor_P', 'reactor_acid_total', 'reactor_agitator_speed', 'reactor_temp']`.
    """
    return load_csv(
        directory="batch_mock_data",
        file_name="sensor_data_batched_phased",
        parse_dates=["datetime"],
    )


def get_mill_data() -> pd.DataFrame:
    """Sample dataset with mining data. Mill data."""
    return load_csv(directory="mining_mock_data", file_name="mill_historic")


def get_throughput_data() -> pd.DataFrame:
    """Sample dataset with mining data. Input & output sensors."""
    return load_csv(directory="mining_mock_data", file_name="in_out_historic")
