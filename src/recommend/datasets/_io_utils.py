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

import json
import pickle
import typing as tp

import pandas as pd
from importlib_resources import files

DATA_DIR = files("recommend.data")


def load_pickle_data(file_name: str) -> tp.Any:
    return pd.read_pickle((DATA_DIR / f"{file_name}.pkl"))


def load_json_data(file_name: str) -> tp.Any:
    with open(DATA_DIR / f"{file_name}.json", 'r') as fp:
        json_data = json.load(fp)
    return json_data


def dump_pickle_data(data: tp.Any, file_name: str) -> None:
    """Save data as pickle"""
    with open(DATA_DIR / f"{file_name}.pkl", "wb") as fw:
        pickle.dump(data, fw)


def dump_json_data(data: tp.Any, file_name: str) -> None:
    """Save data as json"""
    with open(DATA_DIR / f"{file_name}.json", "w") as fw:
        json.dump(data, fw)


def dump_csv_data(data: pd.DataFrame, file_name: str) -> None:
    """Save data as csv"""
    data.to_csv(DATA_DIR / f"{file_name}.csv", index=False)
