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

"""Contains utils for IO operations with sample datasets"""

import pickle
import typing as tp

import pandas as pd
from importlib_resources import files

DATA_DIR = files("reporting.data")
FILE_ENCODING = "utf-8"


def dump_to_csv(
    data: pd.DataFrame, directory: str, file_name: str, **kwargs: tp.Any,
) -> None:
    return data.to_csv(
        DATA_DIR / f"{directory}/{file_name}.csv", **kwargs, encoding=FILE_ENCODING,
    )


def dump_to_pickle(data: tp.Any, directory: str, file_name: str) -> None:
    with open(DATA_DIR / f"{directory}/{file_name}.pkl", "wb") as fw:
        pickle.dump(data, fw)


def load_csv(directory: str, file_name: str, **kwargs: tp.Any) -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / f"{directory}/{file_name}.csv", **kwargs, encoding=FILE_ENCODING,
    )


def load_pickle(directory: str, file_name: str) -> tp.Any:
    with open(DATA_DIR / f"{directory}/{file_name}.pkl", "rb") as fr:
        return pickle.load(fr)
