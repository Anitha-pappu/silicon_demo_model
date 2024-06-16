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
This is a SCRIPT for updating some datasets' artefacts.
Use it when the structure of the module changes and pickles stop working.
"""

import logging
import typing as tp

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from reporting.datasets import CustomModelWrapper, ShapValues, get_master_table
from reporting.datasets.calculate_shap_feature_impoortances import (
    get_shap_explanation,
)
from reporting.datasets.io_utils import (
    DATA_DIR,
    FILE_ENCODING,
    dump_to_csv,
    dump_to_pickle,
)

try:
    from pip.operations import freeze
except ImportError:
    from pip._internal.operations import freeze  # noqa: WPS436, WPS440


_DEFAULT_RANDOM_SEED = 42
_PIP_FREEZE_PATH = "_pip_freeze.txt"

logger = logging.getLogger(__name__)


def _create_datasets(
    train_data: pd.DataFrame, test_data: pd.DataFrame,
) -> tp.Iterable[tp.Tuple[str, pd.DataFrame]]:
    return ("train_data", train_data), ("test_data", test_data)


def _create_model() -> tp.Tuple[pd.DataFrame, pd.DataFrame, CustomModelWrapper]:
    data = get_master_table()
    train_data, test_data = train_test_split(
        data, random_state=_DEFAULT_RANDOM_SEED, shuffle=False,
    )
    features = [
        "inp_quantity",
        "cu_content",
        "inp_avg_hardness",
        "on_off_mill_a",
        "mill_b_power",
        "mill_a_power",
        "mill_b_load",
        "mill_a_load",
        "dummy_feature_tag",
    ]
    model = CustomModelWrapper(
        imputer=SimpleImputer(),
        estimator=RandomForestRegressor(random_state=_DEFAULT_RANDOM_SEED),
        target="outp_quantity",
        features_in=features,
    ).fit(train_data)
    explanation = get_shap_explanation(model, train_data[features])
    model.shap_values = ShapValues(
        data=explanation.data,
        values=explanation.values,
        base_values=explanation.base_values,
        feature_names=explanation.feature_names,
    )
    return train_data, test_data, model


def _update_modeling_artifacts():
    train_data, test_data, model = _create_model()
    model_data_dir = "model_results_mock_data"
    for file_name, dataset in _create_datasets(train_data, test_data):
        dump_to_csv(dataset, model_data_dir, file_name, index=False)
    dump_to_pickle(model, directory=model_data_dir, file_name="model")


def _create_environment_tracking_file() -> None:
    installed_packages = list(freeze.freeze())
    with open(DATA_DIR / _PIP_FREEZE_PATH, "w", encoding=FILE_ENCODING) as fw:
        for package in installed_packages:
            fw.write(f"{package}\n")


def main() -> None:
    _update_modeling_artifacts()
    _create_environment_tracking_file()


if __name__ == "__main__":
    main()
