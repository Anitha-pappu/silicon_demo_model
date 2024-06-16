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
import yaml
from sklearn.pipeline import Pipeline

from recommend.controlled_parameters import ControlledParametersConfig
from recommend.cra_export import (
    MetaDataConfig,
    PlantStatusData,
    TagMetaData,
    TargetMetaData,
)
from recommend.solution import Solutions

from .._data_prep_utils import (  # noqa: WPS436
    add_mock_penalties,
    mask_values_randomly,
)
from .._io_utils import DATA_DIR, load_pickle_data  # noqa: WPS436

_N_READIGS_TO_MASK = 100


def get_sample_recommend_input_data() -> pd.DataFrame:
    """Sample input data for recommend package"""
    df = pd.read_csv(
        DATA_DIR / "sample_recommend_input_data.csv", parse_dates=["timestamp"],
    )
    # some missing data is required to demonstrate some functionality
    # in the solutions' overview report
    mask_values_randomly(df, n_readings_to_mask=_N_READIGS_TO_MASK)
    return df


def get_sample_solutions(include_mock_penalties: bool = False) -> Solutions:
    """
    Sample recommend results

    Args:
        include_mock_penalties: add two mock random penalties
            ("first_random" and "second_random") to each solution.
            This is required for a reporting constraints section demonstration
    """

    solutions = tp.cast(Solutions, load_pickle_data("sample_solutions"))
    if include_mock_penalties:
        add_mock_penalties(solutions)
    return solutions


def get_sample_controlled_parameters_raw_config() -> tp.List[tp.Dict[str, tp.Any]]:
    """Sample controlled parameters RAW config"""
    with open(
        DATA_DIR / "controlled_parameters.yml", encoding="utf-8",
    ) as config_file:
        config = yaml.load(config_file, yaml.SafeLoader)["controlled_parameters"]
        return tp.cast(tp.List[tp.Dict[str, tp.Any]], config)


def get_sample_controlled_parameters_config() -> ControlledParametersConfig:
    """Sample controlled parameters config"""
    return ControlledParametersConfig(get_sample_controlled_parameters_raw_config())


def get_sample_tags_meta() -> MetaDataConfig[TagMetaData]:
    """Sample tags meta data"""
    with open(DATA_DIR / "meta_data.yml", encoding="utf-8") as config_file:
        config = yaml.load(config_file, yaml.SafeLoader)["tags_meta"]
        return MetaDataConfig(config, schema=TagMetaData)


def get_sample_targets_meta() -> MetaDataConfig[TargetMetaData]:
    """Sample tags meta data"""
    with open(DATA_DIR / "meta_data.yml", encoding="utf-8") as config_file:
        config = yaml.load(config_file, yaml.SafeLoader)["targets_meta"]
        return MetaDataConfig(config, schema=TargetMetaData)


def get_sample_plant_info() -> MetaDataConfig[PlantStatusData]:
    """Sample tags meta data"""
    with open(DATA_DIR / "meta_data.yml", encoding="utf-8") as config_file:
        config = yaml.load(config_file, yaml.SafeLoader)["plant_info"]
        return MetaDataConfig(config, schema=PlantStatusData)


def get_trained_model() -> Pipeline:
    """Example sample trained model to run optimization with"""
    return tp.cast(Pipeline, load_pickle_data("sample_trained_model"))
