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
import typing as tp
from pathlib import Path

import pandas as pd
import yaml

from recommend.cra_export import (
    MetaDataConfig,
    PlantStatusData,
    TagMetaData,
    TargetMetaData,
)
from recommend.cra_export.cra_export import TJson

DATA_DIR = Path(__file__).parents[2] / "data"


def get_sample_input_data() -> pd.DataFrame:
    """Example sample input data """
    return pd.read_csv(DATA_DIR / "01_raw/sample_input_data.csv")


def get_sample_controlled_parameters_config() -> tp.List[tp.Dict[str, tp.Any]]:
    """Sample controlled parameters RAW config"""
    with open(
        DATA_DIR / "03_primary/controlled_parameters.yml", encoding="utf-8",
    ) as config_file:
        config = yaml.load(config_file, yaml.SafeLoader)["controlled_parameters"]
        return tp.cast(tp.List[tp.Dict[str, tp.Any]], config)


def get_sample_targets_meta() -> MetaDataConfig[TargetMetaData]:
    """Sample tags meta data"""
    with open(DATA_DIR / "03_primary/meta_data.yml", encoding="utf-8") as config_file:
        config = yaml.load(config_file, yaml.SafeLoader)["targets_meta"]
        return MetaDataConfig(config, schema=TargetMetaData)


def get_sample_tags_meta() -> MetaDataConfig[TagMetaData]:
    """Sample tags meta data"""
    with open(DATA_DIR / "03_primary/meta_data.yml", encoding="utf-8") as config_file:
        config = yaml.load(config_file, yaml.SafeLoader)["tags_meta"]
        return MetaDataConfig(config, schema=TagMetaData)


def get_sample_plant_info() -> MetaDataConfig[PlantStatusData]:
    """Sample tags meta data"""
    with open(DATA_DIR / "03_primary/meta_data.yml", encoding="utf-8") as config_file:
        config = yaml.load(config_file, yaml.SafeLoader)["plant_info"]
        return MetaDataConfig(config, schema=PlantStatusData)


def get_sample_old_recs_cra() -> TJson:
    """Sample old recommendations sent to CRA"""
    with open(
        DATA_DIR / "07_model_output/sample_old_recommendations_cra.json", 'r',
    ) as fp:
        old_recs = json.load(fp)
    return tp.cast(TJson, old_recs)


def get_sample_old_runs_cra() -> TJson:
    """Sample old recommendations sent to CRA"""
    with open(DATA_DIR / "07_model_output/sample_old_runs_cra.json", 'r') as fp:
        old_runs = json.load(fp)
    return tp.cast(TJson, old_runs)


def get_sample_old_states_cra() -> TJson:
    """Sample old recommendations sent to CRA"""
    with open(DATA_DIR / "07_model_output/sample_old_states_cra.json", 'r') as fp:
        old_states = json.load(fp)
    return tp.cast(TJson, old_states)
