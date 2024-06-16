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

import pickle
import typing as tp

import pandas as pd
from importlib_resources import files
from sklearn.model_selection import train_test_split

from modeling.api import Estimator, ShapExplanation
from modeling.models import SklearnModel, SklearnPipeline
from optimus_core import TagDict

DATA_DIR = files("modeling.data")

_RANDOM_SEED = 42

# todo: remove `sample` prefix


def get_sample_model_input_data() -> pd.DataFrame:
    """Sample data for model input"""
    return pd.read_csv(DATA_DIR / "sample_model_input_data.csv")


def get_model_train_data() -> pd.DataFrame:
    """
    Sample dataset with model results. Data used for training model.
    """
    data = get_sample_model_input_data()
    train, _ = train_test_split(data, random_state=_RANDOM_SEED, shuffle=False)
    return train


def get_model_test_data() -> pd.DataFrame:
    """
    Sample dataset with model results. Data used for testing model.
    """
    data = get_sample_model_input_data()
    _, test = train_test_split(data, random_state=_RANDOM_SEED, shuffle=False)
    return test


def get_sample_tag_dict() -> TagDict:
    """Sample tag dictionary data"""
    return TagDict(pd.read_csv(DATA_DIR / "sample_tag_dict.csv"), validate=False)


def get_trained_model(with_imputer: bool = True) -> SklearnPipeline | SklearnModel:
    """
    Sample trained model to run optimization with.

    Args:
        with_imputer: Adds missing values imputer to the model
    """
    model = tp.cast(SklearnPipeline, _get_pickle_data("sample_trained_model"))
    if with_imputer:
        return model
    return SklearnModel(model.estimator, model.features_in, model.target)


def get_sample_shap_explanation() -> ShapExplanation:
    """Load sample shap explanation for the trained model."""
    return tp.cast(ShapExplanation, _get_pickle_data("sample_shap_explanation"))


def get_trained_estimator() -> Estimator:
    """Sample trained model to run optimization with"""
    return get_trained_model().estimator


def get_sample_baselining_historical_data() -> pd.DataFrame:
    """Example sample historical data for baselining"""
    return pd.read_csv(
        DATA_DIR / "sample_baselining_historical_data.csv", parse_dates=["timestamp"],
    )


def get_sample_baselining_recs_data() -> pd.DataFrame:
    """Example sample data after OAI recommendations for baselining"""
    return pd.read_csv(
        DATA_DIR / "sample_baselining_recs_data.csv", parse_dates=["timestamp"],
    )


def _get_pickle_data(file_name: str) -> tp.Any:
    byte_data = (DATA_DIR / f"{file_name}.pkl").read_bytes()
    return pickle.loads(byte_data)
