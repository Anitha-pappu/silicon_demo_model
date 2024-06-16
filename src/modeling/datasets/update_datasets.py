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
import pickle
import typing as tp

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from modeling import SklearnPipeline
from modeling.datasets.datasets import DATA_DIR, get_sample_model_input_data

_DEFAULT_RANDOM_SEED = 123
_DEPTH = 7
_AFTER_RECS_SIZE = 81
_MIN_UPLIFT = -2.2
_MAX_UPLIFT = 0.5

logger = logging.getLogger(__name__)


def main() -> None:
    data = get_sample_model_input_data()
    model = _create_model(data)
    _create_baseline_datasets(data)
    _create_shap_explanation(data, model)
    logger.info("Artifacts updated successfully.")


def _create_shap_explanation(data: pd.DataFrame, model: SklearnPipeline) -> None:
    """Creates a sample shap explanation for the trained model."""
    shap_explanation = model.produce_shap_explanation(data)
    _set_pickle_data(shap_explanation, file_name="sample_shap_explanation")


def _create_model(train_data: pd.DataFrame) -> SklearnPipeline:
    """Creates a sample trained model"""
    features = [
        'iron_feed',
        'silica_feed',
        'starch_flow',
        'amina_flow',
        'ore_pulp_flow',
        'ore_pulp_ph',
        'ore_pulp_density',
        'total_air_flow',
        'total_column_level',
        'feed_diff_divide_silica',
    ]
    target = "silica_conc"
    train_data = train_data.dropna(subset=[*features, target])
    model = SklearnPipeline(
        make_pipeline(
            SimpleImputer(),
            RandomForestRegressor(max_depth=_DEPTH, random_state=_DEFAULT_RANDOM_SEED),
        ).set_output(transform="pandas"),
        features,
        target,
    ).fit(train_data)
    _set_pickle_data(model, file_name="sample_trained_model")
    return model


def _create_baseline_datasets(sample_model_input: pd.DataFrame) -> None:
    """
    Creates historical data to train a baseline model and simulated data after OAI
    recommendations. Both datasets are retrieved from splitting sample model input data.
    Data after OAI recommendations has the target value (``silica_conc``) increased
    by a uniform distribution between ``_MIN_UPLIFT`` and ``_MAX_UPLIFT`` to simulate
    the impact of recommendations. In a real scenario, controls would have also been
    modified by recommendations. However, as they are not used for baselining, it is not
    needed to perform this change.
    """
    sample_model_input = sample_model_input.sort_values("timestamp")
    baseline_data = sample_model_input.iloc[:-_AFTER_RECS_SIZE].copy()
    baseline_data = baseline_data.dropna(
        subset=["iron_feed", "silica_feed", "feed_diff_divide_silica", "silica_conc"],
    )

    after_recs_data = sample_model_input.iloc[-_AFTER_RECS_SIZE:].copy()
    rand_gen = np.random.default_rng(_DEFAULT_RANDOM_SEED)
    random_increase = (
        rand_gen.uniform(
            _MIN_UPLIFT,
            _MAX_UPLIFT,
            len(after_recs_data),
        )
    )
    after_recs_data["silica_conc"] = after_recs_data["silica_conc"] + random_increase
    after_recs_data["timestamp"] = pd.to_datetime(after_recs_data["timestamp"])
    _set_csv_data(baseline_data, file_name="sample_baselining_historical_data")
    _set_csv_data(after_recs_data, file_name="sample_baselining_recs_data")


def _set_pickle_data(data: tp.Any, file_name: str) -> None:
    """Save data as pickle"""
    dump_path = str(DATA_DIR / f"{file_name}.pkl")
    with open(dump_path, "wb") as fw:
        pickle.dump(data, fw)


def _set_csv_data(data: pd.DataFrame, file_name: str) -> None:
    """Save data as csv"""
    data.to_csv(DATA_DIR / f"{file_name}.csv", index=False)


if __name__ == "__main__":
    main()
