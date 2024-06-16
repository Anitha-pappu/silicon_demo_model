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
import warnings
from pathlib import Path

import yaml
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import CSVDataset
from keras_tuner import HyperParameters
from sklearn.exceptions import ConvergenceWarning
from tensorflow import keras

from modeling.kedro_utils import KerasModelDataset
from modeling.models.keras_model import KerasModelFactory, KerasModelTuner
from optimus_core.kedro_utils import TagDictCSVLocalDataset


class UserKerasModelFactory(KerasModelFactory):
    @staticmethod
    def create_model_instance(
        units: int = 32, learning_rate: float = 1e-3,
    ) -> keras.Model:
        model = keras.Sequential(
            [
                keras.layers.Normalization(axis=-1),
                keras.layers.Dense(units=units, activation="tanh"),
                keras.layers.Dense(units=1),
            ],
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=[
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.MeanSquaredError(),
                keras.metrics.MeanAbsolutePercentageError(),
            ],
        )
        return model


class UserKerasModelTuner(KerasModelTuner):
    @staticmethod
    def _create_trial_hyperparameters(
        hp: HyperParameters,
        model_init_config: tp.Dict[str, tp.Any],
        model_hyperparameters_config: tp.Dict[str, tp.Any],
    ) -> tp.Dict[str, tp.Any]:
        units = hp.Int(
            "units",
            min_value=model_hyperparameters_config["units"]["min_value"],
            max_value=model_hyperparameters_config["units"]["max_value"],
            sampling=model_hyperparameters_config["units"]["sampling"],
        )
        learning_rate = model_init_config["learning_rate"]
        return {"units": units, "learning_rate": learning_rate}


def test_pipeline(tmp_path, kedro_pipeline):
    here = Path(__file__)
    data = (here / "../../data").resolve()
    params = (here / "../parameters.yml").resolve()
    params = yaml.safe_load(params.read_text())
    params["train"]["factory_class_name"] = UserKerasModelFactory
    params["tune"]["class_name"] = UserKerasModelTuner
    catalog = DataCatalog(
        datasets={
            "data": CSVDataset(filepath=str(data / "sample_model_input_data.csv")),
            "td": TagDictCSVLocalDataset(str(data / "sample_tag_dict.csv")),
            "trained_model": KerasModelDataset(
                filepath=str(data / "sample_trained_model"),
            ),
            "tuned_model": KerasModelDataset(filepath=str(data / "sample_tuned_model")),
            "model": KerasModelDataset(filepath=str(data / "sample_model")),
            "test_data": CSVDataset(
                filepath=str(data / "sample_recommend_input_data.csv"),
            ),
        },
        feed_dict=params,
    )

    def update_feed_dict(p, parent):  # noqa: WPS430
        for k, v in p.items():
            catalog.add_feed_dict({f"{parent}.{k}": v})
            if isinstance(v, dict):
                update_feed_dict(v, f"{parent}.{k}")

    update_feed_dict(params["train"], "params:train")
    update_feed_dict(params["tune"], "params:tune")
    update_feed_dict(params["split"], "params:split")
    catalog.add_feed_dict({"params:tune": params["tune"]})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        SequentialRunner().run(kedro_pipeline, catalog)
