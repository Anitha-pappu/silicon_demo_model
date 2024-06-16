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

import warnings
from pathlib import Path

import yaml
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import CSVDataset
from kedro_datasets.pickle import PickleDataset
from sklearn.exceptions import ConvergenceWarning

from optimus_core.kedro_utils import TagDictCSVLocalDataset


def test_pipeline(tmp_path, kedro_pipeline):
    here = Path(__file__)
    data = here.parents[1] / "data"
    params = here.parents[1] / "sklearn_pipeline/parameters.yml"

    params = yaml.safe_load(params.read_text())

    params["tune"]["tuner"]["kwargs"]["n_jobs"] = 1

    catalog = DataCatalog(
        datasets={
            "data": CSVDataset(filepath=str(data / "sample_model_input_data.csv")),
            "td": TagDictCSVLocalDataset(str(data / "sample_tag_dict.csv")),
            "trained_model": PickleDataset(
                filepath=str(data / "sample_trained_model.pkl"),
            ),
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
