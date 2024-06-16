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
import os
from pathlib import Path

import pytest
import yaml
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline, node
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import CSVDataset

from feature_factory import create_features


@pytest.fixture
def kedro_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                create_features,
                inputs={
                    "params": "params:feature_factory",
                    "data": "input_data",
                },
                outputs="post_create_features",
                name="create_features",
            ),
        ],
    )


def test_pipeline(kedro_pipeline: Pipeline):
    runner = SequentialRunner()
    params = yaml.safe_load(
        (
            Path(os.path.dirname(__file__))
            / "fixtures"
            / "parameters"
            / "feature_factory.yml"
        ).read_text(),
    )
    data = Path(os.path.dirname(__file__)) / "fixtures" / "data"
    catalog = DataCatalog(
        datasets={"input_data": CSVDataset(filepath=str(data / "sample_data.csv"))},
        feed_dict=params,
    )

    catalog.add_feed_dict({"params:feature_factory": params["feature_factory"]})

    runner.run(kedro_pipeline, catalog)
