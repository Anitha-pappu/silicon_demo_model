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
import pytest

from modeling import datasets
from modeling.api import Estimator


class TestDatasets(object):
    @pytest.mark.parametrize(
        "dataset_factory_fn, expected_type",
        [
            (
                datasets.get_trained_estimator,
                Estimator,
            ),
            (
                datasets.get_sample_model_input_data,
                pd.DataFrame,
            ),
            (
                datasets.get_sample_tag_dict,
                object,
            ),
            (
                datasets.get_sample_baselining_historical_data,
                pd.DataFrame,
            ),
            (
                datasets.get_sample_baselining_recs_data,
                pd.DataFrame,
            ),
        ],
    )
    def test_datasets_are_of_correct_class(
        self,
        dataset_factory_fn: tp.Callable[..., tp.Any],
        expected_type: tp.Type[tp.Any],
    ) -> None:
        assert isinstance(dataset_factory_fn(), expected_type)
