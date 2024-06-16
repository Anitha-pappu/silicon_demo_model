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
Tests the export nodes
"""
import logging

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import TypeAdapter
from sklearn.linear_model import LinearRegression

from preprocessing.imputing import ModelBasedImputer, interpolate_cols
from preprocessing.tags_config import TagImputationParameters, TagsConfig


class TestInterpolateCols(object):

    @pytest.fixture()
    def input_data(self):
        """Mock input dataframe"""
        inp = {
            "inp_avg_hardness": [0.1, 0.2, 0.3, np.nan, np.nan, np.nan, 0.8],
            "inp_quantity": [1, 2, 3, np.nan, np.nan, np.nan, 12],
            "cu_content": [0.06, 0.05, 0.03, np.nan, np.nan, np.nan, 0.12],
        }

        df = pd.DataFrame(data=inp)
        return df

    @pytest.fixture()
    def cols_list(self):
        """Create simplified version of input parameters"""

        return ["inp_avg_hardness", "inp_quantity", "cu_content"]

    @pytest.fixture()
    def interpolate_kwargs(self):
        """Create simplified version of input parameters"""

        return {"method": "ffill", "limit": 3}

    @pytest.fixture()
    def parameters_all_cols(self):
        """Create simplified version of input parameters"""
        params = {"all_cols": {"method": "ffill", "limit": 3}}
        return params

    @pytest.fixture()
    def interpolate_cols_expected_output(self):
        """Expected output after running through the interpolate cols function"""
        output = pd.DataFrame(
            data={
                "inp_avg_hardness": {
                    0: 0.1,
                    1: 0.2,
                    2: 0.3,
                    3: 0.3,
                    4: 0.3,
                    5: 0.3,
                    6: 0.8,
                },
                "inp_quantity": {
                    0: 1.0,
                    1: 2.0,
                    2: 3.0,
                    3: 3,
                    4: 3,
                    5: 3,
                    6: 12.0,
                },
                "cu_content": {
                    0: 0.06,
                    1: 0.05,
                    2: 0.03,
                    3: 0.03,
                    4: 0.03,
                    5: 0.03,
                    6: 0.12,
                },
            },
        )
        return output

    @pytest.fixture()
    def interpolate_all_cols_expected_output(self):
        inp = {
            "inp_avg_hardness": [0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.8],
            "inp_quantity": [1.0, 2.0, 3.0, 3, 3, 3, 12],
            "cu_content": [0.06, 0.05, 0.03, 0.03, 0.03, 0.03, 0.12],
        }

        df = pd.DataFrame(data=inp)
        return df

    @pytest.fixture
    def tags_imputation_data(self):
        raw_config = [
            {
                "tag_name": "inp_avg_hardness",
                "raw_tag": "inp_avg_hardness",
                "imputation_rule": "ffill",
            },
            {
                "tag_name": "inp_quantity",
                "raw_tag": "inp_quantity",
                "imputation_rule": "ffill",
            },
            {
                "tag_name": "cu_content",
                "raw_tag": "cu_content",
                "imputation_rule": "ffill",
            },
        ]

        return TagsConfig(
            TypeAdapter(list[TagImputationParameters]).validate_python(raw_config),
            TagImputationParameters,
        )

    def test_interpolate_cols(
        self,
        input_data,
        interpolate_cols_expected_output,
        tags_imputation_data,
    ):
        """Test interpolate_cols to ensure we are getting the proper output"""
        result, _ = interpolate_cols(input_data, tags_imputation_data)
        expected = interpolate_cols_expected_output
        assert_frame_equal(result, expected)


class TestValidateInput(object):
    def test_string_in_x(self):
        imputer = ModelBasedImputer("sklearn.ensemble.RandomForestRegressor", "target")
        with pytest.raises(ValueError, match="X must be numeric"):
            imputer.fit(pd.DataFrame([["test"]], columns=["a"]))

    def test_nan_in_x(self):
        imputer = ModelBasedImputer("sklearn.ensemble.RandomForestRegressor", "target")
        with pytest.raises(ValueError, match="X mustn't have NaNs"):
            fit_data = pd.DataFrame([[np.nan, 1]], columns=["a", "target"])
            imputer.fit(fit_data)

    def test_no_y(self):
        imputer = ModelBasedImputer("sklearn.ensemble.RandomForestRegressor", "target")
        with pytest.raises(ValueError, match="Target variable must be in X"):
            imputer.fit(pd.DataFrame([[np.nan]], columns=["a"]))

    def test_no_values_to_predict(self, caplog):
        imputer = ModelBasedImputer("sklearn.ensemble.RandomForestRegressor", "target")
        caplog.set_level(logging.WARNING)
        imputer.fit(pd.DataFrame([[1, 1]], columns=["a", "target"]))
        assert "No missing values to imput" in caplog.text

    def test_models_similarity(self):
        imputer = ModelBasedImputer(
            "sklearn.linear_model.LinearRegression", "C",
        )  # NOQA
        sensitivity_cols = ["A", "B", "C"]
        result_frame = pd.DataFrame(columns=sensitivity_cols)
        count = 1
        for cols in sensitivity_cols[:-1]:
            result_frame[cols] = np.arange(0, 100, 1) + count
            count = count + 1
        result_frame[sensitivity_cols[-1]] = 2 * result_frame["A"] + result_frame["B"]
        for i in range(0, 6):
            result_frame["C"].iloc[i] = np.nan
        x_test = result_frame[result_frame["C"].isnull()]
        imputer.fit(result_frame)
        x_train = result_frame.dropna().reset_index(drop=True)
        regr = LinearRegression()
        regr.fit(x_train[["A", "B"]], x_train["C"])
        assert np.array_equal(
            np.array(imputer.transform(result_frame)["C"][result_frame["C"].isnull()]),
            regr.predict(x_test.drop("C", axis=1)),
        )

    def test_check_overwriting(self):
        imputer = ModelBasedImputer(
            "sklearn.linear_model.LinearRegression",
            "C",
        )  # NOQA
        sensitivity_cols = ["A", "B", "C"]
        result_frame = pd.DataFrame(columns=sensitivity_cols)
        count = 1
        for cols in sensitivity_cols[:-1]:
            result_frame[cols] = np.arange(0, 100, 1) + count
            count = count + 1
        result_frame[sensitivity_cols[-1]] = 2 * result_frame["A"] + result_frame["B"]
        for i in range(0, 12, 2):
            result_frame["C"].iloc[i] = np.nan
        imputer.fit(result_frame)
        np.array_equal(
            np.array(imputer.transform(result_frame)["C"][~result_frame["C"].isnull()]),
            np.array(result_frame["C"].dropna()),
        )
