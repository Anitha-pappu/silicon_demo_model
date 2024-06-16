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

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from catboost import CatBoostRegressor
from importlib_resources import files
from kedro.io.core import generate_timestamp
from kedro.pipeline import Pipeline, node
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from modeling import (
    SklearnModel,
    api,
    calculate_feature_importance,
    calculate_metrics,
    calculate_model_predictions,
    calculate_shap_feature_importance,
    create_model,
    create_model_factory_from_tag_dict,
    create_splitter,
    create_tuner,
    drop_nan_rows,
    split_data,
    train_model,
    tune_model,
)
from optimus_core import SklearnTransform, TagDict
from optimus_core.tag_dict.validation import REQUIRED_COLUMNS

random = np.random.RandomState(42)

DATA_DIR = files("tests.data")

N_FEATURES = 10
N_SAMPLES = 250
N_INFORMATIVE = 3
N_NOISE = 5


@pytest.fixture
def simple_data():
    path = DATA_DIR / "simple_data.csv"
    return pd.read_csv(path, parse_dates=["date"])


@pytest.fixture
def regression_data():
    regression_data, regression_target = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        random_state=0,
        noise=2.0,
    )
    regression_df = pd.DataFrame(
        regression_data, columns=[f"Column_{i + 1}" for i in range(N_FEATURES)],
    )
    target_df = pd.DataFrame(regression_target, columns=["Target"])
    noised_data = np.random.randn(N_SAMPLES, N_NOISE)
    noised_df = pd.DataFrame(
        noised_data, columns=[f"Useless_feature_{i + 1}" for i in range(N_NOISE)],
    )
    return pd.concat([regression_df, noised_df, target_df], axis=1)


@pytest.fixture(scope="session")
def simple_data_tag_dict():
    td_data = pd.DataFrame(
        {
            "tag": [
                "% Iron Feed",
                "% Silica Feed",
                "Starch Flow",
                "Amina Flow",
                "% Iron Concentrate",
                "% Silica Concentrate",
            ],
            "name": [
                "Iron Feed",
                "Silica Feed",
                "Starch Flow",
                "Amina Flow",
                "Conc Iron",
                "Conc Silica",
            ],
            "tag_type": ["state", "state", "control", "control", "control", "state"],
            "silica_model_feature": [True, True, False, False, True, False],
            "no_controls_model": [True, True, False, False, False, True],
        },
    )

    for column in REQUIRED_COLUMNS:
        if column not in td_data.columns:
            td_data[column] = None

    return TagDict(td_data)


@pytest.fixture
def estimator():
    return LinearRegression()


@pytest.fixture(
    params=[
        ElasticNet(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
        GradientBoostingRegressor(),
        XGBRegressor(),
        LinearRegression(),
        RandomForestRegressor(),
        KernelRidge(),
        AdaBoostRegressor(),
        BaggingRegressor(),
        Ridge(),
        Lasso(),
        ElasticNet(),
        SGDRegressor(),
        ExtraTreesRegressor(),
        AdaBoostRegressor(),
        BaggingRegressor(),
        MLPRegressor(),
        GaussianProcessRegressor(),
        RANSACRegressor(),
        HuberRegressor(),
        OrthogonalMatchingPursuit(),
        BayesianRidge(),
        PassiveAggressiveRegressor(),
        ARDRegression(),
        TweedieRegressor(),
        SVR(),
        # Prohibiting a CatBoost model to generate files and excess logs
        CatBoostRegressor(silent=True, allow_writing_files=False),
        LGBMRegressor(),
    ],
)
def estimator_to_test(request: FixtureRequest) -> api.Estimator:
    return request.param


@pytest.fixture
def silica_model_features():
    return ["% Iron Feed", "% Silica Feed", "% Iron Concentrate"]


@pytest.fixture
def trained_sklearn_model(simple_data, silica_model_features) -> SklearnModel:
    model = RandomForestRegressor()
    return SklearnModel(
        estimator=model,
        features_in=silica_model_features,
        target="% Silica Concentrate",
    ).fit(simple_data)


@pytest.fixture
def mixed_transformers():
    return [
        ("min_max", SklearnTransform(MinMaxScaler(feature_range=(0, 100)))),
        {
            "class_name": "sklearn.feature_selection.SelectKBest",
            "kwargs": {"k": 15, "score_func": "sklearn.feature_selection.chi2"},
            "name": "select_best_15",
            "wrapper": "preserve_columns",
        },
    ]


@pytest.fixture
def kedro_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                drop_nan_rows,
                inputs={
                    "data": "data",
                    "td": "td",
                    "td_features_column": "params:train.td_features_column",
                    "target_column": "params:train.target_column",
                },
                outputs="data_dropna",
                name="drop_nan_rows",
            ),
            node(
                create_splitter,
                inputs={
                    "split_method": "params:split.split_method",
                    "splitting_parameters": "params:split.split_parameters",
                },
                outputs="splitter",
                name="create_splitter",
            ),
            node(
                split_data,
                inputs={
                    "data": "data_dropna",
                    "splitter": "splitter",
                },
                outputs=["train_data", "test_data"],
                name="split_data",
            ),
            node(
                create_model_factory_from_tag_dict,
                inputs={
                    "model_factory_type": "params:train.factory_class_name",
                    "model_init_config": "params:train.init",
                    "tag_dict": "td",
                    "target": "params:train.target_column",
                    "tag_dict_features_column": "params:train.td_features_column",
                },
                outputs="model_factory",
            ),
            node(
                create_model,
                inputs={
                    "model_factory": "model_factory",
                },
                outputs="model",
            ),
            node(
                create_tuner,
                inputs={
                    "model_factory": "model_factory",
                    "model_tuner_type": "params:tune.class_name",
                    "tuner_config": "params:tune.tuner",
                },
                outputs="model_tuner",
            ),
            node(
                tune_model,
                inputs={
                    "model_tuner": "model_tuner",
                    "hyperparameters_config": "params:tune.hyperparameters",
                    "data": "train_data",
                },
                outputs="tuned_model",
            ),
            node(
                train_model,
                inputs={
                    "model": "tuned_model", "data": "train_data",
                },
                outputs="trained_model",
                name="train_model",
            ),
            node(
                calculate_model_predictions,
                inputs={
                    "data": "train_data",
                    "model": "trained_model",
                },
                outputs="train_data_predictions",
                name="train_predict",
            ),
            node(
                calculate_model_predictions,
                inputs={
                    "data": "test_data",
                    "model": "trained_model",
                },
                outputs="test_data_predictions",
                name="test_predict",
            ),
            node(
                calculate_metrics,
                inputs={
                    "data": "train_data",
                    "model": "trained_model",
                },
                outputs="train_metrics",
                name="create_train_metrics",
            ),
            node(
                calculate_metrics,
                inputs={
                    "data": "test_data",
                    "model": "trained_model",
                },
                outputs="test_metrics",
                name="create_test_metrics",
            ),
            node(
                calculate_feature_importance,
                inputs={
                    "data": "train_data",
                    "model": "trained_model",
                },
                outputs="feature_importance",
                name="feature_importance",
            ),
            node(
                calculate_shap_feature_importance,
                inputs={
                    "data": "train_data",
                    "shap_producer": "trained_model",
                },
                outputs="shap_feature_importance",
                name="shap_feature_importance",
            ),
        ],
    )


@pytest.fixture(params=[None])
def load_version(request):
    return request.param


@pytest.fixture(params=[None])
def save_version(request):
    return request.param or generate_timestamp()


@pytest.fixture(params=[None])
def load_args(request):
    return request.param


@pytest.fixture(params=[None])
def save_args(request):
    return request.param


@pytest.fixture(params=[None])
def fs_args(request):
    return request.param


@pytest.fixture(scope="session")
def data_timeseries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range(
                start="2020-02-01", end="2020-02-02", freq="2.5H",
            ),
            "sensor1": random.normal(100, 20, size=10),
            "sensor2": random.normal(50, 10, size=10),
            "sensor3": random.normal(10, 5, size=10),
        },
    )


@pytest.fixture(scope="session")
def weekly_data():
    """
    Creates a dataframe with weekly data that can be used for testing some
    common cases.

    Originally intended to test the baseline models.
    This dataframe contains the following columns:
    - a timestamp with weekly frequence
    - a constant series
    - a series of values increasing by 1
    - a constant series with a block of missing values in the middle
    - a constant series starting with a block of missing values
    - a constant series with every 10th point missing
    """
    timestamp = (
        pd.date_range(start="2020-01-01", end="2023-01-01", freq="7D", inclusive="left")
        .to_series()
        .reset_index(drop=True)
    )
    index = timestamp.index
    constant_ser = pd.Series([1.0 for i in timestamp], index=index)
    increasing = pd.Series(list(index), index=index).astype(float)
    const_with_long_missing_inside = constant_ser.where(
        ~timestamp.dt.month.isin([4, 5, 6, 7, 8, 9]),
    )
    const_with_long_missing_at_beginning = constant_ser.where(
        ~(timestamp < pd.to_datetime("2020-07-01")), np.NaN,
    )
    const_with_sparsely_missing = constant_ser.where(index % 10 != 0, np.NaN)
    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "constant": constant_ser,
            "increasing": increasing,
            "long_missing_inside": const_with_long_missing_inside,
            "long_missing_at_beginning": const_with_long_missing_at_beginning,
            "sparsely_missing": const_with_sparsely_missing,
        },
    )
    return df
