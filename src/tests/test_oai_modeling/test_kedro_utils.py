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


import pathlib
import typing as tp

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from kedro.io import DatasetError, Version
from kedro.io.core import PROTOCOL_DELIMITER
from s3fs.core import S3FileSystem
from typing_extensions import Self

from modeling.kedro_utils import (
    KerasModelDataset,
    convert_metrics_to_nested_mlflow_dict,
)
from modeling.kedro_utils.mlflow_utils import _TKedroMLflowMetrics

_N_COLS_DATASET = 10
_N_FEATURES_IN_MODEL = 5
_N_ROWS_DATASET = 25


TFileSystem = tp.TypeVar("TFileSystem")


class TensorFlowModelWrapper(object):
    """
    A wrapper for TensorFlow model, that imitates
    modeling.load.keras_model.KerasModel for testing purposes.
    """

    def __init__(
        self,
        keras_model: tensorflow.keras.Model,
        target: str,
        features_in: tp.List[str],
    ) -> None:
        self._keras_model = keras_model
        self._features_in = features_in
        self._target = target

    def predict(self, data: pd.DataFrame) -> np.array:
        return self._keras_model.predict(data[self._features_in]).squeeze()

    def fit(self, data) -> Self:
        self._keras_model.fit(
            data[self._features_in], data[self._target],
        )
        return self


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.DataFrame(
        data=np.random.randn(_N_ROWS_DATASET, _N_COLS_DATASET),
        columns=[f"Feature_{i}" for i in range(_N_COLS_DATASET)],
    )


@pytest.fixture(scope="function")
def dir_path(tmp_path: pathlib.Path) -> str:
    return (tmp_path / "test_keras_model").as_posix()


@pytest.fixture(scope="function")
def keras_model_dataset(
    dir_path: str,
    save_args: tp.Dict[str, tp.Any],
    load_args: tp.Dict[str, tp.Any],
    fs_args: tp.Dict[str, tp.Any],
) -> KerasModelDataset:
    return KerasModelDataset(
        filepath=dir_path,
        pickle_dataset_save_args=save_args,
        pickle_dataset_load_args=load_args,
        tensorflow_dataset_save_args=save_args,
        tensorflow_dataset_load_args=load_args,
        fs_args=fs_args,
    )


@pytest.fixture(scope="function")
def versioned_keras_model_dataset(
    dir_path: str,
    save_args: tp.Dict[str, tp.Any],
    load_args: tp.Dict[str, tp.Any],
    fs_args: tp.Dict[str, tp.Any],
    load_version: tp.Optional[str],
    save_version: tp.Optional[str],
) -> KerasModelDataset:
    return KerasModelDataset(
        filepath=dir_path,
        pickle_dataset_save_args=save_args,
        pickle_dataset_load_args=load_args,
        tensorflow_dataset_save_args=save_args,
        tensorflow_dataset_load_args=load_args,
        fs_args=fs_args,
        version=Version(load_version, save_version),
    )


@pytest.fixture(scope="module", params=[16, 32])
def keras_model(request, data: pd.DataFrame) -> TensorFlowModelWrapper:
    keras_model = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.Normalization(axis=-1),
            tensorflow.keras.layers.Dense(units=request.param, activation="tanh"),
            tensorflow.keras.layers.Dense(units=1),
        ],
    )
    keras_model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=[
            tensorflow.keras.metrics.MeanAbsoluteError(),
            tensorflow.keras.metrics.MeanSquaredError(),
            tensorflow.keras.metrics.MeanAbsolutePercentageError(),
        ],
    )
    return TensorFlowModelWrapper(
        keras_model=keras_model,
        features_in=[f"Feature_{i}" for i in range(_N_FEATURES_IN_MODEL)],
        target="Feature_0",
    ).fit(data)


@pytest.fixture(scope="module")
def subclassed_keras_model(data: pd.DataFrame) -> TensorFlowModelWrapper:
    """Demonstrate that own class models cannot be saved
    using HDF5 format but can using TF format
    """
    @keras.saving.register_keras_serializable()
    class MyModel(tensorflow.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tensorflow.keras.layers.Dense(
                _N_FEATURES_IN_MODEL, activation=tensorflow.nn.relu,
            )
            self.dense2 = tensorflow.keras.layers.Dense(
                5, activation=tensorflow.nn.softmax,
            )

        def call(self, inputs, training=None, mask=None):
            x = self.dense1(inputs)
            return self.dense2(x)

    model = MyModel()
    model.compile("rmsprop", "mse")
    return TensorFlowModelWrapper(
        keras_model=model,
        features_in=[f"Feature_{i}" for i in range(_N_FEATURES_IN_MODEL)],
        target="Feature_0",
    ).fit(data)


class TestKerasModelDataset(object):
    def test_keras_model_predicts_same_after_save_and_load(
        self,
        keras_model: TensorFlowModelWrapper,
        keras_model_dataset: KerasModelDataset,
        data: pd.DataFrame,
    ) -> None:
        initial_prediction = keras_model.predict(data)
        keras_model_dataset.save(keras_model)
        keras_model_reloaded = keras_model_dataset.load()
        prediction_after_reload = keras_model_reloaded.predict(data)
        assert np.array_equal(initial_prediction, prediction_after_reload)

    def test_save_into_directory_with_another_model(
        self,
        keras_model: TensorFlowModelWrapper,
        keras_model_dataset: KerasModelDataset,
        data: pd.DataFrame,
    ) -> None:
        initial_prediction = keras_model.predict(data)
        keras_model_dataset.save(keras_model)
        initial_model_reloaded = keras_model_dataset.load()
        keras_model_dataset.save(keras_model.fit(data))
        refitted_model_reloaded = keras_model_dataset.load()
        assert np.array_equal(initial_prediction, initial_model_reloaded.predict(data))
        assert np.array_equal(
            keras_model.predict(data), refitted_model_reloaded.predict(data),
        )
        assert not np.array_equal(
            refitted_model_reloaded.predict(data), initial_prediction,
        )

    def test_hdf5_save_format(
        self, keras_model: TensorFlowModelWrapper, data: pd.DataFrame, dir_path: str,
    ):
        """Test KerasModelDataset can save TF graph models in HDF5 format"""
        keras_model_hdf5_dataset = KerasModelDataset(
            filepath=dir_path, tensorflow_dataset_save_args={"save_format": "h5"},
        )
        keras_model_hdf5_dataset.save(keras_model)
        reloaded_keras_model = keras_model_hdf5_dataset.load()
        np.testing.assert_allclose(
            keras_model.predict(data),
            reloaded_keras_model.predict(data),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_subclass_model_hdf5_save_format(
        self,
        dir_path: str,
        subclassed_keras_model: TensorFlowModelWrapper,
        data: pd.DataFrame,
    ):
        """Test KerasModelDataset cannot save subclassed user models in HDF5 format
        Subclassed model
        From TF docs
        First of all, a subclassed model that has never been used cannot be saved.
        That's because a subclassed model needs to be called on some data in order to
        create its weights.
        """
        keras_model_hdf5_dataset = KerasModelDataset(
            filepath=dir_path, tensorflow_dataset_save_args={"save_format": "h5"},
        )
        subclassed_keras_model.predict(data)
        pattern = (
            "Saving the model to HDF5 format requires the model"
            " to be a Functional model or a"
            " Sequential model. It does not work for subclassed"
            " models, because such models are"
            " defined via the body of a Python method, which"
            r" isn\'t safely serializable. Consider"
            r" saving to the Tensorflow SavedModel format \(by"
            r" setting save_format=\"tf\"\)"
            " or using `save_weights`."
        )
        with pytest.raises(DatasetError, match=pattern):
            keras_model_hdf5_dataset.save(subclassed_keras_model)

    def test_subclass_model(
        self,
        dir_path: str,
        subclassed_keras_model: TensorFlowModelWrapper,
        keras_model_dataset: KerasModelDataset,
        data: pd.DataFrame,
    ):
        """Test KerasModelDataset cannot save subclassed user models in HDF5 format
        Subclassed model
        From TF docs
        First of all, a subclassed model that has never been used cannot be saved.
        That's because a subclassed model needs to be called on some data in order to
        create its weights.
        """
        keras_model_dataset.save(subclassed_keras_model)
        reloaded_subclassed_keras_model = keras_model_dataset.load()
        np.testing.assert_allclose(
            subclassed_keras_model.predict(data),
            reloaded_subclassed_keras_model.predict(data),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_exists(
        self,
        keras_model: TensorFlowModelWrapper,
        keras_model_dataset: KerasModelDataset,
    ) -> None:
        """
        Test `exists` method invocation for both existing and
        nonexistent data set.
        """
        assert not keras_model_dataset.exists()
        keras_model_dataset.save(keras_model)
        assert keras_model_dataset.exists()

    @pytest.mark.parametrize(
        "save_args", [{"k1": "v1", "index": "value"}], indirect=True,
    )
    def test_save_extra_params(
        self, keras_model_dataset: KerasModelDataset, save_args: tp.Dict[str, tp.Any],
    ) -> None:
        """Test overriding the default save arguments."""
        for key, value in save_args.items():
            assert keras_model_dataset._tensorflow_dataset_save_args[key] == value
            assert keras_model_dataset._pickle_dataset_save_args[key] == value

    def test_catalog_release(self, mocker):
        fs_mock = mocker.patch("fsspec.filesystem").return_value
        filepath = "test.tf"
        data_set = KerasModelDataset(filepath=filepath)
        assert data_set._version_cache.currsize == 0  # no cache if unversioned
        data_set.release()
        fs_mock.invalidate_cache.assert_called_once_with(filepath)
        assert data_set._version_cache.currsize == 0

    @pytest.mark.parametrize("fs_args", [{"storage_option": "value"}])
    def test_fs_args(self, fs_args, mocker):
        fs_mock = mocker.patch("fsspec.filesystem")
        KerasModelDataset("test.tf", fs_args=fs_args)
        fs_mock.assert_called_with("file", auto_mkdir=True, storage_option="value")

    def test_load_missing_file(self, keras_model_dataset: KerasModelDataset):
        """Check the error when trying to load missing file."""
        pattern = r"Failed while loading data from data set KerasModelDataset\(.*\)"
        with pytest.raises(DatasetError, match=pattern):
            keras_model_dataset.load()

    @pytest.mark.parametrize(
        "dir_path,instance_type",
        [
            ("s3://bucket/file", S3FileSystem),
            ("file:///tmp/test", LocalFileSystem),
            ("/tmp/test", LocalFileSystem),  # noqa: S108
            ("gcs://bucket/file", GCSFileSystem),
            ("https://example.com/file", HTTPFileSystem),
        ],
    )
    def test_protocol_usage(
        self, dir_path: str, instance_type: tp.Type[TFileSystem],
    ) -> None:
        data_set = KerasModelDataset(filepath=dir_path)
        assert isinstance(data_set._fs, instance_type)
        path = dir_path.split(PROTOCOL_DELIMITER, 1)[-1]
        assert str(data_set._filepath) == path
        assert isinstance(data_set._filepath, pathlib.PurePosixPath)


class TestKerasModelDatasetVersioned(object):
    @pytest.mark.parametrize(
        "load_version,save_version",
        [
            (
                "2019-01-01T23.59.59.999Z",
                "2019-01-01T23.59.59.999Z",
            ),  # long version names can fail on Win machines due to 260 max filepath
            (
                None,
                None,
            ),
        ],
        indirect=True,
    )
    def test_keras_model_predicts_same_after_save_and_load(
        self,
        keras_model: TensorFlowModelWrapper,
        versioned_keras_model_dataset: KerasModelDataset,
        data: pd.DataFrame,
    ) -> None:
        initial_prediction = keras_model.predict(data)
        versioned_keras_model_dataset.save(keras_model)
        keras_model_reloaded = versioned_keras_model_dataset.load()
        prediction_after_reload = keras_model_reloaded.predict(data)
        assert np.array_equal(initial_prediction, prediction_after_reload)

    def test_no_versions(
        self, versioned_keras_model_dataset: KerasModelDataset,
    ) -> None:
        """Check the error if no versions are available for load."""
        pattern = r"Did not find any versions for KerasModelDataset\(.+\)"
        with pytest.raises(DatasetError, match=pattern):
            versioned_keras_model_dataset.load()

    def test_prevent_overwrite(
        self,
        versioned_keras_model_dataset: KerasModelDataset,
        keras_model: TensorFlowModelWrapper,
    ) -> None:
        """Check the error when attempting to override the data set if the
        corresponding json file for a given save version already exists."""
        versioned_keras_model_dataset.save(keras_model)
        pattern = (
            r"Save path \'.+\' for KerasModelDataset\(.+\) must "
            r"not exist if versioning is enabled\."
        )
        with pytest.raises(DatasetError, match=pattern):
            versioned_keras_model_dataset.save(keras_model)

    @pytest.mark.parametrize(
        "load_version", ["2019-01-01T23.59.59.999Z"], indirect=True,
    )
    @pytest.mark.parametrize(
        "save_version", ["2019-01-02T00.00.00.000Z"], indirect=True,
    )
    def test_save_version_warning(
        self,
        versioned_keras_model_dataset: KerasModelDataset,
        load_version: tp.Optional[str],
        save_version: tp.Optional[str],
        keras_model: TensorFlowModelWrapper,
    ):
        """Check the warning when saving to the path that differs from
        the subsequent load path."""
        pattern = (
            f"Save version '{save_version}' did not match "
            f"load version '{load_version}' for "
            r"KerasModelDataset\(.+\)"
        )
        with pytest.warns(UserWarning, match=pattern):
            versioned_keras_model_dataset.save(keras_model)

    def test_http_filesystem_no_versioning(self):
        pattern = "Versioning is not supported for HTTP protocols."
        with pytest.raises(DatasetError, match=pattern):
            KerasModelDataset(
                filepath="https://example.com/file",
                version=Version(None, None),
            )

    def test_versioning_existing_dataset(
        self,
        keras_model_dataset: KerasModelDataset,
        versioned_keras_model_dataset: KerasModelDataset,
        keras_model: TensorFlowModelWrapper,
    ) -> None:
        """Check behavior when attempting to save a versioned dataset on top of an
        already existing (non-versioned) dataset. Note: because KerasModelDataset
        saves to a directory even if non-versioned, an error is not expected."""
        keras_model_dataset.save(keras_model)
        assert keras_model_dataset.exists()
        assert keras_model_dataset._filepath == versioned_keras_model_dataset._filepath
        versioned_keras_model_dataset.save(keras_model)
        assert versioned_keras_model_dataset.exists()

    def test_version_str_repr(
        self,
        keras_model_dataset: KerasModelDataset,
        versioned_keras_model_dataset: KerasModelDataset,
    ) -> None:
        """Test that version is in string representation of the class instance
        when applicable."""

        assert str(keras_model_dataset._filepath) in str(keras_model_dataset)
        assert "version=" not in str(keras_model_dataset)
        assert "protocol" in str(keras_model_dataset)
        assert str(versioned_keras_model_dataset._filepath) in str(
            versioned_keras_model_dataset,
        )
        ver_str = f"version={versioned_keras_model_dataset._version}"
        assert ver_str in str(versioned_keras_model_dataset)
        assert "protocol" in str(versioned_keras_model_dataset)


@pytest.mark.parametrize("metrics,expected_output", [
    (
        {
            "MAE": 42,
            "R2": 0.9,
        },
        {
            "MAE": {
                "value": 42,
                "step": None,
            },
            "R2": {
                "value": 0.9,
                "step": None,
            },
        },
    ),
    (
        {
            "MAE": 0,
            "R2": -0.1,
        },
        {
            "MAE": {
                "value": 0,
                "step": None,
            },
            "R2": {
                "value": -0.1,
                "step": None,
            },
        },
    ),
])
def test_convert_metrics_to_nested_mlflow_dict(
    metrics: tp.Dict[str, float],
    expected_output: _TKedroMLflowMetrics,
) -> None:
    assert convert_metrics_to_nested_mlflow_dict(metrics) == expected_output
