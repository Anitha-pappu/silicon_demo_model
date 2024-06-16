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
import typing as tp
from pathlib import Path

import pytest
import yaml
from kedro.io import DataCatalog
from kedro.io.core import Version
from kedro.pipeline import Pipeline, node
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import CSVDataset

from preprocessing import (
    TagsConfig,
    TTagParameters,
    calculate_tag_range,
    deduplicate_pandas,
    enforce_schema,
    get_tag_config,
    interpolate_cols,
    remove_null_columns,
    remove_outlier,
    replace_inf_values,
    resample_data,
    round_timestamps,
    set_off_equipment_to_zero,
    unify_timestamp_col_name,
)


class TagsConfigCSVLocalDataSet(CSVDataset):

    def __init__(
        self,
        filepath: str,
        load_args: tp.Dict[str, tp.Any] = None,
        save_args: tp.Dict[str, tp.Any] = None,
        version: Version = None,
        credentials: tp.Dict[str, tp.Any] = None,
        fs_args: tp.Dict[str, tp.Any] = None,
    ) -> None:
        self._tc_load_args = {
            load_arg: load_args.pop(load_arg)
            for load_arg in frozenset(("delimiter", "parameters_schema"))
            if load_arg in load_args
        } if load_args else {}

        super().__init__(
            filepath=filepath,
            load_args=load_args,
            save_args=save_args,
            version=version,
            credentials=credentials,
            fs_args=fs_args,
        )

    def _load(self) -> TagsConfig[TTagParameters]:
        return get_tag_config(
            str(self._filepath),
            "csv",
            **self._tc_load_args,
        )

    def _save(self, config: TagsConfig[TTagParameters]) -> None:
        df = config.to_df()
        super()._save(df)


@pytest.fixture
def kedro_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                replace_inf_values,
                inputs={"data": "input_data"},
                outputs="post_replace_inf_values",
                name="replace_inf_values",
            ),
            node(
                remove_null_columns,
                inputs={"data": "post_replace_inf_values"},
                outputs="post_remove_null_columns",
                name="remove_null_columns",
            ),
            node(
                unify_timestamp_col_name,
                inputs={
                    "datetime_col": "params:cleaning.datetime_col",
                    "data": "post_remove_null_columns",
                    "unified_name": "params:cleaning.unitifed_timestamp",
                },
                outputs="post_unify_timestamp_col_name",
                name="unify_timestamp_col_name",
            ),
            node(
                round_timestamps,
                inputs={
                    "frequency": "params:cleaning.round_timestamps.frequency",
                    "data": "post_unify_timestamp_col_name",
                    "datetime_col": "params:cleaning.unitifed_timestamp",
                },
                outputs="post_round_timestamps",
                name="round_timestamps",
            ),
            node(
                deduplicate_pandas,
                inputs={"data": "post_round_timestamps"},
                outputs="post_deduplicate_pandas",
                name="deduplicate_pandas",
            ),
            node(
                enforce_schema,
                inputs={
                    "data": "post_deduplicate_pandas",
                    "meta_config": "meta_config",
                },
                outputs="post_enforce_schema",
                name="enforce_schema",
            ),
            node(
                calculate_tag_range,
                inputs={
                    "data": "post_enforce_schema",
                    "method": "params:cleaning.get_tag_range.method",
                },
                outputs="tag_range",
                name="get_tag_range",
            ),
            node(
                remove_outlier,
                inputs={
                    "data": "post_enforce_schema",
                    "outliers_config": "outliers_config",
                    "rule": "params:cleaning.outlier",
                },
                outputs=["post_remove_outlier", "outlier_summary"],
                name="remove_outlier",
            ),
            node(
                interpolate_cols,
                inputs={
                    "data": "post_remove_outlier",
                    "impute_config": "impute_config",
                    "kwargs": "params:imputing.interpolate.interpolate_kwargs",
                },
                outputs=["post_interpolate_cols", "interpolate_summary"],
                name="interpolate_cols",
            ),
            node(
                set_off_equipment_to_zero,
                inputs={
                    "data": "post_interpolate_cols",
                    "meta_config": "meta_config",
                    "on_off_dep_config": "on_off_dep_config",
                },
                outputs="post_set_off_equipment_to_zero",
                name="set_off_equipment_to_zero",
            ),
            node(
                func=resample_data,
                inputs={
                    "data": "post_set_off_equipment_to_zero",
                    "resample_config": "resample_config",
                    "timestamp_col": "params:cleaning.unitifed_timestamp",
                    "errors": "params:resampling.errors",
                    "default_method": "params:resampling.default_method",
                    "default_rule": "params:resampling.default_rule",
                    "default_offset": "params:resampling.default_offset",
                },
                outputs="post_resample_data",
                name="resample_data",
            ),
        ],
    )


def test_pipeline(kedro_pipeline):
    runner = SequentialRunner()
    params = yaml.safe_load(
        (
            Path(os.path.dirname(__file__))
            / "fixtures"
            / "parameters"
            / "preprocessing.yml"
        ).read_text(),
    )

    data = Path(os.path.dirname(__file__)) / "fixtures" / "data"

    catalog = DataCatalog(
        datasets={
            "input_data": CSVDataset(filepath=str(data / "sample_data.csv")),
            "meta_config": TagsConfigCSVLocalDataSet(
                filepath=str(data / "sample_tags_meta_config.csv"),
                load_args={"parameters_schema": "meta", "delimiter": ";"},
            ),
            "impute_config": TagsConfigCSVLocalDataSet(
                filepath=str(data / "sample_tags_imputation_config.csv"),
                load_args={"parameters_schema": "impute", "delimiter": ";"},
            ),
            "outliers_config": TagsConfigCSVLocalDataSet(
                filepath=str(data / "sample_tags_outliers_config.csv"),
                load_args={"parameters_schema": "outliers", "delimiter": ";"},
            ),
            "on_off_dep_config": TagsConfigCSVLocalDataSet(
                filepath=str(data / "sample_tags_on_off_dependencies_config.csv"),
                load_args={"parameters_schema": "on_off", "delimiter": ";"},
            ),
            "resample_config": TagsConfigCSVLocalDataSet(
                filepath=str(data / "sample_tags_resample_config.csv"),
                load_args={"parameters_schema": "resample", "delimiter": ";"},
            ),
        },
        feed_dict=params,
    )

    def update_feed_dict(p, parent):  # noqa: WPS430
        for k, v in p.items():
            catalog.add_feed_dict({f"{parent}.{k}": v})
            if isinstance(v, dict):
                update_feed_dict(v, f"{parent}.{k}")

    update_feed_dict(params["cleaning"], "params:cleaning")
    update_feed_dict(params["imputing"], "params:imputing")
    update_feed_dict(params["resampling"], "params:resampling")

    runner.run(kedro_pipeline, catalog)
