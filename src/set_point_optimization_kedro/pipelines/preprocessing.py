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

from kedro.pipeline import Pipeline, node

from preprocessing import (
    deduplicate_pandas,
    enforce_schema,
    interpolate_cols,
    preprocessing_output_summary,
    remove_null_columns,
    remove_outlier,
    rename_tags,
    replace_inf_values,
    resample_data,
    round_timestamps,
    set_off_equipment_to_zero,
    unify_timestamp_col_name,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_preprocessing_steps() -> Pipeline:
    return Pipeline(
        [
            node(
                rename_tags,
                inputs={
                    "tags_raw_config": "tags_raw_config",
                    "data_to_rename": "input_data",
                },
                outputs="renamed_input_data",
                name="rename_tags",
            ),
            node(
                replace_inf_values,
                inputs={"data": "renamed_input_data"},
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
                    "meta_config": "tags_meta_config",
                },
                outputs="post_enforce_schema",
                name="enforce_schema",
            ),
            node(
                remove_outlier,
                inputs={
                    "data": "post_enforce_schema",
                    "outliers_config": "tags_outliers_config",
                    "rule": "params:cleaning.outliers_rule",
                },
                outputs=["post_remove_outlier", "outlier_summary"],
                name="remove_outlier",
            ),
            node(
                interpolate_cols,
                inputs={
                    "data": "post_remove_outlier",
                    "impute_config": "tags_impute_config",
                    "kwargs": "params:cleaning.interpolate_kwargs",
                },
                outputs=["post_interpolate_cols", "interpolate_summary"],
                name="interpolate_cols",
            ),
            node(
                set_off_equipment_to_zero,
                inputs={
                    "data": "post_interpolate_cols",
                    "meta_config": "tags_meta_config",
                    "on_off_dep_config": "tags_on_off_config",
                },
                outputs="post_set_off_equipment_to_zero",
                name="set_off_equipment_to_zero",
            ),
            node(
                func=resample_data,
                inputs={
                    "data": "post_set_off_equipment_to_zero",
                    "resample_config": "tags_resample_config",
                    "timestamp_col": "params:cleaning.unitifed_timestamp",
                    "errors": "params:cleaning.resample_errors",
                },
                outputs="preprocessed_data",
                name="resample_data",
            ),
            node(
                preprocessing_output_summary,
                inputs={
                    "tags_raw_config": "tags_raw_config",
                    "tags_meta_config": "tags_meta_config",
                    "tags_outliers_config": "tags_outliers_config",
                    "tags_impute_config": "tags_impute_config",
                    "tags_on_off_config": "tags_on_off_config",
                    "tags_resample_config": "tags_resample_config",
                    "outlier_summary": "outlier_summary",
                    "interpolate_summary": "interpolate_summary",
                },
                outputs="tag_config_summary",
                name="preprocessing_summary",
            ),
        ],
    ).tag("preprocessing")
