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

from optimus_core.utils import partial_wrapper
from recommend import (
    calculate_implementation_status,
    collect_recs_vs_actual_implementation,
    cra_export,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_export_steps() -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                lambda dict_: dict_["tags_meta"],
                inputs="meta",
                outputs="tag_meta_config",
            ),
            node(
                partial_wrapper(
                    cra_export.MetaDataConfig,
                    schema=cra_export.TagMetaData,
                ),
                inputs={"meta_data": "tag_meta_config"},
                outputs="tags_meta",
            ),
            node(
                lambda dict_: dict_["targets_meta"],
                inputs="meta",
                outputs="target_meta_config",
            ),
            node(
                partial_wrapper(
                    cra_export.MetaDataConfig,
                    schema=cra_export.TargetMetaData,
                ),
                inputs={"meta_data": "target_meta_config"},
                outputs="target_meta",
            ),
            node(
                lambda dict_: dict_["plant_info"],
                inputs="meta",
                outputs="plant_status_config",
            ),
            node(
                partial_wrapper(
                    cra_export.MetaDataConfig,
                    schema=cra_export.PlantStatusData,
                ),
                inputs={"meta_data": "plant_status_config"},
                outputs="plant_status",
            ),
            node(
                cra_export.prepare_runs,
                inputs={
                    "solutions": "solutions",
                    "iso_format": "params:recommend.cra_export.timestamp_format",
                    "timestamp_column": "params:recommend.cra_export.timestamp_column",
                },
                outputs="cra_runs",
            ),
            node(
                cra_export.prepare_tags,
                inputs={
                    "tag_meta": "tags_meta",
                    "plant_status": "plant_status",
                },
                outputs="cra_tags",
            ),
            node(
                cra_export.prepare_targets,
                inputs={
                    "target_meta": "target_meta",
                },
                outputs="cra_targets",
            ),
            node(
                cra_export.prepare_plant_info,
                inputs={
                    "plant_info": "plant_status",
                    "solutions": "solutions",
                    "actual_data": "test_data",
                    "iso_format": "params:recommend.cra_export.timestamp_format",
                    "timestamp_column": "params:recommend.cra_export.timestamp_column",
                },
                outputs="cra_plant_info",
            ),
            node(
                lambda df: df.rename({"value_after_recs": "silica_conc"}, axis=1),
                inputs="value_after_recs_act",
                outputs="actual_values_data",
            ),
            node(
                cra_export.prepare_actuals,
                inputs={
                    "actual_values_data": "actual_values_data",
                    "actual_values_col": "params:recommend.cra_export.target_col",
                    "target_meta": "target_meta",
                    "iso_format": "params:recommend.cra_export.timestamp_format",
                    "timestamp_column": "params:recommend.cra_export.timestamp_column",
                },
                outputs="cra_actuals",
            ),
            node(
                cra_export.prepare_predictions,
                inputs={
                    "baseline_values": "test_data_with_baseline_predictions",
                    "optimized_values": "value_after_recs_opt",
                    "model_prediction_bounds": "model_prediction_bounds",
                    "solutions": "solutions",
                    "target_meta": "target_meta",
                    "target_name": "params:recommend.cra_export.target_col",
                    "cols_export": "params:recommend.cra_export.cols_export",
                    "iso_format": "params:recommend.cra_export.timestamp_format",
                    "timestamp_column": "params:recommend.cra_export.timestamp_column",
                },
                outputs="cra_predictions",
            ),
            node(
                cra_export.prepare_states,
                inputs={
                    "solutions": "solutions",
                    "tag_meta": "tags_meta",
                },
                outputs="cra_states",
            ),
            node(
                cra_export.prepare_recommendations,
                inputs={
                    "solutions": "solutions",
                    "controlled_parameters_config": "controlled_parameters_config",
                    "tag_meta": "tags_meta",
                    "target_meta": "target_meta",
                    "target_name": "params:recommend.cra_export.target_col",
                },
                outputs="cra_recommendations",
            ),

            node(
                collect_recs_vs_actual_implementation,
                inputs={
                    "cra_recommendations": "old_cra_recommendations",
                    "cra_states": "old_cra_states",
                    "cra_runs": "old_cra_runs",
                    "imp_data": "test_data",
                    "tag_meta": "tags_meta",
                    "offset": "params:recommend.imp_tracking.offset",
                },
                outputs="implementation_input",
            ),

            node(
                calculate_implementation_status,
                inputs={
                    "implementation_data": "implementation_input",
                    "method": "params:recommend.imp_tracking.method",
                    "clip": "params:recommend.imp_tracking.imp_tracking_params.clip",
                },
                outputs="implementation_status",
            ),
            node(
                cra_export.prepare_implementation_status,
                inputs={
                    "implementation_status": "implementation_status",
                },
                outputs="cra_implementation_status",
            ),
            node(
                cra_export.prepare_sse,
                inputs={},
                outputs="cra_sse",
            ),
        ],
    ).tag("export")
