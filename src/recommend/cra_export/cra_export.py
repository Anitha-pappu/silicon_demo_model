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
CRA export preparation functions
"""

import typing as tp
import uuid

import pandas as pd

from ..controlled_parameters import ControlledParametersConfig
from ..solution import Solution, Solutions
from .meta_data import (
    MetaDataConfig,
    PlantStatusData,
    TagMetaData,
    TargetMetaData,
)
from .utils import (
    get_id_mapping,
    get_run_id,
    get_timestamp_in_iso_format,
    parse_timestamp,
)

ValidTypes = tp.Union[str, float, tp.Sequence['ValidTypes'], None]
TSingleDict = tp.Dict[str, ValidTypes]
TJson = tp.List[TSingleDict]
_DEFAULT_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # noqa: WPS323
_DEFAULT_TIMESTAMP_COLUMN = "timestamp"


def prepare_actuals(
    actual_values_data: pd.DataFrame,
    actual_values_col: str,
    target_meta: MetaDataConfig[TargetMetaData],
    iso_format: str = _DEFAULT_ISO_FORMAT,
    timestamp_column: str = _DEFAULT_TIMESTAMP_COLUMN,
) -> TJson:
    """Creates a list of actual values in the format of the ``actuals`` endpoint.

    Args:
        actual_values_data: dataframe with actual values.
        actual_values_col: column with actual values.
        target_meta: target meta information.
        iso_format: format for timestamp.
        timestamp_column: column name for timestamp.

    Returns:
        An input to 'actuals' endpoint of cra_api.
    """
    target_id_mapping = get_id_mapping(target_meta)
    actual_values_data = actual_values_data.copy()
    actual_values_data[timestamp_column] = get_timestamp_in_iso_format(
        actual_values_data[timestamp_column], iso_format,
    )
    actuals = []
    for _, row in actual_values_data.iterrows():
        actual: TSingleDict = {
            "id": str(uuid.uuid4()),
            "target_id": target_id_mapping[actual_values_col],
            "value": float(row[actual_values_col]),
            "timestamp": str(row[timestamp_column]),
        }
        actuals.append(actual)

    return tp.cast(TJson, actuals)


def prepare_predictions(
    baseline_values: pd.DataFrame,
    optimized_values: pd.DataFrame,
    model_prediction_bounds: pd.DataFrame,
    solutions: Solutions,
    target_meta: MetaDataConfig[TargetMetaData],
    target_name: str,
    cols_export: dict[str, str],
    iso_format: str = _DEFAULT_ISO_FORMAT,
    timestamp_column: str = _DEFAULT_TIMESTAMP_COLUMN,
) -> TJson:
    """Creates a list of predictions in the format of the ``predictions`` endpoint.

    Args:
        baseline_values: baseline values.
        optimized_values: optimized values.
        model_prediction_bounds: a dataframe with columns timestamp, actuals,
        predictions, upper_bound, lower_bound
        solutions: a mapping of optimization results.
        target_meta: target meta information.
        target_name: name of the target.
        cols_export: columns to export for each dataframe. If consists of a dictionary
            with keys 'baseline' and 'optimized' and values the column names to export
            for the corresponding dataframe.
        iso_format: format for timestamp.
        timestamp_column: column name for timestamp.

    Returns:
        An input to 'predictions' endpoint of cra_api.

    Raises:
        ValueError: if keys of cols_export are not 'baseline', 'optimized' and
        'predicted'.
    """
    if not set(cols_export.keys()) == {"baseline", "optimized"}:
        raise ValueError(
            "Dictionaries keys in cols_list should be 'baseline' and 'optimized', "
            f"but got {cols_export.keys()}",
        )

    run_ids = get_run_id(solutions, timestamp_column, iso_format)
    target_id_mapping = get_id_mapping(target_meta)
    baseline_values[timestamp_column] = get_timestamp_in_iso_format(
        baseline_values[timestamp_column], iso_format,
    )
    optimized_values[timestamp_column] = get_timestamp_in_iso_format(
        optimized_values[timestamp_column], iso_format,
    )
    model_prediction_bounds[timestamp_column] = get_timestamp_in_iso_format(
        model_prediction_bounds[timestamp_column], iso_format,
    )

    predictions = []
    for timestamp in run_ids.keys():
        single_model_preds_bounds = model_prediction_bounds.loc[
            model_prediction_bounds[timestamp_column] == timestamp,
        ]
        prediction: TSingleDict = {
            "id": str(uuid.uuid4()),
            "run_id": run_ids[timestamp],
            "target_id": target_id_mapping[target_name],
            "baseline": baseline_values.loc[
                baseline_values[timestamp_column] == timestamp,
                cols_export["baseline"],
            ].iloc[0],
            "optimized": optimized_values.loc[
                optimized_values[timestamp_column] == timestamp,
                cols_export["optimized"],
            ].iloc[0],
            "predicted": single_model_preds_bounds["predictions"].iloc[0],
            "upper_bound": single_model_preds_bounds["upper_bound"].iloc[0],
            "lower_bound": single_model_preds_bounds["lower_bound"].iloc[0],
        }
        predictions.append(prediction)

    return tp.cast(TJson, predictions)


def prepare_runs(
    solutions: Solutions,
    iso_format: str = _DEFAULT_ISO_FORMAT,
    timestamp_column: str = _DEFAULT_TIMESTAMP_COLUMN,
) -> TJson:
    """Create a list of runs in the format of ``runs`` endpoint of cra_api.

    Args:
        solutions: optimization solutions.
        iso_format: format for timestamp.
        timestamp_column: column name for timestamp.

    Returns:
        An input to 'runs' endpoint of cra_api.
    """
    runs = []
    for solution in solutions.values():
        # timestamp parsing
        timestamp = solution.context_parameters[timestamp_column]
        timestamp = parse_timestamp(iso_format, timestamp)
        single_run: TSingleDict
        if solution.is_successful_optimization:
            single_run = {"id": solution.run_id, "timestamp": timestamp}
        else:
            single_run = {
                "id": solution.run_id,
                "error_message": "Optimization did not improve current set points",
                "timestamp": timestamp,
            }
        runs.append(single_run)
    return runs


def prepare_states(
    solutions: Solutions,
    tag_meta: MetaDataConfig[TagMetaData],
    states_to_export: tp.Optional[tp.List[str]] = None,
) -> TJson:
    """
    Creates a list of states in the format of ``states`` endpoint of cra_api.

    Args:
        solutions: a mapping of optimization results.
        tag_meta: controlled parameters meta information.
        states_to_export: a list of state variables to show on UI.

    Returns:
        An input to 'states' endpoint of cra_api.
    """

    tag_id_mapping = get_id_mapping(tag_meta)
    states = []
    for solution in solutions.values():
        states_selection = (
            states_to_export
            if states_to_export is not None
            else solution.control_parameters_before.keys()
        )
        for state in states_selection:
            single_state: TSingleDict = {
                "id": str(uuid.uuid4()),
                "value": solution.control_parameters_before[state],
                "run_id": solution.run_id,
                "tag_id": tag_id_mapping[state],
            }
            states.append(single_state)
    return states


def prepare_tags(
    tag_meta: MetaDataConfig[TagMetaData],
    plant_status: MetaDataConfig[PlantStatusData],
) -> TJson:
    """
    Creates a list of tags in the format of ``tags`` endpoint of cra_api.

    Args:
        tag_meta: controlled parameters meta information.
        plant_status: plant status information.

    Returns:
        An input to 'tags' endpoint of cra_api.
    """
    export_tags = []
    single_tag: TSingleDict
    for metadata_tag in tag_meta:
        single_tag = {
            "id": metadata_tag.id,
            "clear_name": metadata_tag.clear_name,
            "unit": metadata_tag.unit,
            "area": metadata_tag.area,
            "precision": metadata_tag.precision,
            "priority": metadata_tag.priority,
        }
        export_tags.append(single_tag)
    for metadata_plant in plant_status:
        single_tag = {
            "id": metadata_plant.id,
            "clear_name": metadata_plant.clear_name,
            "unit": metadata_plant.unit,
            "area": metadata_plant.area,
            "precision": metadata_plant.precision,
        }
        export_tags.append(single_tag)

    return tp.cast(TJson, export_tags)


def prepare_targets(target_meta: MetaDataConfig[TargetMetaData]) -> TJson:
    """
    Creates a list of tags in the format of ``targets`` endpoint of cra_api.

    Args:
        target_meta: target meta information.

    Returns:
        An input to 'targets' endpoint of cra_api.
    """
    export_targets = []
    for metadata in target_meta:
        single_target: TSingleDict = {
            "id": metadata.id,
            "name": metadata.name,
            "unit": metadata.unit,
            "aggregation": metadata.aggregation,
            "objective": metadata.objective,
            "precision": metadata.precision,
        }
        export_targets.append(single_target)
    return tp.cast(TJson, export_targets)


def prepare_recommendations(
    solutions: Solutions,
    controlled_parameters_config: ControlledParametersConfig,
    tag_meta: MetaDataConfig[TagMetaData],
    target_meta: MetaDataConfig[TargetMetaData],
    target_name: str,
    default_status: str = "Pending",
    default_flagged: bool = False,
    active_controls_only: bool = True,
) -> TJson:
    """Creates a list of recommendations in the format of the
    ``recommendations`` endpoint.

    Args:
        solutions: a mapping of optimization results.
        controlled_parameters_config: controlled parameters config.
        tag_meta: controlled parameters meta information.
        target_meta: target meta information.
        target_name: name of the target.
        default_status: default status. Defaults to "Pending".
        default_flagged: default flagged. Defaults to False.
        active_controls_only: exports only active controls if set True,
            all controls otherwise

    Returns:
        An input to 'recommendations' endpoint of cra_api.
    """
    tag_id_mapping = get_id_mapping(tag_meta)
    target_id_mapping = get_id_mapping(target_meta)
    tolerance_mapping = {conf.tag: conf.tolerance for conf in tag_meta}

    recommendations = []
    solution: Solution
    for solution in solutions.values():
        controls_to_export = (
            solution.control_parameters_after.keys()
            if active_controls_only
            else list(controlled_parameters_config)
        )
        for control in controls_to_export:
            single_recommendation: TSingleDict = {
                "id": str(uuid.uuid4()),
                "value": solution.control_parameters_after[control],
                "tolerance": tolerance_mapping[control],
                "run_id": solution.run_id,
                "tag_id": tag_id_mapping[control],
                "target_id": target_id_mapping[target_name],
                "is_flagged": default_flagged,
                "status": default_status,
            }
            recommendations.append(single_recommendation)
    return recommendations


def prepare_implementation_status(implementation_status: pd.DataFrame) -> TJson:
    """
    Creates a list of implementation statuses to update the ``Recommendations``
    endpoint of cra_api.

    Args:
        implementation_status: a percentage status.

    Returns:
        An input to update 'Recommendations' endpoint of cra_api.
    """
    implementation_status_cra = (
        implementation_status[["id", "implementation_perc"]]
        .rename({"implementation_perc": "implementation_status"}, axis=1)
        .to_dict(orient="records")
    )
    return tp.cast(TJson, implementation_status_cra)


def prepare_plant_info(
    plant_info: MetaDataConfig[PlantStatusData],
    solutions: Solutions,
    actual_data: pd.DataFrame,
    iso_format: str = _DEFAULT_ISO_FORMAT,
    timestamp_column: str = _DEFAULT_TIMESTAMP_COLUMN,
) -> TJson:
    """
    Creates a list of plant information to update the ``plant_info`` endpoint of
    cra_api.

    Returns:
        An input to update 'plant_info' endpoint of cra_api.
    """

    run_ids = get_run_id(solutions, timestamp_column, iso_format)

    actual_data = actual_data.copy()
    actual_data[timestamp_column] = get_timestamp_in_iso_format(
        actual_data[timestamp_column], iso_format,
    )

    plant_info_dict = []
    for tag in plant_info:
        for timestamp in run_ids.keys():
            actual_value = actual_data.loc[
                actual_data[timestamp_column] == timestamp, tag.tag,
            ].iloc[0]
            single_plant_info: TSingleDict = {
                "id": str(uuid.uuid4()),
                "run_id": run_ids[timestamp],
                "tag_id": tag.id,
                "value": actual_value,
                "column_name": tag.column_name,
                "section": tag.section,
            }
            plant_info_dict.append(single_plant_info)
    return plant_info_dict


def prepare_sse() -> TSingleDict:
    """
    Signals the end of recommendations updates in the format of the ``sse`` endpoint.

    Returns:
        An input to 'see' endpoint of cra_api.
    """
    return {"event": "ui_update"}
