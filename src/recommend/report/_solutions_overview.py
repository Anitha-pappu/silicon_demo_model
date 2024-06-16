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
import warnings

import pandas as pd
import plotly.graph_objects as go
from pydantic import TypeAdapter

from recommend.controlled_parameters import ControlledParameter
from recommend.solution import Solutions
from reporting.rendering.identifiers import Table
from reporting.rendering.types import TRenderingAgnosticDict
from reporting.types import TColumn

from . import charts

# WPS432 can be violated here to increase readability
_SUMMARY_LAYOUT = dict(height=370, width=700, margin=dict(l=18, t=0, b=0))  # noqa: WPS432,E501
_OBJECTIVE_COMPARISON_LAYOUT = dict(title="", margin=dict(l=100, t=10, b=50, r=50))  # noqa: WPS432,E501
_ACTUAL_VS_UPLIFT_LAYOUT = dict(title="", margin=dict(l=100, t=25, b=50, r=50))  # noqa: WPS432,E501
_CONTROLS_ANALYSIS_LAYOUT = dict(title="", margin=dict(l=100, t=0, b=50, r=50))  # noqa: WPS432,E501
_MISSINGS_HEATMAP_LAYOUT = dict(title="", margin=dict(l=100, t=25, b=50, r=50))  # noqa: WPS432,E501
_CONTROLS_COMPARISON_LAYOUT = dict(title="", margin=dict(l=100, t=50, b=50, r=50))  # noqa: WPS432,E501
_CLUSTER_ANALYSIS_LAYOUT = dict(title="", margin=dict(l=100, t=50, b=50, r=50))  # noqa: WPS432,E501
_PENALTY_TIMELINE_LAYOUT = dict(title="", margin=dict(l=100, t=50, b=100, r=50))  # noqa: WPS432,E501

_TControlsConfig = tp.Union[
    tp.List[tp.Dict[str, tp.Any]],
    tp.List[ControlledParameter],
    None,
]


def get_solutions_overview(
    solutions: Solutions,
    actual_target_column: str,
    timestamp_column: str = "timestamp",
    objective_units: str | None = None,
    controls_config: _TControlsConfig = None,
    reference_data: pd.DataFrame | None = None,
    cluster_column: str | None = None,
) -> TRenderingAgnosticDict:
    """
    Generates multi-level dict of figures for the Solutions Overview.

    There are 7 main sections in this report:
        1. Overall summary (what's the overall uplift;
           how many rows were optimized; objective before & after comparison)
        2. [WIP] Recs availability (what are the reasons our rows aren't optimized)
        3. Constraints analysis (constraints timeline,
           summary table by index and by constraint)
        4. Data insights (includes basic EDA + solutions' data vs reference comparison)
        5. [WIP] Model insights (compare model performance on two datasets
           used for optimization and for training)
        6. Controls analysis (compare controls on before & after the optimization)


    This report can be viewed in the notebook or
    used in report generation to produce standalone report file.

    Args:
        solutions: results of optimization (``recommend.optimize`` call)
        actual_target_column: actual target values
            (i.e. true, non-modelled value of objective seen at this step)
        timestamp_column:
        objective_units: is shown in the summary section
        controls_config: list of controls' parameters that is parsed into
            list[ControlledParameter]
        reference_data: dataset used for models training;
            if provided, used as a reference for comparing missing values,
            controls from ``controls_config`` and
            ``actual_target_column`` distributions
        cluster_column: input data cluster column;
            if provided, adds "Cluster Analysis" chart to "Data Insights" section

    Returns:
        Dictionary of recommendations analysis figures
    """

    if not solutions:
        warnings.warn("Solutions are empty")
        return {}

    df = solutions.to_frame()
    df_before_after = df[df["is_successful_optimization"]]

    return {
        "Results Overview": _get_results_overview_section(
            solutions,
            df_before_after,
            actual_target_column,
            objective_units,
            timestamp_column,
        ),
        "Controls Analysis": _get_controls_analysis_section(
            solutions,
            df_before_after,
            timestamp_column,
            controls_config,
        ),
        "Constraints Analysis": _get_constraints_section(
            solutions,
            timestamp_column,
        ),
        "Data Insights": _get_data_insights_section(
            solutions,
            actual_target_column,
            timestamp_column,
            reference_data,
            cluster_column,
        ),
    }


def _get_results_overview_section(
    solutions: Solutions,
    df_before_after: pd.DataFrame,
    actual_target_column: str,
    objective_units: str | None,
    timestamp_column: str,
) -> TRenderingAgnosticDict:
    return {
        "Short Summary": [
            charts
            .plot_period_details(solutions, timestamp_column),

            charts
            .plot_summary(solutions, objective_units)
            .update_layout(_SUMMARY_LAYOUT),
        ],
        "Objective Comparison": [
            charts
            .plot_objective_comparison(
                df_before_after[solutions.export_columns.objective],
            )
            .update_layout(_OBJECTIVE_COMPARISON_LAYOUT),
        ],
        "Actual Target vs. Optimization Uplift": [
            charts.plot_actual_vs_uplift(
                df_before_after,
                actual_target_column=(
                    actual_target_column, solutions.export_columns.initial,
                ),
                timestamp_column=(timestamp_column, solutions.export_columns.initial),
                target_name="Silica Conc.",
            ).update_layout(_ACTUAL_VS_UPLIFT_LAYOUT),
        ],
    }


def _get_controls_analysis_section(
    solutions: Solutions,
    df_before_after: pd.DataFrame,
    timestamp_column: str,
    controls_config: _TControlsConfig,
) -> TRenderingAgnosticDict:
    controls_config_parsed = (
        TypeAdapter(tp.List[ControlledParameter]).validate_python(controls_config)
        if controls_config is not None
        else None
    )
    controls_domains = (
        _extract_controls_domains(solutions)
        if controls_config_parsed is None
        else {ctrl.name: (ctrl.op_min, ctrl.op_max) for ctrl in controls_config_parsed}
    )

    section: TRenderingAgnosticDict = {}

    if controls_config is not None:
        df_ood = (
            charts.get_out_of_domain_summary(
                df_before_after,
                solutions.export_columns.initial,
                solutions.export_columns.optimized,
                controls_domains,
            )
        )
        section["Optimized Variables Domain Summary"] = Table(
            table=df_ood,
            precision=0,
            columns_to_color_as_bars=list(df_ood.columns),
        )

    section["Before vs. After Comparison"] = {
        control: charts.plot_controls_comparison(  # noqa: WPS317 (false alarm)
            df_before_after,
            (control, solutions.export_columns.initial),
            (control, solutions.export_columns.optimized),
            (timestamp_column, solutions.export_columns.initial),
            control,
            controls_domains[control],
        ).update_layout(_CONTROLS_ANALYSIS_LAYOUT)
        for control in solutions.controls
    }
    return section


def _extract_controls_domains(
    solutions: Solutions,
) -> tp.Dict[str, tp.Tuple[float, float]]:
    """
    Extracts widest range from ``solution.controls_domain`` by picking min/max
    for lower/upper bounds from all ranges for each control.

    Note, this doesn't converge to op_min, op_max specified in recommendation's
    controlled parameters config since these domains include step size limitation.
    """
    df_controls_domain = (
        pd.DataFrame([solution.controls_domain for solution in solutions.values()])
        .apply(pd.Series.explode)
    )
    df_controls_ranges = pd.concat(
        [df_controls_domain.min(), df_controls_domain.max()], axis=1,
    )
    return dict(zip(df_controls_ranges.index, df_controls_ranges.values.astype(float)))


def _get_data_insights_section(
    solutions: Solutions,
    actual_target_column: str | None,
    timestamp_column: str,
    reference_data: pd.DataFrame | None,
    cluster_column: str | None,
) -> TRenderingAgnosticDict:
    data = (
        solutions.to_frame().reorder_levels([1, 0], axis=1)
        [solutions.export_columns.initial]
        [solutions.states]
    )
    missing_heatmap = charts.plot_missing_data_heatmap(data, timestamp_column)
    if isinstance(missing_heatmap, go.Figure):
        missing_heatmap.update_layout(_MISSINGS_HEATMAP_LAYOUT)
    section: TRenderingAgnosticDict = {"Missing Data Heatmap": missing_heatmap}
    missing_data_summary = charts.get_missing_data_summary(data, reference_data)
    contains_missing_values = bool(missing_data_summary.sum().sum())
    if contains_missing_values:
        section["Missing Data Summary"] = Table(
            table=missing_data_summary,
            columns_to_color_as_bars=list(missing_data_summary.columns),
            sort_by=[(column, "desc") for column in missing_data_summary.columns],
            precision=0,
        )
    _add_data_vs_ref_distributions_comparison(
        section, solutions, data, reference_data, actual_target_column,
    )
    _add_cluster_analysis_section(section, data, cluster_column)
    return section


def _add_cluster_analysis_section(
    section: TRenderingAgnosticDict, data: pd.DataFrame, cluster_column: TColumn | None,
) -> None:
    # todo: add cluster analysis for reference data as well
    if cluster_column is not None:
        section["Cluster Analysis"] = (
            charts.plot_cluster_analysis(data, cluster_column)
            .update_layout(_CLUSTER_ANALYSIS_LAYOUT)
        )


def _add_data_vs_ref_distributions_comparison(
    section: TRenderingAgnosticDict,
    solutions: Solutions,
    data: pd.DataFrame,
    reference_data: pd.DataFrame | None,
    actual_target_column: str | None,
) -> None:
    if reference_data is not None:
        columns_to_compare = (
            solutions.controls
            if actual_target_column is None
            else [actual_target_column, *solutions.controls]
        )
        section["Data to Optimize vs Reference Distributions Comparison"] = (
            charts.plot_features_comparison(data, columns_to_compare, reference_data)
            .update_layout(_CONTROLS_COMPARISON_LAYOUT)
        )


def _get_constraints_section(
    solutions: Solutions,
    timestamp_column: str,
) -> TRenderingAgnosticDict:
    (
        df_by_index_penalties,
        df_by_penalty_violation_count,
    ) = charts.get_penalty_summaries(solutions, timestamp_column)
    penalty_columns = df_by_index_penalties.columns.drop(
        [charts.TOTAL_VIOLATIONS, charts.TOTAL_PENALTY],
    )
    return {
        "By Constraint Summary": Table(
            table=df_by_penalty_violation_count,
            precision=0,
            sort_by=[(pnlt, "desc") for pnlt in df_by_penalty_violation_count.columns],
            columns_to_color_as_bars=list(df_by_penalty_violation_count.columns),
        ),
        "Penalty Over Time": [
            charts.plot_penalty_over_time(df_by_index_penalties)
            .update_layout(_PENALTY_TIMELINE_LAYOUT),

            Table(
                table=df_by_index_penalties,
                precision=(
                    {pnlt: 2 for pnlt in (charts.TOTAL_PENALTY, *penalty_columns)}
                    | {charts.TOTAL_VIOLATIONS: 0}
                ),
                columns_to_color_as_bars=list(df_by_index_penalties.columns),
                sort_by=[(timestamp_column, "asc")],
            ),
        ],
    }
