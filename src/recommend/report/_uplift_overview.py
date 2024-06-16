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

import numpy as np
import pandas as pd

from reporting.charts.utils import calculate_optimal_bin_width
from reporting.rendering.identifiers import Table, Text
from reporting.rendering.types import TRenderingAgnosticDict

from . import charts

_UPLIFT_DIRECTION: tp.Literal["greater", "smaller"] = "greater"
_TEXT_TITLE_SIZE = 16
_TEXT_SIZE = 14
_TEXT_LEFT_MARGIN = 50
_MAX_CHARACTERS_PER_LINE_TESTS = 130
_MAX_CHARACTERS_PERFORMANCE_SUMMARY = 150
_MAX_CHARACTERS_OPERATORS_ADHERENCE = 120


def get_impact_overview(
    baseline_data: pd.DataFrame,
    baseline_historical_data: pd.DataFrame,
    optimized_uplift: pd.DataFrame,
    actual_uplift: pd.DataFrame,
    significance_mean_no_zero: pd.DataFrame,
    significance_uplift_no_model_error: pd.DataFrame,
    significance_baseline_mean_zero: pd.DataFrame,
    impact: pd.DataFrame,
    data_filtering: tp.Optional[pd.DataFrame] = None,
    timestamp_column: str = "timestamp",
    baseline_prediction_column: str = "model_prediction",
    baseline_target_column: str = "target",
) -> TRenderingAgnosticDict:
    """
    Generates nested dict with figures for the impact reporting.

    There are 4 main sections in this report:
        1. Impact Summary
        2. Data Filtering
        3. Impact Deep Dive
        4. Impact Significance


    This report can be viewed in the notebook or used in report generation to produce
    a standalone report file.

    Args:
        baseline_data: Dataframe with baseline data
        baseline_historical_data: Dataframe with baseline predictions and actual values
            over a historical dataset to assess its bias
        optimized_uplift: Dataframe with uplifts for counterfactual analysis
        actual_uplift: Dataframe with uplifts for impact analysis
        significance_mean_no_zero: Dataframe with significance analysis that proves that
            the mean uplift is not zero
        significance_uplift_no_model_error: Dataframe with significance analysis that
            proves that the uplift is not due to model error
        significance_baseline_mean_zero: Dataframe with significance analysis that
            proves that the baseline mean is zero
        impact: Dataframe with impact analysis
        data_filtering: Dataframe with acceptance ratio of recommendations and operators
            adherence to them. If ``None``, this section will not be included in the
            report
        timestamp_column: Name of the timestamp column
        baseline_prediction_column: Name of the column with baseline values
        baseline_target_column: Name of the column in ``baseline_historical_data`` with
            target values

    Returns:
        Dictionary of recommendations analysis figures

    """

    data = _preprocess_data_to_report(
        baseline_data,
        optimized_uplift,
        actual_uplift,
        timestamp_column,
        baseline_prediction_column,
    )
    baseline_historical_data = _get_baseline_error_to_report(
        baseline_historical_data,
        timestamp_column,
        baseline_prediction_column,
        baseline_target_column,
    )

    impact_overview: TRenderingAgnosticDict = {
        "Impact Summary": charts.plot_impact_summary(data, impact),
        "Data Filtering": _get_data_filtering_section(
            data_filtering,
        ),
        "Impact Deep Dive": _get_impact_deep_dive_section(
            data,
            impact,
            timestamp_column,
        ),
        "Impact Significance": _get_impact_significance_section(
            data,
            baseline_historical_data,
            significance_mean_no_zero,
            significance_uplift_no_model_error,
            significance_baseline_mean_zero,
        ),
    }

    if impact_overview["Data Filtering"] is None:
        impact_overview.pop("Data Filtering", None)

    return impact_overview


def _preprocess_data_to_report(
    baseline_data: pd.DataFrame,
    optimized_uplift: pd.DataFrame,
    actual_uplift: pd.DataFrame,
    timestamp_column: str = "timestamp",
    baseline_prediction_column: str = "model_prediction",
) -> pd.DataFrame:
    """
    Preprocesses dataframes to be used in the report by:
        - Selecting dataframe columns and renaming them to the standard names that will
            be used to report.
        - Merging dataframes into a unique source.
        - Calculating the optimal and actual values from the baseline and uplifts.
        - Finding a common group column if available (groups in ``actual_uplift`` take
            precedence).
        - Melting the dataframe to have a column wih the value type and another with its
            value

    Args:
        baseline_data: Dataframe with baseline data
        optimized_uplift: Dataframe with uplifts for counterfactual analysis
        actual_uplift: Dataframe with uplifts for impact analysis
        timestamp_column: Name of the timestamp column
        baseline_prediction_column: Name of the column with baseline values

    Returns:
        Dataframe with preprocessed data

    """
    baseline_data = baseline_data[
        [timestamp_column, baseline_prediction_column]
    ].rename(
        {baseline_prediction_column: "baseline"}, axis=1,
    )

    optimized_uplift = _select_and_rename_uplifts(
        data=optimized_uplift,
        timestamp_column=timestamp_column,
        rename_prefix="optimized",
    )

    actual_uplift = _select_and_rename_uplifts(
        data=actual_uplift,
        timestamp_column=timestamp_column,
        rename_prefix="actual",
    )

    data = (
        baseline_data
        .merge(optimized_uplift, on="timestamp")
        .merge(actual_uplift, on="timestamp")
        .sort_values(timestamp_column)
    )
    data["optimized"] = data["baseline"] + data["optimized_uplift"]
    data["actual"] = data["baseline"] + data["actual_uplift"]
    data["gap_to_optimal"] = data["optimized"] - data["actual"]

    if "optimized_group" in data.columns and "actual_group" in data.columns:
        num_discrepancies = np.sum(data["optimized_group"] != data["actual_group"])
        if num_discrepancies > 0:
            warnings.warn(
                message=f"Groups on optimized and actual uplift do not match in "
                f"{num_discrepancies} observations. Groups on actual uplifts will be "
                f"used for the report.",
                category=UserWarning,
            )
        data = data.rename({"actual_group": "group"}, axis=1)
        data = data.drop(["optimized_group"], axis=1)
    elif sum(["_group" in col for col in data.columns]) == 1:
        group_col = [col for col in data.columns if "_group" in col][0]
        data = data.rename({group_col: "group"}, axis=1)

    group_col = ["group"] if "group" in data.columns else []
    return data.melt(
        id_vars=["timestamp"] + group_col,
        var_name="value_type",
        value_name="value",
    )


def _select_and_rename_uplifts(
    data: pd.DataFrame,
    timestamp_column: str,
    rename_prefix: str,
) -> pd.DataFrame:
    """
    Selects and renames uplift columns.

    Args:
        data: Dataframe with uplifts
        timestamp_column: Name of the timestamp column
        rename_prefix: Prefix to add to the uplift columns

    Returns:
        Dataframe with selected and renamed columns

    """
    group_col = ["group"] if "group" in data.columns else []
    data = data[
        [timestamp_column, "uplift"] + group_col
    ]
    cols_to_rename = data.drop(timestamp_column, axis=1).columns
    return data.rename(
        {col: f"{rename_prefix}_{col}" for col in cols_to_rename}, axis=1,
    )


def _get_baseline_error_to_report(
    baseline_historical_data: pd.DataFrame,
    timestamp_column: str = "timestamp",
    baseline_prediction_column: str = "model_prediction",
    baseline_target_column: str = "target",
) -> pd.DataFrame:
    """
    Calculates the baseline error to be used in the report.

    Args:
        baseline_historical_data: Dataframe with baseline predictions and actual values
            over a historical dataset to assess its bias
        timestamp_column: Name of the timestamp column
        baseline_prediction_column: Name of the column with baseline values
        baseline_target_column: Name of the column in ``baseline_historical_data`` with
            target values

    Returns:
        Dataframe with baseline error

    """

    baseline_historical_data["error"] = (
        baseline_historical_data[baseline_prediction_column]
        - baseline_historical_data[baseline_target_column]
    )

    if "group" in baseline_historical_data.columns:
        return baseline_historical_data[[timestamp_column, "group", "error"]]
    return baseline_historical_data[[timestamp_column, "error"]]


def _get_data_filtering_section(
    data_filtering: tp.Optional[pd.DataFrame],
) -> TRenderingAgnosticDict | None:
    """
    Generates nested dict with figures for the data filtering section.

    Args:
        data_filtering: Dataframe with acceptance ratio of recommendations and operators
            adherence to them

    Returns:
        Dictionary of data filtering figures

    """
    if data_filtering is None:
        return None

    clean_data_filtering = charts.clean_data_filtering(data_filtering)

    return {
        "Performance summary": [
            Text(
                text="Performance summary explores the reasons why recommendations are "
                "not implemented:\n"
                "    - Downtime: System did not provide recommendations\n"
                "    - Not reviewed: Operators did not review the recommendation "
                "before the next one was issued\n"
                "    - Not approved: Operators did not approve the recommendation\n"
                "    - Not implemented: Operators did not implement the recommendation",
                left_margin=_TEXT_LEFT_MARGIN,
                max_characters_per_text_line=_MAX_CHARACTERS_PERFORMANCE_SUMMARY,
            ),
            (
                charts
                .plot_performance_summary(clean_data_filtering)
                .update_layout({"margin": {"t": 25}})
            ),
            *charts.plot_performance_summary_by_tag(clean_data_filtering),
        ],
        "Operators adherence to recommendations": [
            Text(
                text="Adherence to recommendations is measured by the implementation "
                "status. The closer the value is to 1, the better the recommendation "
                "is implemented.",
                left_margin=_TEXT_LEFT_MARGIN,
                max_characters_per_text_line=_MAX_CHARACTERS_OPERATORS_ADHERENCE,
            ),
            (
                charts
                .plot_implementation_ratio(clean_data_filtering)
                .update_layout({"title": "", "margin": {"t": 0}})
            ),
        ],
    }


def _get_impact_deep_dive_section(
    data: pd.DataFrame,
    impact: pd.DataFrame,
    timestamp_column: str,
) -> TRenderingAgnosticDict:
    """
    Generates nested dict with figures for the impact deep dive section.

    Args:
        data: Dataframe with baseline, optimal and actual values
        impact: Dataframe with impact analysis
        timestamp_column: Name of the timestamp column

    Returns:
        Dictionary of impact deep dive figures

    """
    if impact.shape[0] > 1:
        title_table = "Total impact and impact by group"
    else:
        title_table = "Total impact"

    return {
        "Impact figures": [
            Text(
                text="",
                title=title_table,
                title_size=_TEXT_TITLE_SIZE,
                left_margin=_TEXT_LEFT_MARGIN,
            ),
            Table(
                table=impact,
                width=100,
                show_index=False,
                precision=0,
            ),
        ] + charts.plot_impact_waterfall(impact),
        "Impact timeline": [
            charts.plot_impact_timeline(data, timestamp_column),
            charts.plot_impact_timeline_cumulative(data, timestamp_column),
        ],
        "Objective values": (
            charts
            .plot_objective_values(data, timestamp_column)
            .update_layout({"title": ""})
        ),
        "Gap to optimal": (
            charts
            .plot_gap_to_optimal(data, timestamp_column)
            .update_layout({"title": ""})
        ),
    }


def _get_impact_significance_section(
    data: pd.DataFrame,
    baseline_historical_data: pd.DataFrame,
    significance_mean_no_zero: pd.DataFrame,
    significance_uplift_no_model_error: pd.DataFrame,
    significance_baseline_mean_zero: pd.DataFrame,
) -> TRenderingAgnosticDict:
    """
    Generates nested dict with figures for the impact significance section.

    Args:
        data: Dataframe with baseline, optimal and actual values
        baseline_historical_data: Dataframe with baseline predictions and actual values
            over a historical dataset to assess its bias
        significance_mean_no_zero: Dataframe with significance analysis that proves that
            the mean uplift is not zero
        significance_uplift_no_model_error: Dataframe with significance analysis that
            proves that the uplift is not due to model error
        significance_baseline_mean_zero: Dataframe with significance analysis that
            proves that the baseline mean is zero

    Returns:
        Dictionary of impact significance figures

    """
    uplifts = data[data["value_type"].isin(["actual_uplift"])]["value"]
    baseline = baseline_historical_data["error"]
    data_for_bins = np.concatenate((uplifts, baseline))
    bin_width = calculate_optimal_bin_width(
        data_for_bins,
    )

    uplift_and_baseline_test_results = charts.get_text_for_test_results(
        data_test_results=significance_uplift_no_model_error,
        null_hypothesis="Uplift mean is the same as baseline model error mean",
        alternative_hypothesis=(
            f"Uplift mean is {_UPLIFT_DIRECTION} than baseline model error mean"
        ),
        rejection_comment="This is the expected behavior in order to prove impact",
        no_rejection_comment=(
            f"In order to prove impact, the null hypothesis has to be rejected. "
            f"Baseline model error mean should be smaller of uplifts mean should be "
            f"{_UPLIFT_DIRECTION}"
        ),
        result_summary_rejection=(
            f"The hypothesis test shows that the uplift mean is significantly "
            f"{_UPLIFT_DIRECTION} than baseline model errors. Therefore, we can trust "
            f"that the impact is not caused by the baseline model error."
        ),
        result_summary_no_rejection=(
            f"The hypothesis test shows that the uplift mean and the baseline model "
            f"errors mean are similar. However, they should be different to trust the "
            f"impact estimation. Baseline model error mean should be smaller of "
            f"uplifts mean should be {_UPLIFT_DIRECTION}."
        ),
    )

    uplift_test_results = charts.get_text_for_test_results(
        data_test_results=significance_mean_no_zero,
        null_hypothesis="Uplift mean is zero",
        alternative_hypothesis=f"Uplift mean is {_UPLIFT_DIRECTION} than zero",
        rejection_comment="This is the expected behavior in order to prove impact",
        no_rejection_comment=(
            f"In order to prove impact, the null hypothesis has to be rejected. "
            f"Uplifts mean should be {_UPLIFT_DIRECTION}"
        ),
        result_summary_rejection=(
            f"The hypothesis test shows that the uplift mean is significantly "
            f"{_UPLIFT_DIRECTION} than zero. Therefore, we can trust that impact from "
            f"the uplifts is not zero."
        ),
        result_summary_no_rejection=(
            f"The hypothesis test shows that the uplift mean is close to zero. "
            f"However, it should be {_UPLIFT_DIRECTION} than zero to trust the "
            f"impact estimation."
        ),
    )

    baseline_test_results = charts.get_text_for_test_results(
        data_test_results=significance_baseline_mean_zero,
        null_hypothesis="Baseline error mean is zero",
        alternative_hypothesis="Baseline error mean is not zero",
        rejection_comment=(
            "In order to prove impact, the null hypothesis cannot be rejected. "
            "Baseline errors mean should be closer to zero"
        ),
        no_rejection_comment="This is the expected behavior in order to prove impact",
        result_summary_rejection=(
            "The hypothesis test shows that the baseline error mean is significantly "
            "different than zero. However, it should be close to zero to trust the "
            "impact estimation."
        ),
        result_summary_no_rejection=(
            "The hypothesis test shows that the baseline error mean is close to zero. "
            "Therefore, we can trust that the baseline is not introducing a bias in "
            "the impact estimation."
        ),
    )

    return {
        "Uplift and Baseline error": [
            (
                charts
                .plot_uplifts_and_baseline_error_histogram(
                    data,
                    baseline_historical_data,
                    bin_width,
                )
                .update_layout({"margin": {"b": 0}})
            ),
            Text(
                text=uplift_and_baseline_test_results["summary"],
                title="Hypothesis testing results",
                title_size=_TEXT_TITLE_SIZE,
                text_size=_TEXT_SIZE,
                left_margin=_TEXT_LEFT_MARGIN,
                max_characters_per_text_line=_MAX_CHARACTERS_PER_LINE_TESTS,
            ),
            Text(
                text=uplift_and_baseline_test_results["deep_dive"],
                title="Results deep dive",
                title_size=_TEXT_SIZE,
                text_size=_TEXT_SIZE,
                left_margin=_TEXT_LEFT_MARGIN,
                max_characters_per_text_line=_MAX_CHARACTERS_PER_LINE_TESTS,
            ),
            Table(
                table=significance_uplift_no_model_error, width=100, show_index=False,
            ),
        ],
        "Uplift mean": [
            (
                charts
                .plot_uplift_histogram(data, bin_width)
                .update_layout({"margin": {"b": 0}})
            ),
            Text(
                text=uplift_test_results["summary"],
                title="Hypothesis testing results",
                title_size=_TEXT_TITLE_SIZE,
                text_size=_TEXT_SIZE,
                left_margin=_TEXT_LEFT_MARGIN,
                max_characters_per_text_line=_MAX_CHARACTERS_PER_LINE_TESTS,
            ),
            Text(
                text=uplift_test_results["deep_dive"],
                title="Results deep dive",
                title_size=_TEXT_SIZE,
                text_size=_TEXT_SIZE,
                left_margin=_TEXT_LEFT_MARGIN,
                max_characters_per_text_line=_MAX_CHARACTERS_PER_LINE_TESTS,
            ),
            Table(table=significance_mean_no_zero, width=100, show_index=False),
        ],
        "Baseline bias": [
            (
                charts
                .plot_baseline_bias_histogram(baseline_historical_data, bin_width)
                .update_layout({"margin": {"b": 0}})
            ),
            Text(
                text=baseline_test_results["summary"],
                title="Hypothesis testing results",
                title_size=_TEXT_TITLE_SIZE,
                text_size=_TEXT_SIZE,
                left_margin=_TEXT_LEFT_MARGIN,
                max_characters_per_text_line=_MAX_CHARACTERS_PER_LINE_TESTS,
            ),
            Text(
                text=baseline_test_results["deep_dive"],
                title="Results deep dive",
                title_size=_TEXT_SIZE,
                text_size=_TEXT_SIZE,
                left_margin=_TEXT_LEFT_MARGIN,
                max_characters_per_text_line=_MAX_CHARACTERS_PER_LINE_TESTS,
            ),
            Table(table=significance_baseline_mean_zero, width=100, show_index=False),
        ],
    }
