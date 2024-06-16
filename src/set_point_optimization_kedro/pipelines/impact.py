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

from recommend import (
    BaselineUplifts,
    check_uplift_model_error_stat_significance,
    check_uplift_stat_significance,
    get_impact_estimation,
    get_value_after_recs_counterfactual,
    get_value_after_recs_impact,
)
from recommend.report import get_impact_overview

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_impact_steps() -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                get_value_after_recs_counterfactual,
                inputs={
                    "counterfactual_type": "params:impact.counterfactual_type",
                    "solutions": "solutions",
                    "data": "test_data",
                    "actual_value": "params:impact.target_col",
                    "datetime_column": "params:impact.datetime_col",
                },
                outputs="value_after_recs_opt",
                name="get_value_after_recs_opt",
            ),
            node(
                get_value_after_recs_impact,
                inputs={
                    "data": "test_data",
                    "value_after_recs": "params:impact.target_col",
                    "datetime_column": "params:impact.datetime_col",
                },
                outputs="value_after_recs_act",
                name="get_value_after_recs_act",
            ),
            node(
                BaselineUplifts,
                inputs={
                    "baseline_data": "test_data_with_baseline_predictions",
                    "after_implementation_data": "value_after_recs_act",
                    "baseline_column": "params:impact.baseline_col",
                    "after_implementation_column": "params:impact.after_imp_col",
                    "datetime_column": "params:impact.datetime_col",
                    "original_granularity": "params:impact.uplifts_original_gran",
                    "group_characteristics": "params:impact.group_characteristics",
                    "default_group": "params:impact.default_group",
                    "agg_granularity": "params:impact.agg_granularity",
                    "agg_granularity_function": "params:impact.agg_granularity_func",
                    "agg_granularity_method": "params:impact.agg_granularity_method",
                },
                outputs="uplifts",
                name="get_uplifts",
            ),
            node(
                lambda data, datetime_column, baseline_column: (
                    data[[datetime_column, baseline_column]]
                ),
                inputs={
                    "data": "baseline_test_data_with_baseline_predictions",
                    "datetime_column": "params:impact.datetime_col",
                    "baseline_column": "params:impact.baseline_col",
                },
                outputs="baseline_test_data_only_predictions",
                name="baseline_test_data_only_predictions",
            ),
            node(
                BaselineUplifts,
                inputs={
                    "baseline_data": "baseline_test_data_only_predictions",
                    "after_implementation_data": "baseline_test_data",
                    "baseline_column": "params:impact.baseline_col",
                    "after_implementation_column": "params:impact.target_col",
                    "datetime_column": "params:impact.datetime_col",
                    "original_granularity": "params:impact.baseline_original_gran",
                    "group_characteristics": "params:impact.group_characteristics",
                    "default_group": "params:impact.default_group",
                    "agg_granularity": "params:impact.agg_granularity",
                    "agg_granularity_function": "params:impact.agg_granularity_func",
                    "agg_granularity_method": "params:impact.agg_granularity_method",
                },
                outputs="baseline_error",
                name="get_baseline_error",
            ),
            node(
                BaselineUplifts,
                inputs={
                    "baseline_data": "test_data_with_predictions",
                    "after_implementation_data": "value_after_recs_opt",
                    "baseline_column": "params:impact.baseline_col",
                    "after_implementation_column": "params:impact.after_imp_col",
                    "datetime_column": "params:impact.datetime_col",
                    "original_granularity": "params:impact.uplifts_original_gran",
                    "group_characteristics": "params:impact.group_characteristics",
                    "default_group": "params:impact.default_group",
                    "agg_granularity": "params:impact.agg_granularity",
                    "agg_granularity_function": "params:impact.agg_granularity_func",
                    "agg_granularity_method": "params:impact.agg_granularity_method",
                },
                outputs="optimization_uplifts",
                name="get_optimization_uplifts",
            ),
            node(
                check_uplift_stat_significance,
                inputs={
                    "uplift": "baseline_error",
                    "alternative_hypothesis": "params:impact.baseline_alt",
                },
                outputs="sig_baseline_error",
                name="check_baseline_error_significance",
            ),
            node(
                check_uplift_stat_significance,
                inputs={
                    "uplift": "uplifts",
                    "alternative_hypothesis": "params:impact.uplifts_alt",
                },
                outputs="sig_uplift",
                name="check_uplift_significance",
            ),
            node(
                lambda baseline_test_data: baseline_test_data["error"].mean(),
                inputs="baseline_test_data_with_baseline_predictions",
                outputs="baseline_errors_mean",
                name="baseline_errors_mean",
            ),
            node(
                lambda baseline_test_data: baseline_test_data["error"].std(),
                inputs="baseline_test_data_with_baseline_predictions",
                outputs="baseline_errors_std",
                name="baseline_errors_std",
            ),
            node(
                lambda baseline_test_data: len(baseline_test_data["error"]),
                inputs="baseline_test_data_with_baseline_predictions",
                outputs="baseline_errors_n",
                name="baseline_errors_n",
            ),
            node(
                check_uplift_model_error_stat_significance,
                inputs={
                    "uplift": "uplifts",
                    "baseline_errors_mean": "baseline_errors_mean",
                    "baseline_errors_std": "baseline_errors_std",
                    "baseline_errors_n": "baseline_errors_n",
                    "alternative_hypothesis": "params:impact.baseline_uplift_alt",
                },
                outputs="sig_uplift_no_model_error",
                name="check_uplift_model_error_significance",
            ),
            node(
                get_impact_estimation,
                inputs={
                    "uplifts": "uplifts",
                    "annualize": "params:impact.annualize_impact",
                    "timestamp_col": "params:impact.datetime_col",
                },
                outputs="impact",
                name="get_impact_estimation",
            ),
            node(
                lambda uplifts: uplifts.data,
                inputs="uplifts",
                outputs="uplifts_data",
                name="uplifts_data",
            ),

            node(
                lambda optimization_uplifts: optimization_uplifts.data,
                inputs="optimization_uplifts",
                outputs="optimization_uplifts_data",
                name="optimization_uplifts_data",
            ),
            node(
                get_impact_overview,
                inputs={
                    "baseline_data": "test_data_with_predictions",
                    "baseline_historical_data":
                        "baseline_test_data_with_baseline_predictions",
                    "optimized_uplift": "optimization_uplifts_data",
                    "actual_uplift": "uplifts_data",
                    "significance_mean_no_zero": "sig_uplift",
                    "significance_uplift_no_model_error": "sig_uplift_no_model_error",
                    "significance_baseline_mean_zero": "sig_baseline_error",
                    "impact": "impact",
                    "timestamp_column": "params:impact.datetime_col",
                    "baseline_prediction_column": "params:impact.baseline_col",
                    "baseline_target_column": "params:impact.target_col",
                },
                outputs="impact_report",
                name="get_impact_overview",
            ),
        ],
    ).tag("impact")
