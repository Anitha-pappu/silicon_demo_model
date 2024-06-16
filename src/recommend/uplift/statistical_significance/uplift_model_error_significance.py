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

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats

from ..uplifts import BaselineUplifts
from .utils import drop_uplifts_na_values, select_method

_RANDOM_SEED = 123


def check_uplift_model_error_stat_significance(
    uplift: BaselineUplifts,
    baseline_errors_mean: float,
    baseline_errors_std: float,
    baseline_errors_n: int,
    baseline_granularity: tp.Optional[str] = None,
    alternative_hypothesis: tp.Literal["two-sided", "less", "greater"] = "greater",
    forced_method: tp.Optional[tp.Literal["t-test", "bootstrap"]] = None,
    bootstrap_samples: int = 1000,
) -> pd.DataFrame:
    """
    Checks whether the uplift is caused by model errors using a T-test or bootstrapping.
    It tests whether the uplift values are statistically significantly different from
    the model errors or are just a result of the model error distribution. If uplift has
    more than 30 observations or can be proved to be normal, a T-test is used.
    Otherwise, bootstrapping is used.

    Baseline model results are assumed to be at the same granularity as uplifts.

    This test will conclude average behavior. For example, if we test the "greater"
    hypothesis and the result is statistically significant, we conclude that on average,
    produced uplifts are greater than the model error and not a consequence of them.

    Args:
        uplift: Uplifts to check statistical significance for.
        baseline_errors_mean: Mean of the baseline errors.
        baseline_errors_std: Standard deviation of the baseline errors.
        baseline_errors_n: Number of observations of the baseline errors.
        baseline_granularity: Granularity of the baseline errors. If None, the
            original granularity of the uplifts will be used.
        alternative_hypothesis: Alternative hypothesis for the test.
        forced_method: If not None, the indicated method will be used to test
            statistical significance.
        bootstrap_samples: Number of samples to draw from the uplifts.

    Returns:
        P-value of the test for all data and by group.

    """
    baseline_granularity = baseline_granularity or uplift.original_granularity
    baseline_granularity = pd.Timedelta(baseline_granularity)
    baseline_agg_factor = uplift.agg_granularity / baseline_granularity

    uplift_data = drop_uplifts_na_values(uplift.data)
    method = select_method(uplift_data, forced_method)

    p_values = pd.DataFrame(
        {
            "group": uplift.group_names,
            "p_value": np.nan,
        },
    )

    for group, uplift_group in uplift.group_iterator(dropna=True):
        if method == "t-test":
            p_values.loc[p_values["group"] == group, "p_value"] = (
                _run_t_test_uplift_no_model_error(
                    uplift_group,
                    baseline_errors_mean * baseline_agg_factor,
                    baseline_errors_std * np.sqrt(baseline_agg_factor),
                    baseline_errors_n,
                    alternative_hypothesis,
                )
            )
        elif method == "bootstrap":
            p_values.loc[p_values["group"] == group, "p_value"] = (
                _run_boostrap_uplift_no_model_error(
                    uplift_group,
                    baseline_errors_mean * baseline_agg_factor,
                    baseline_errors_std * np.sqrt(baseline_agg_factor),
                    baseline_errors_n,
                    alternative_hypothesis,
                    bootstrap_samples,
                )
            )

    return p_values


def _run_t_test_uplift_no_model_error(
    uplift_data: pd.DataFrame,
    baseline_errors_mean: float,
    baseline_errors_std: float,
    baseline_errors_n: int,
    alternative_hypothesis: tp.Literal["two-sided", "less", "greater"],
) -> float:
    """
    Runs a T-test to check statistical significance.

    Args:
        uplift_data: Uplift data to check statistical significance for.
        baseline_errors_mean: Mean of the baseline errors.
        baseline_errors_std: Standard deviation of the baseline errors.
        baseline_errors_n: Number of observations of the baseline errors.
        alternative_hypothesis: Alternative hypothesis for the test.

    Returns:
        P-value of the test.

    """
    uplift_col_data = uplift_data["uplift"]
    p_value = ttest_ind_from_stats(
        uplift_col_data.mean(),
        uplift_col_data.std(),
        uplift_col_data.count(),
        baseline_errors_mean,
        baseline_errors_std,
        baseline_errors_n,
        alternative=alternative_hypothesis,
    ).pvalue
    return float(p_value)


def _run_boostrap_uplift_no_model_error(
    uplift_data: pd.DataFrame,
    baseline_errors_mean: float,
    baseline_errors_std: float,
    baseline_errors_n: int,
    alternative_hypothesis: tp.Literal["two-sided", "less", "greater"],
    bootstrap_samples: int = 1000,
) -> float:
    """
    Runs a bootstrap to check statistical significance.

    Args:
        uplift_data: Uplift data to check statistical significance for.
        baseline_errors_mean: Mean of the baseline errors.
        baseline_errors_std: Standard deviation of the baseline errors.
        baseline_errors_n: Number of observations of the baseline errors.
        alternative_hypothesis: Alternative hypothesis for the test.
        bootstrap_samples: Number of samples to draw from the uplifts.

    Returns:
        P-value of the test.

    """
    uplift_n = uplift_data.shape[0]
    original_mean_diff = uplift_data["uplift"].mean() - baseline_errors_mean
    mean_diffs = np.array([])
    rand_gen = np.random.default_rng(_RANDOM_SEED)
    for _ in range(bootstrap_samples):
        baseline_errors = rand_gen.normal(
            baseline_errors_mean, baseline_errors_std, baseline_errors_n,
        )
        total_errors = np.append(np.array(uplift_data["uplift"]), baseline_errors)
        sample_uplifts = rand_gen.choice(total_errors, size=uplift_n, replace=True)
        sample_baseline = rand_gen.choice(
            baseline_errors, size=baseline_errors_n, replace=True,
        )
        mean_diffs = np.append(
            mean_diffs,
            sample_uplifts.mean() - sample_baseline.mean(),
        )

    if alternative_hypothesis == "greater":
        return float((mean_diffs >= original_mean_diff).mean())
    elif alternative_hypothesis == "less":
        return float((mean_diffs <= original_mean_diff).mean())
    elif alternative_hypothesis == "two-sided":
        return float((np.abs(mean_diffs) >= np.abs(original_mean_diff)).mean())
    raise ValueError(
        f"Invalid alternative_hypothesis {alternative_hypothesis}. "
        f"Valid options are 'two-sided', 'less' and 'greater'.",
    )
