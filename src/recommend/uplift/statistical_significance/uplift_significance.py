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
from scipy.stats import ttest_1samp

from ..uplifts import BaselineUplifts
from .utils import drop_uplifts_na_values, select_method

_RANDOM_SEED = 123


def check_uplift_stat_significance(
    uplift: BaselineUplifts,
    alternative_hypothesis: tp.Literal["two-sided", "less", "greater"] = "greater",
    forced_method: tp.Optional[tp.Literal["t-test", "bootstrap"]] = None,
    bootstrap_samples: int = 1000,
    pop_mean: float = 0,
) -> pd.DataFrame:
    """
    Check the statistical significance of uplift by comparing the sample mean against
    zero using a T-test or bootstrapping. If uplift has more than 30 observations or can
    be proved to be normal, a T-test is used. Otherwise, bootstrapping is used.

    This test will conclude average uplift behavior. For example, if we test the
    "greater" hypothesis and the result is statistically significant, we conclude that
    on average, produced uplift is positive.

    Args:
        uplift: Uplifts to check statistical significance for.
        alternative_hypothesis: Alternative hypothesis for the test.
        forced_method: If not None, the indicated method will be used to test
            statistical significance.
        bootstrap_samples: Number of samples to draw from the uplifts.
        pop_mean: Mean of the population under the null hypothesis.

    Returns:
        P-value of the test for all data and by group.

    """
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
                _run_t_test_mean_no_zero(uplift_group, alternative_hypothesis, pop_mean)
            )
        elif method == "bootstrap":
            p_values.loc[p_values["group"] == group, "p_value"] = (
                _run_boostrap_mean_no_zero(
                    uplift_group, alternative_hypothesis, pop_mean, bootstrap_samples,
                )
            )

    return p_values


def _run_t_test_mean_no_zero(
    uplift_data: pd.DataFrame,
    alternative_hypothesis: tp.Literal["two-sided", "less", "greater"],
    pop_mean: float = 0,
) -> float:
    """
    Runs a T-test to check statistical significance.

    Args:
        uplift_data: Uplift data to check statistical significance for.
        alternative_hypothesis: Alternative hypothesis for the test.
        pop_mean: Mean of the population under the null hypothesis.

    Returns:
        P-value of the test.

    """
    p_value = ttest_1samp(
        uplift_data["uplift"], pop_mean, alternative=alternative_hypothesis,
    ).pvalue
    return float(p_value)


def _run_boostrap_mean_no_zero(
    uplift_data: pd.DataFrame,
    alternative_hypothesis: tp.Literal["two-sided", "less", "greater"],
    pop_mean: float = 0,
    bootstrap_samples: int = 1000,
) -> float:
    """
    Runs a bootstrap to check statistical significance.

    Args:
        uplift_data: Uplift data to check statistical significance for.
        alternative_hypothesis: Alternative hypothesis for the test.
        pop_mean: Mean of the population under the null hypothesis.
        bootstrap_samples: Number of samples to draw from the uplifts.

    Returns:
        P-value of the test.

    """
    rand_gen = np.random.default_rng(_RANDOM_SEED)
    means = rand_gen.choice(
        uplift_data["uplift"],
        size=(uplift_data.shape[0], bootstrap_samples),
        replace=True,
    ).mean(axis=0)
    if alternative_hypothesis == "greater":
        return float((means <= pop_mean).mean())
    elif alternative_hypothesis == "less":
        return float((means >= pop_mean).mean())
    elif alternative_hypothesis == "two-sided":
        return float(
            np.minimum((means <= pop_mean).mean(), (means >= pop_mean).mean()) * 2,
        )
    raise ValueError(
        f"Invalid alternative_hypothesis {alternative_hypothesis}. "
        f"Valid options are 'two-sided', 'less' and 'greater'.",
    )
