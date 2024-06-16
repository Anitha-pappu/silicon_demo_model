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

import pandas as pd
from scipy.stats import shapiro

_THRESHOLD_OBS_T_TEST = 30
_SIGNIFICANCE_LEVEL_NORMAL_TEST = 0.05


def drop_uplifts_na_values(
    uplift_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drops NA values from uplifts.

    Args:
        uplift_data: Uplift data to check statistical significance for.

    Returns:
        Uplifts without NA values.

    """
    return uplift_data.dropna(subset=["uplift"])


def select_method(
    uplift_data: pd.DataFrame,
    forced_method: tp.Optional[tp.Literal["t-test", "bootstrap"]] = None,
) -> tp.Literal["t-test", "bootstrap"]:
    """
    Selects the method to use to test statistical significance.

    Args:
        uplift_data: Uplift data to check statistical significance for.
        forced_method: If not None, the indicated method will be used to test
            statistical significance.

    Returns:
        Method to use to test statistical significance.

    """
    if forced_method is None:
        if _check_criteria_for_t_test(uplift_data):
            return "t-test"
        return "bootstrap"
    elif forced_method in {"t-test", "bootstrap"}:
        return forced_method
    raise ValueError(
        f"Invalid forced method {forced_method}. "
        f"Valid options are 't-test' and 'bootstrap'.",
    )


def _check_criteria_for_t_test(
    uplift_data: pd.DataFrame,
) -> bool:
    """
    Checks whether the uplift can be tested using a T-test. Criteria are:
        - 30 observations or more due to the central limit theorem
        - Less than 30 observations but normal distribution

    Args:
        uplift_data: Uplift data to check statistical significance for.

    Returns:
        True if the uplift can be tested using a T-test, False otherwise.

    """
    uplift_col_data = uplift_data["uplift"]
    more_threshold_obs = bool(uplift_col_data.count() >= _THRESHOLD_OBS_T_TEST)
    normal_dist = bool(
        shapiro(uplift_col_data).pvalue > _SIGNIFICANCE_LEVEL_NORMAL_TEST,
    )

    return more_threshold_obs or normal_dist
