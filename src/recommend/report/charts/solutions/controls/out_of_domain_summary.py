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

_TExpectedDomain = tp.Dict[str, tp.Tuple[float, float]]
_TOTAL_NOT_NULL = "total_not_null"
_OUTSIDE_OF_DOMAIN_PERC = "outside_of_domain_perc"
_OUTSIDE_OF_DOMAIN = "outside_of_domain"


def get_out_of_domain_summary(
    df_before_after: pd.DataFrame,
    initial_column: str,
    optimized_column: str,
    expected_domain: _TExpectedDomain,
) -> pd.DataFrame:
    """
    Calculates number of cases when control is outside its domain
    (nan controls are dropped).

    Args:
        df_before_after: dataframe with two columns before the optimization and after
            for controls; typically generated via ``Solutions.to_frame``
        initial_column: column level name with initial values
        optimized_column: column level name with optimized values
        expected_domain: mapping from control name into its expected domain

    Returns:

    """

    if df_before_after.empty:
        return pd.DataFrame()

    df_res = _create_ood_dataframe(
        df_before_after, initial_column, optimized_column, expected_domain,
    )
    total_not_null = (
        df_before_after.reorder_levels([1, 0], axis=1)
        [initial_column][list(expected_domain)]
        .notnull().sum()
    )
    _add_percentage_percentage_columns(
        df_res, initial_column, optimized_column, total_not_null,
    )
    # add column in the very end to avoid column reorder hassle
    df_res[_TOTAL_NOT_NULL] = total_not_null
    return df_res


def _add_percentage_percentage_columns(
    df_res: pd.DataFrame,
    initial_column: str,
    optimized_column: str,
    total_not_null: pd.Series,
) -> pd.DataFrame:
    for type_ in (initial_column, optimized_column):
        df_res[(_OUTSIDE_OF_DOMAIN_PERC, type_)] = (
            df_res[(_OUTSIDE_OF_DOMAIN, type_)] / total_not_null * 100
        )
    return df_res


def _create_ood_dataframe(
    df_before_after: pd.DataFrame,
    initial_column: str,
    optimized_column: str,
    expected_domain: _TExpectedDomain,
) -> pd.DataFrame:
    """
    Produces out-of-domain dataframe based on the before-after dataframe

    The output dataset is indexed by ``expected_domain.keys()``, and columns are::

        [(     'outside_of_domain',   'initial'),
         (     'outside_of_domain', 'optimized'),
         outside_of_domain_perc',   'initial'),
         ('outside_of_domain_perc', 'optimized'),
         (        'total_not_null',          '')]
    """
    by_control_stats = {}
    for control, domain in expected_domain.items():
        for type_ in (initial_column, optimized_column):
            key = (control, type_)
            oob_count = (~df_before_after[key].dropna().between(*domain)).sum()
            by_control_stats[key] = {_OUTSIDE_OF_DOMAIN: oob_count}
    return pd.DataFrame(by_control_stats).T.astype(int).unstack()
