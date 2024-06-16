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
Constraint related plots.
"""

import typing as tp
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class TPenaltyLogRecord(tp.TypedDict):
    value: tp.List[float]
    penalty: tp.List[float]
    lower_bound: tp.Optional[tp.List[float]]
    upper_bound: tp.Optional[tp.List[float]]


TAggFunction = tp.Callable[[tp.List[float]], float]
TLogRecords = tp.List[tp.Dict[str, TPenaltyLogRecord]]


def penalty_plot(
    penalty_log_records: TLogRecords,
    ax: tp.Optional[plt.Axes] = None,
    normalize_penalties: bool = True,
    agg: TAggFunction = np.mean,
) -> plt.Axes:
    """
    Show the calculated penalty values over time.

    Args:
        penalty_log_records: list of by penalty records;
            reference ``loggers.penalty.TPenaltyLogRecord`` for more details
            on record format
        ax: optional axes.
        normalize_penalties: converts penalty values to 0 to 1 if True.
        agg: aggregation function to use per a penalty's values.

    Returns:
        plt.Axes, Axes objects of the plot.

    """
    ax = ax or plt.gca()

    x_axis_ticks = np.arange(1, len(penalty_log_records) + 1)
    penalty_means = _aggregate_penalty_values(agg, penalty_log_records)
    for constraint_name in sorted(penalty_means):
        per_constraint_means = np.array(penalty_means[constraint_name])
        if normalize_penalties:
            per_constraint_means = (
                MinMaxScaler().fit_transform(per_constraint_means[:, np.newaxis])
            )
        agg_function_name = agg.__name__.title()
        ax.plot(
            x_axis_ticks,
            per_constraint_means,
            label=f"{agg_function_name} {constraint_name}",
        )

    ax.spines[['right', 'top']].set_visible(False)
    if len(x_axis_ticks):
        ax.set_xlim(min(x_axis_ticks), max(x_axis_ticks))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        "Penalty Value (Normalized)"
        if normalize_penalties
        else "Penalty Value"
    )
    ax.legend()
    return ax


def _aggregate_penalty_values(
    agg: TAggFunction, penalty_log_records: TLogRecords,
) -> tp.Dict[str, tp.List[float]]:
    penalty_means = defaultdict(list)
    for record in penalty_log_records:
        for penalty_name, penalty_record in record.items():
            penalty_values = penalty_record["penalty"]
            penalty_means[penalty_name].append(agg(penalty_values))
    return dict(penalty_means)
