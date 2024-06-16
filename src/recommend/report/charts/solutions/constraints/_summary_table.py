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

import numpy as np
import pandas as pd

from optimizer.diagnostics import get_penalties_table
from recommend.solution import Solutions

TOTAL_VIOLATIONS = "total_violations"
TOTAL_PENALTY = "total_penalty"


def get_penalty_summaries(
    solutions: Solutions,
    timestamp_column: str | None = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produces two penalty summaries:
    * by index summary (includes all penalty values for given index
      + total penalty and total violations count)
    * grouped by penalty stats (only violations count for now)

    Args:
        solutions: solutions object
        timestamp_column: if this column is not None, then it's used as index
            in the resulting ``by_index_summary``

    Returns: tuple of two dataframes: (by index summary, by penalty summary)
    """
    df_by_index_penalties = _get_by_index_penalties(solutions, timestamp_column)
    df_by_penalty_violation_count = (
        (df_by_index_penalties > 0)
        .sum(axis=0)
        .to_frame(TOTAL_VIOLATIONS)
    )

    total_penalty = df_by_index_penalties.sum(axis=1)
    n_penalties_violated = (df_by_index_penalties > 0).sum(axis=1)
    # we first calculate stats and only then add new columns to the table
    original_columns = df_by_index_penalties.columns
    df_by_index_penalties[TOTAL_PENALTY] = total_penalty
    df_by_index_penalties[TOTAL_VIOLATIONS] = n_penalties_violated
    df_by_index_penalties = df_by_index_penalties.reindex(
        columns=[TOTAL_VIOLATIONS, TOTAL_PENALTY, *original_columns],
    )

    return df_by_index_penalties, df_by_penalty_violation_count


def _get_by_index_penalties(
    solutions: Solutions, timestamp_column: str | None,
) -> pd.DataFrame:
    by_index_penalties = []
    for index, solution in solutions.items():
        # update penalties' states
        solution.problem(
            parameters=np.array(
                [
                    solution.control_parameters_after[column]
                    for column in solution.problem.optimizable_columns
                ],
            ).reshape(1, -1),
        )

        penalties_table = get_penalties_table(solution.problem.penalties)
        penalties_table.index = (
            pd.Index([index], name="index")
            if timestamp_column is None
            else pd.Index(
                [solution.context_parameters[timestamp_column]],
                name=timestamp_column,
            )
        )
        by_index_penalties.append(penalties_table)
    return pd.concat(by_index_penalties)
