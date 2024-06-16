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

from recommend.solution import Solutions
from reporting.rendering.identifiers import Text

_TITLE_SIZE = 20
_TEXT_SIZE = 18
_LEFT_MARGIN = 50


def plot_period_details(
    solutions: Solutions, timestamp_column: str = "timestamp",
) -> Text:
    """
    Returns a plotly figure with text info about the period

    Args:
        solutions: result of optimization
        timestamp_column: column used as a timestamp in the initial data to optimize
    """
    if not solutions:
        Text(
            title="Empty dataset to optimize provided",
            text="",
            title_size=_TITLE_SIZE,
            text_size=_TEXT_SIZE,
            left_margin=_LEFT_MARGIN,
        )

    solution = next(iter(solutions.values()))
    if timestamp_column not in solution.context_parameters:
        available_columns = solution.context_parameters.keys()
        raise ValueError(
            f"Can't find {timestamp_column = } in solutions' context parameters. "
            f"Available columns: {available_columns}",
        )

    start = min(
        solution.context_parameters[timestamp_column]
        for solution in solutions.values()
    )
    end = max(
        solution.context_parameters[timestamp_column]
        for solution in solutions.values()
    )

    return Text(
        text=f"start: {start}\nend : {end}",
        title="Optimization Timeframe",
        title_size=_TITLE_SIZE,
        text_size=_TEXT_SIZE,
        left_margin=_LEFT_MARGIN,
    )
