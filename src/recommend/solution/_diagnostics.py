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

from matplotlib import pyplot as plt

import optimizer

from ._solution import Solution  # noqa: WPS436

_TLogger = tp.TypeVar("_TLogger")


def _get_logger(
    loggers: tp.List[optimizer.loggers.LoggerMixin], logger_type: tp.Type[_TLogger],
) -> _TLogger:
    for logger_ in loggers:
        if isinstance(logger_, logger_type):
            return logger_
    raise ValueError(
        "Couldn't find `PenaltyLogger` in loggers. "
        "Please provide an instance of this logger to the `optimize` function via "
        "`loggers` argument: optimise(..., loggers=[PenaltyLogger(), ...], ...).",
    )


def plot_best_trajectory_summary(
    solution: Solution,
    normalize_penalties: bool = False,
    show_xaxis_grid: bool = False,
) -> plt.Figure:
    """
    Plots best trajectory summary for the solution.

    Given the `solution`, this function searches for
    `optimizer.loggers.BestTrajectoryLogger` and uses it
    for retrieving the data for the plot.

    The plot is built via
    :py:func:`optimizer.plotting.trajectory_summary_plot_from_artifacts`.
    Reference this function for more details.

    Args:
        solution: a solution that contains `optimizer.loggers.BestTrajectoryLogger` in
            `solution.loggers` attribute
        normalize_penalties: If True, this will make a normalization for each Penalty.
            This might be useful if at some iteration one of the penalties happens to
            be very large and thus the shades in the heatmap towards the end of the
            run get very light. To read it better, it could be interesting to normalize
            these values.
        show_xaxis_grid: shows grid for xaxis on each subplot

    Returns:
        Figure with the best trajectory summary
    """
    best_trajectory_logger = _get_logger(
        solution.loggers, optimizer.loggers.BestTrajectoryLogger,
    )
    control_parameters_names = solution.control_parameters_before
    bounds = [
        solution.controls_domain[control]
        for control in control_parameters_names
    ]
    fig, _ = optimizer.plotting.trajectory_summary_plot_from_artifacts(
        best_trajectory_logger,
        solution.problem,
        bounds,
        normalize_penalties=normalize_penalties,
        variable_labels=control_parameters_names,
        show_xaxis_grid=show_xaxis_grid,
    )
    return tp.cast(plt.Figure, fig)


def plot_convergence_evolution(solution: Solution) -> plt.Figure:
    """
    Plots objective convergence plot for the solution.

    Given the `solution`, this function searches for
    `optimizer.loggers.BasicLogger` and uses it
    for retrieving the data for the plot.

    The plot is built via
    :py:func:`optimizer.plotting.convergence_plot`.
    Reference this function for more details.

    Args:
        solution: a solution that contains `optimizer.loggers.BasicLogger` in
            `solution.loggers` attribute

    Returns:
        Figure with convergence plot
    """
    basic_logger = _get_logger(
        solution.loggers, optimizer.loggers.BasicLogger,
    )
    fig = plt.Figure()
    optimizer.plotting.convergence_plot(basic_logger, ax=fig.gca())
    return fig


def plot_penalties(solution: Solution) -> plt.Figure:
    """
    Plots mean penalty evolution plot for the solution.

    Given the `solution`, this function searches for
    `optimizer.loggers.PenaltyLogger` and uses it
    for retrieving the data for the plot.

    The plot is built via
    :py:func:`optimizer.plotting.penalty_plot`.
    Reference this function for more details.

    Args:
        solution: a solution that contains `optimizer.loggers.PenaltyLogger` in
            `solution.loggers` attribute

    Returns:
        Figure with mean penalty evolution plot
    """
    penalties_logger = _get_logger(
        solution.loggers, optimizer.loggers.PenaltyLogger,
    )
    fig = plt.Figure()
    optimizer.plotting.penalty_plot(penalties_logger.log_records, ax=fig.gca())
    return fig
