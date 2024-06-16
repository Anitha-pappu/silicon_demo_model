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
Convergence plots.
"""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from optimizer.loggers import BasicLogger


def convergence_plot(logger: BasicLogger, ax: tp.Optional[plt.Axes] = None) -> plt.Axes:
    """
    Simple convergence plot of the given values.

    Args:
        logger: basic logger that tracked convergence progress
        ax: Axes, optional axes.

    Returns:
        plt.Axes, Axes objects of the plot.

    Raises:
        ValueError: if no values or BasicLogger are provided.
    """

    ax = ax or plt.gca()

    x_axis_ticks = np.arange(1, len(logger.best_seen_objective) + 1)
    ax.plot(
        x_axis_ticks,
        logger.best_seen_objective,
        color="green",
        label="Best Seen Objective",
    )
    median_objective = logger.median_objective
    ax.plot(x_axis_ticks, median_objective, color="grey", label="Median Objective")
    ax.fill_between(
        x_axis_ticks,
        logger.first_quartile_objective,
        logger.third_quartile_objective,
        alpha=0.2,
        color="grey",  # C1
    )

    ax.spines[['right', 'top']].set_visible(False)
    if len(x_axis_ticks):
        ax.set_xlim(min(x_axis_ticks), max(x_axis_ticks))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.legend()
    return ax
