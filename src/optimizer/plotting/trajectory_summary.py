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
Plots the best solution at each iteration.
"""
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from sklearn.preprocessing import MinMaxScaler

from optimizer import OptimizationProblem
from optimizer.constraint import InequalityConstraint, Penalty
from optimizer.diagnostics import get_penalties_table, get_slack_table
from optimizer.loggers import BestTrajectoryLogger
from optimizer.types import (
    Matrix,
    PenaltyCompatible,
    ReducesMatrixToSeries,
    TBounds,
    TBoundsList,
    Vector,
)

_TITLE_STYLE: tp.Dict[str, tp.Any] = dict(loc="left", fontweight="bold")  # pylint: disable=use-dict-literal
_TPenalties = tp.Sequence[PenaltyCompatible]


def _minmax_bounds_scale(solutions: Matrix, bounds: Matrix) -> Matrix:
    """
    Scales solution variables using the bounds for each of them. Thus, if
    x = max_bound_x, then it will be 1; and if x = min_bound_x the outcome will be 0.
    If x is outside its bounds, then the result will be outside the range [0,1].

    Args:
        solutions: unscaled solution variables
        bounds: Matrix where i-th row contains the lower and upper bound for
            corresponding variable (i-th column in the solution matrix)

    Returns:
        Matrix of shape ``solutions.shape`` with variables scaled by using the bounds
        (if no bound is violated, then the range of this matrix is [0,1])
    """
    bounds_range = bounds.max(axis=1) - bounds.min(axis=1)
    scaled = (solutions - bounds.min(axis=1)) / bounds_range
    return scaled


def _plot_slack(fig: plt.Figure, ax: plt.Axes, penalties: _TPenalties) -> plt.Axes:
    """
    Plots heatmap with color-coded distances for evaluated constraints
    (only Inequality constraints) to the bounds.
    The dark the blue, the smaller the slack / room the constraint has until reaching
    the bound (i.e., the tighter the constraint). Penalties that are not
    InequalityConstraint based will be silently omitted. The rationale is that we might
    receive a problem that has more than one type of penalties, and we don't want to
    alert more than needed.

    Args:
        fig: Figure in which this will be drawn
        ax: Axes on which this will be drawn
        penalties: Penalties from which ``InequalityContraints`` are extracted
            and evaluated

    Returns:
        plt.Axes with the plot on it. This is the same object as the received by
        parameter.

    """
    inequality_penalties = [
        p
        for p in penalties
        if isinstance(p, Penalty) and isinstance(p.constraint, InequalityConstraint)
    ]
    if not inequality_penalties:
        ax.set_title("Slack Values: No inequality constraints", **_TITLE_STYLE)
        return ax
    slack_table = get_slack_table(inequality_penalties, mask_negative_slack=False)
    # `get_slack_table` has a convention of positive value
    # i.e. "amount of space left"; in plot, a slack is a "negative" value,
    # so we negate the produced table and also mask all violated penalties' slack
    slack_table_scaled = MinMaxScaler().fit_transform(
        np.where(slack_table < 0, 0, -slack_table),
    )
    n_records, n_penalties = slack_table_scaled.shape
    extent = _get_img_extent(n_penalties, n_records)
    im = ax.imshow(
        slack_table_scaled.T, aspect="auto", cmap="Blues", vmax=1, extent=extent,
    )
    ax.imshow(
        np.where(slack_table.T < 0, 0.7, np.nan),  # .7 is makes a nice red shade
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=1,
        extent=extent,
    )
    _update_layout(
        ax=ax,
        fig=fig,
        im=im,
        title="Slack Values",
        n_y_ticks=n_penalties,
        y_tick_labels=slack_table.columns,
        color_bar_range=(slack_table_scaled.min(), 1),
        color_bar_limits_labels=("Far from Bound", "Close to Bound"),
    )
    patch = Patch(color="red", label="Violated")
    ax.legend(handles=[patch], loc=(0.91, 1.07), borderaxespad=0.0)
    return ax


def _plot_penalties(
    fig: plt.Figure,
    ax: plt.Axes,
    penalties: _TPenalties,
    normalize_penalties: bool,
) -> plt.Axes:
    """
    Plots heatmap with penalties color-coded. Blue means that penalty was not violated.
    If a penalty was triggered, then the darker the red the bigger the penalty.
    If normalize_penalties is True, the red shades will be scaled independently for each
    penalty.

    Args:
        fig: plt.Figure on which the plot is displayed
        ax: plt.Axes used for the plot
        penalties: List of penalties to evaluate on solutions
        normalize_penalties: set to True for making a relative coloring for each of the
            penalties independently. Note that if True, the plot will will have intense
            reds for penalties that might be penalizing little relative to other
            penalties that are actually damaging our objective value.

    Returns:
        plt.Axes with the plot on it. This is the same object as the received as
        parameter

    """
    if len(penalties) == 0:
        ax.set_title(
            "Penalty constraint violations: No penalties available", **_TITLE_STYLE,
        )
        return ax
    penalties_table = get_penalties_table(penalties)
    penalty_values = (
        MinMaxScaler().fit_transform(penalties_table)
        if normalize_penalties
        else penalties_table.values
    )
    n_records, n_penalties = penalty_values.shape
    extent = _get_img_extent(n_penalties, n_records)
    im = ax.imshow(penalty_values.T, aspect="auto", cmap="Reds", extent=extent)
    ax.imshow(
        np.where(penalty_values.T == 0, 0.7, np.nan),  # .7 makes a nice blue shade
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
        extent=extent,
    )
    _update_layout(
        ax=ax,
        fig=fig,
        im=im,
        title="Penalty constraint violations",
        n_y_ticks=n_penalties,
        y_tick_labels=penalties_table.columns,
        color_bar_range=(penalty_values.min(), penalty_values.max()),
        color_bar_limits_labels=("Low", "High"),
    )
    patch = Patch(color="tab:blue", label="Non Violated")
    ax.legend(handles=[patch], loc=(0.91, 1.07), borderaxespad=0.0)
    return ax


def _plot_convergence(ax: plt.Axes, objective_values: Vector) -> plt.Axes:
    """Plots a time series with the objective values.

    Args:
        ax: Axes on which the time series will be plotted
        objective_values: Objective function values

    Returns:
        plt.Axes with the plot on it. This is the same object as the received as
        parameter

    """
    n_records = len(objective_values)
    ax.plot(np.arange(1, n_records + 1), objective_values, drawstyle="steps-post")
    ax.set_title("Convergence Plot", **_TITLE_STYLE)
    ax.set_ylabel("Objective Value")
    return ax


def _plot_solutions(
    fig: plt.Figure,
    ax: plt.Axes,
    solutions: Matrix,
    bounds: Matrix,
    variable_labels: tp.Optional[tp.Iterable[str]] = None,
) -> plt.Axes:
    """
    Plots heatmap with variable values color coded: the darker the color, the closer
    the value to it's upper bound.
    If the value for some variable is out of bounds, then it will be colored by red or
    blue in the heatmap.

    Args:
        fig: Figure on which this will be plotted
        ax: Axes on which this will be plotted
        solutions: Matrix where each row is a solution to be plotted
        bounds: Matrix where each row i contains the bounds for variable i (ith column
            on the solutions matrix)
        variable_labels: Labels to display on the left of the heatmap. If not provided,
            the label will be the index

    Returns:
        plt.Axes with the plot on it. This is the same object as the received as
        parameter

    """
    y = _minmax_bounds_scale(solutions, bounds)
    im = _imshow_solutions(ax, y)
    _update_layout(
        ax=ax,
        fig=fig,
        im=im,
        title="Optimized Variables",
        n_y_ticks=y.shape[1],
        y_tick_labels=variable_labels,
        color_bar_range=(0, 1),
        color_bar_limits_labels=("Lower Bound", "Upper Bound"),
    )
    patches = [
        Patch(color="red", label="Above bound"),
        Patch(color="tab:blue", label="Below bound"),
    ]
    ax.legend(handles=patches, loc=(0.91, 1.07), borderaxespad=0.0)
    return ax


def _imshow_solutions(ax: plt.Axes, y: Vector) -> AxesImage:
    n_records, n_penalties = y.shape
    extent = _get_img_extent(n_penalties, n_records)
    im = ax.imshow(
        y.T,
        aspect="auto",
        cmap="Greens",
        vmin=0,
        vmax=1,
        extent=extent,
    )
    ax.imshow(
        np.where(y.T > 1, 0.7, np.nan),  # .7 makes a nice shade of red
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=1,
        extent=extent,
    )
    ax.imshow(
        np.where(y.T < 0, 0.7, np.nan),  # .7 makes a nice shade of blue
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
        extent=extent,
    )
    return im


def _estimate_plot_size(
    solutions: Matrix,
    penalties: _TPenalties,
    height_scale: float = 0.5,
    time_series_size: float = 7,
) -> tp.Tuple[float, tp.List[float]]:
    """Estimate the total height and the ratios of the axes for plotting. It parses the
    input and does the estimation based on the number of variables, penalties and the
    number of penalties with slack. If you wish to increment/reduce the plot size, pass
    a different height_scale factor.

    Args:
        solutions: matrix containing the solutions
        penalties: list of problem's penalties used to produce each solution
        height_scale: multiplier of the height
        time_series_size: factor that indicates how many heatmap row heights contains
            the time series plot (approx.) Default value is 7, that it, the time series
            height is equivalent to 7 heatmap rows.

    Returns:
        A tuple (total_height, height_ratios), where total_height is an estimation of
        the figure total height, while the height_ratios is a list that's useful as
        height_ratios entry in matplotlib's gridspec_kw dictionary. The ratios are in
        the following order: [convergence time series, variables heatmap,
        slack heatmap, penalties heatmap]
    """
    n_inequality_penalties = len(
        [
            p for p in penalties
            if isinstance(p, Penalty) and isinstance(p.constraint, InequalityConstraint)
        ]
    )
    n_penalty_compatible = len(
        [p for p in penalties if isinstance(p, PenaltyCompatible)]
    )
    sizes = [
        time_series_size,
        max(solutions.shape[1], 2),
        max(n_inequality_penalties, 2),
        max(n_penalty_compatible, 2),
    ]
    return sum(sizes) * height_scale, sizes


def trajectory_summary_plot_from_artifacts(
    best_trajectory_logger: BestTrajectoryLogger,
    problem: OptimizationProblem,
    bounds: TBoundsList,
    normalize_penalties: bool = False,
    variable_labels: tp.Optional[tp.Iterable[str]] = None,
    show_xaxis_grid: bool = False,
) -> tp.Tuple[plt.Figure, tp.List[plt.Axes]]:
    """
    Generates a plot on how the optimization was executed, containing 4 charts:
        - The first chart is the objective value as function of the iteration in logger
        - The second captures how far from its bounds were the optimizable varibles in
          a heatmap. The darker the shade, the closer the value to the upper bound.
        - The third captures the distance of each constraint to it's bound (inequality
          penalty). The darker the shade, the closer the constraint was to the bound
          –– in other words, the tighter the constraint. If the bound is violated, then
          the occurrence is colored in red.
        - The fourth captures the impact of the penalties on the objective value. The
          darker the shade, the more this penalty contributed to the objective value.
          If the penalty was not triggered (value = 0), then the occurrence is
          colored in blue.

    Args:
        best_trajectory_logger: The logger that logged during the optimization
            execution.
        problem: A StatefulOptimizationProblem. This plotting function will use the
            problem penalties to build the plots.
        bounds: A list of (min_bound, max_bound) for each of the variables. This has to
            be in the same order as the solutions columns.
        normalize_penalties: If True, this will make a normalization for each Penalty.
            This might be useful if at some iteration one of the penalties happens to
            be very large and thus the shades in the heatmap towards the end of the
            run get very light. To read it better, it could be interesting to normalize
            these values.
        variable_labels: labels list for adding them on the y axis of the variable vs.
            their bounds plot.
        show_xaxis_grid: shows grid for xaxis on each subplot

    Returns:
        A tuple (plt.Figure, List[plt.Axes]) with the plots.
    """
    problem(best_trajectory_logger.solutions)  # updates the penalties' states
    return trajectory_summary_plot(
        best_trajectory_logger.solutions,
        best_trajectory_logger.objective_values,
        bounds,
        problem.penalties,
        variable_labels=variable_labels,
        normalize_penalties=normalize_penalties,
        show_xaxis_grid=show_xaxis_grid,
    )


def trajectory_summary_plot(
    solutions: Matrix,
    objective_values: Vector,
    bounds: TBoundsList,
    penalties: _TPenalties,
    normalize_penalties: bool = False,
    variable_labels: tp.Optional[tp.Iterable[str]] = None,
    show_xaxis_grid: bool = False,
) -> tp.Tuple[plt.Figure, tp.List[plt.Axes]]:
    """
    Generates a plot on how the optimization was executed, containing 4 charts:
        - The first chart is the objective value as function of the iteration in logger
        - The second captures how far from its bounds were the optimizable variables
          in a heatmap. The darker the shade, the closer the value to the upper bound.
        - The third captures the distance of each constraint to it's bound (inequality
          penalty). The darker the shade, the closer the constraint was to the bound
          –– in other words, the tighter the constraint. If the bound is violated,
          then the occurrence is colored in red.
        - The fourth captures the impact of the penalties on the objective value. The
          darker the shade, the more this penalty contributed to the objective value. If
          the penalty was not triggered (value = 0), then the occurrence is colored in
          blue.

    Args:
        solutions: Solutions that will be plotted in the optimized variables heatmap
            (2nd plot)
        objective_values: Values that will appear in the convergence plot
        penalties: List of penalties PRE-EXECUTED on solutions (since we have only
            optimal parameters in our solutions, that won't include state
            for ContextfulOptimizationProblem, we ask users to evaluate penalties
            on their own using the problem's state if needed)
        bounds: A list of (min_bound, max_bound) for each of the variables.
            This has to be in the same order as the solutions' columns.
        normalize_penalties: If True, this will make a normalization for each Penalty.
            This might be useful if at some iteration one of the penalties happens to be
            very large and thus the shades in the heatmap towards the end of the run get
            very light. To read it better, it could be interesting to normalize these
            values.
        variable_labels: labels list for adding them on the y axis of the variable vs.
            their bounds plot.
        show_xaxis_grid: shows grid for xaxis on each subplot

    Returns:
        A tuple (plt.Figure, List[plt.Axes]) with the plots.
    """
    if any(not isinstance(bound, tuple) for bound in bounds):
        raise NotImplementedError("Can't display discrete solvers at this point")

    height, height_ratios = _estimate_plot_size(solutions, penalties)

    fig, axes = plt.subplots(
        figsize=(15, height),
        nrows=4,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    _plot_convergence(axes[0], objective_values)
    _plot_solutions(fig, axes[1], solutions, np.stack(bounds), variable_labels)
    _plot_slack(fig, axes[2], penalties)
    _plot_penalties(fig, axes[3], penalties, normalize_penalties)

    _update_xaxis(axes, len(solutions), show_xaxis_grid)

    return fig, axes


def _update_xaxis(
    axes: tp.List[plt.Axes], n_records: int, show_xaxis_grid: bool,
) -> None:
    axes[-1].set_xlabel("Iteration", loc="left")
    locs = np.linspace(start=1, stop=n_records, num=21, dtype=int)
    axes[-1].xaxis.set_major_locator(FixedLocator(locs.tolist()))
    axes[-1].set_xlim(0.5, n_records + 0.5)
    if show_xaxis_grid:
        colors = ("lightgrey", "black", "black", "black")
        for ax, color in zip(axes, colors):
            ax.xaxis.grid(True, color=color)


def _update_layout(
    ax: plt.Axes,
    fig: plt.Figure,
    im: AxesImage,
    title: str,
    n_y_ticks: int,
    y_tick_labels: tp.Optional[tp.Iterable[str]],
    color_bar_range: TBounds,
    color_bar_limits_labels: tp.Tuple[str, str],
) -> None:
    """
    Update ax's layout:
        * set title
        * set y ticks & labels
        * add color bar
    """
    ax.set_title(title, **_TITLE_STYLE)
    ax.set_yticks(range(n_y_ticks))
    if y_tick_labels is not None:
        ax.set_yticklabels(list(y_tick_labels))
    cax = ax.inset_axes((1.01, 0.0, 0.01, 1.0))
    cbar = fig.colorbar(im, cax=cax, ticks=color_bar_range, orientation="vertical")
    cbar.ax.set_yticklabels(list(color_bar_limits_labels))


def _get_img_extent(
    n_penalties: int,
    n_records: int,
) -> tp.Tuple[float, float, float, float]:
    """Returns extent such that we show image starting from x=1"""
    return 0.5, n_records + 0.5, n_penalties - 0.5, -0.5
