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


import logging
import typing as tp
import warnings

import numpy as np

from optimizer import Repair, SetRepair

N_POINTS_IN_GRID_BETWEEN_BOUNDS = 15
MAX_POINT_IN_GRID_OUTSIDE_BOUNDS = 3

_TBounds = tp.Tuple[float, float]

logger = logging.getLogger(__name__)


def create_grid_for_optimizable_parameter(
    repairs: tp.List[Repair],
    optimizable_parameter: str,
    bounds: _TBounds,
    initial_position: float,
    optimized_position: float,
    n_points_in_grid: tp.Optional[int] = None,
) -> tp.List[float]:
    """
    Create grid for ``optimizable_parameter`` to be used for visualization.
    The grid is built in a such way that we have a linear space within the bounds plus
    some margin from additional points out of bounds.

    If optimized position is out of bounds, the grid will be expanded
    up to the optimized point.

    If the initial position is out of bounds, the grid won't be expanded to include it.

    If ``problem`` contains ``SetRepairs`` constraining ``optimizable_parameter``,
    then grid is extracted using the allowed values from ``SetRepairs``.

    If ``problem`` does not contain ``SetRepairs``
    constraining ``optimizable_parameter``,
    then grid is populated uniformly between ``lhs_bound`` and ``rhs_bound``.

    Args:
        repairs: used in solution's problem
        optimizable_parameter: optimization parameter name
        bounds: lower and upper optimization bound
        initial_position: initial position of the ``optimizable_parameter``
            to be added into grid
        optimized_position: optimized position of the ``optimizable_parameter``
            to be added into grid
        n_points_in_grid: number of points to use
            in grid if repairs can't be produced by ``SetRepairs``

    Warnings:
        UserWarning: Warns user that parameter `n_points_in_grid` is ignored
            when relevant set repairs that constrain ``optimizable_parameter`` are found

    Returns:
        Grid of values to be
    """
    _validate_bounds(bounds)

    # todo: what if not SetRepair but other constraining repair?
    relevant_set_repairs = [
        repair
        for repair in repairs
        if _is_parameter_constrained_by_set_repair(repair, optimizable_parameter)
    ]
    if any(relevant_set_repairs):
        if n_points_in_grid is not None:
            warnings.warn(
                "Parameter `n_points_in_grid` is ignored. Problem"
                f" contains `SetRepairs` that constrain {optimizable_parameter}"
                " and define the grid.",
            )
        logger.debug(
            "Using allowed values by SetRepairs"
            f" for building a grid for {optimizable_parameter}.",
        )
        grid = _create_grid_from_set_repairs(relevant_set_repairs, bounds)
    else:
        n_points_in_grid = (
            n_points_in_grid
            if n_points_in_grid is not None
            else N_POINTS_IN_GRID_BETWEEN_BOUNDS
        )
        logger.debug(
            f"Populating the grid for {optimizable_parameter}"
            " uniformly between optimization bounds",
        )
        grid = _create_uniform_grid(
            bounds, initial_position, optimized_position, n_points_in_grid,
        )
    return sorted(grid)


def _validate_bounds(bounds: _TBounds) -> None:
    lower_bound, upper_bound = bounds
    if lower_bound > upper_bound:
        raise ValueError(
            f"Please provide correct bounds: lower bound is less or equal than upper. "
            f"Got {lower_bound = } is greater than {upper_bound = }",
        )


def _is_parameter_constrained_by_set_repair(
    repair: Repair, optimizable_parameter: str,
) -> tp.TypeGuard[SetRepair]:
    if isinstance(repair, SetRepair):
        constrained_parameter = repair.constraint.constraint_func.keywords["col"]
        if constrained_parameter == optimizable_parameter:
            return True
    return False


def _create_uniform_grid(
    bounds: _TBounds,
    initial_position: float,
    optimized_position: float,
    n_points_in_grid: int,
) -> tp.List[float]:
    lower_bound, upper_bound = bounds
    grid_between_bounds, grid_step = np.linspace(
        lower_bound,
        upper_bound,
        num=n_points_in_grid,
        endpoint=True,
        retstep=True,
    )
    additional_points_to_add_in_grid = [optimized_position]
    if lower_bound <= initial_position <= upper_bound:
        additional_points_to_add_in_grid.append(initial_position)
    if np.isnan(grid_step) or np.equal(grid_step, 0):
        return list(grid_between_bounds) + additional_points_to_add_in_grid
    for _ in range(MAX_POINT_IN_GRID_OUTSIDE_BOUNDS):
        lower_bound -= float(grid_step)
        additional_points_to_add_in_grid.append(lower_bound)
        upper_bound += float(grid_step)
        additional_points_to_add_in_grid.append(upper_bound)
    return list(grid_between_bounds) + additional_points_to_add_in_grid


def _create_grid_from_set_repairs(
    set_repairs: tp.List[SetRepair],
    bounds: _TBounds,
) -> tp.List[float]:
    lower_bound, upper_bound = bounds
    candidate_points = [
        set(repair.constraint.constraint_set.collection)  # fixme
        for repair in set_repairs
    ]
    allowed_points = sorted(set.intersection(*candidate_points))
    points_between_bounds = []
    points_less_than_lhs_bound = []
    points_greater_than_rhs_bound = []
    for point in allowed_points:
        if point < lower_bound:
            points_less_than_lhs_bound.append(point)
        if lower_bound <= point <= upper_bound:
            points_between_bounds.append(point)
        if upper_bound < point:
            points_greater_than_rhs_bound.append(point)
    if not points_between_bounds:
        raise ValueError(
            "No points to plot in grid: no points allowed"
            " by SetRepairs, check provided repairs.",
        )
    return (
        points_less_than_lhs_bound[-MAX_POINT_IN_GRID_OUTSIDE_BOUNDS:]
        + points_between_bounds
        + points_greater_than_rhs_bound[:MAX_POINT_IN_GRID_OUTSIDE_BOUNDS]
    )
