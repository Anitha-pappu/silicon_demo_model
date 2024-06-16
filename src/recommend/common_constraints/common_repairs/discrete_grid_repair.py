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

from optimizer import SetRepair
from optimizer import repair as create_repair
from recommend.controlled_parameters import ControlledParametersConfig


def create_repairs_for_discrete_grid_controls(
    controls: tp.List[str],
    controlled_parameters_config: ControlledParametersConfig,
) -> tp.List[SetRepair]:
    """
    Generate repair for each control in ``controls`` that will map ``control``
    to a discrete grid using ``controlled_parameters_config``.

    Note: the repair will be returned only for those controlled parameters
    that have notnull step size in ``controlled_parameters_config``.

    Returns:
        list of repairs, one for reach control
        that has a specified step size in its config
    """
    missing_step_sizes = [
        control
        for control in controls
        if controlled_parameters_config[control].step_size is None
    ]
    if missing_step_sizes:
        raise ValueError(
            f"Following requested controls are missing "
            f"step sizes: {missing_step_sizes}",
        )

    repairs = []
    for control in controls:
        control_config = controlled_parameters_config[control]
        repairs.append(
            create_discrete_grid_repair(
                column=control_config.name,
                op_min=control_config.op_min,
                op_max=control_config.op_max,
                # current version of python doesn't understand type guards
                # we've checked that step size is present earlier
                step_size=control_config.step_size,  # type: ignore
            ),
        )
    return repairs


def create_discrete_grid_repair(
    column: str,
    op_min: float,
    op_max: float,
    step_size: float,
) -> SetRepair:
    """
    Creates a new set-based constraint set repair for a given ``column``;
    this repair will map column to a discrete grid [op_min, op_max]
    with a fixed ``step_size``.

    Notes:
        Grid is created via ``_get_linear_search_space`` with following rules:
        * ``op_min`` and ``op_max`` is always included.
        * the step size from ``op_max`` to previous value might differ from
            ``step_size``

        Grid examples::
            >>> _get_linear_search_space(2.0, 4.0, 0.5)
            [2.0, 2.5, 3.0, 3.5, 4]
            >>> _get_linear_search_space(2.0, 4.0, 0.7)
            [2.0, 2.7, 3.4, 4.0]
            >>> _get_linear_search_space(2.0, 4.0, 5.0)
            [2.0, 4.0]
    """

    discrete_grid = _get_linear_search_space(op_min, op_max, step_size)
    set_repair = create_repair(column, "in", discrete_grid)
    return tp.cast(SetRepair, set_repair)


def _get_linear_search_space(
    op_min: float, op_max: float, step_size: float,
) -> tp.List[float]:
    """
    This function creates linear search space for the control.

    Notes:
        * ``op_min`` and ``op_max`` is always included.
        * the step size from ``op_max`` to previous value might differ from
            ``step_size``
        * resulting grid is cast to most precise value of args;
            e.g. if bounds are int and step size = 0.01,
            then resulting grid is calculated with ``value:.2f`` precision

    Examples::
        >>> _get_linear_search_space(2.0, 4.0, 0.5)
        [2.0, 2.5, 3.0, 3.5, 4]
        >>> _get_linear_search_space(2.0, 4.0, 0.7)
        [2.0, 2.7, 3.4, 4.0]
        >>> _get_linear_search_space(2.0, 4.0, 5.0)
        [2.0, 4.0]

    Args:
        op_min: minimum value of linear search space
        op_max: maximum value of linear search space
        step_size: distance between each value in linear search space

    Returns:
        linear search space
    """
    precision = _get_mutual_precision(op_min, op_max, step_size)
    search_space = list(np.arange(op_min, op_max, step_size, dtype=float))
    if search_space[-1] != op_max:
        search_space.append(float(op_max))
    return [round(step, precision) for step in search_space]


def _get_mutual_precision(*numbers: float) -> int:
    return max(_get_precision(number) for number in numbers)


def _get_precision(numeric_value: float) -> int:
    return max(str(numeric_value)[::-1].find('.'), 0)
