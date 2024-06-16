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
from functools import partial

from recommend.controlled_parameters import ControlledParameter
from recommend.types import TRowToOptimize

logger = logging.getLogger(__name__)

_TRange = tp.Tuple[float, float]
_PRECISION = 2


def get_single_variable_optimization_range(
    row: TRowToOptimize,
    control: ControlledParameter,
    warn: bool = True,
) -> _TRange:
    """
    Determines the optimization range of the control based on its current value,
    max step and operations' min/max limits.

    Determines the optimization range for a single variable (intended for domains where
    the domains of the variables are defined independent of one another, can be applied
    in some cases where the domain for some of the variables can be defined in this
    way).

    Solves edge cases in the following way (``curr_value = row[control.name]``):

    - if ``curr_value`` is between ``op_min`` and ``op_max``,
      returns the intersection of
      ``[op_min, op_max]`` and ``[curr_value - max_delta, curr_value + max_delta]``
    - if ``curr_value`` is below ``op_min``,
      returns ``[op_min, op_min + max_delta]``
    - if ``curr_value`` is above ``op_max``,
      returns ``[op_min - max_delta, op_max]``

    Args:
        row: contains current value of the control (column named ``control.name``)
        control: control definition
        warn: use ``warnings.warn`` if True, ``logging.warning`` otherwise

    Warnings:
        OutOfDomainWarning: thrown if warn is True and ``curr_value`` is outside
          operating range: ``[op_min, op_max]``
    """
    curr_value = _get_current_value(row, control.name)

    throw_warning = tp.cast(
        tp.Callable[[str], None],
        partial(warnings.warn, category=OutOfDomainWarning) if warn else logger.warning,
    )
    rounded_curr_value = round(curr_value, _PRECISION)
    if curr_value > control.op_max:
        throw_warning(
            f"Current value = {rounded_curr_value} for {control.name}"
            f"is above `op_max` = {control.op_max}. "
            f"Proceeding as if the current value was equal to `op_max`",
        )
        curr_value = control.op_max
    if curr_value < control.op_min:
        throw_warning(
            f"Current value = {rounded_curr_value} for {control.name} "
            f"is below `op_min` = {control.op_min}. "
            f"Proceeding as if the current value was equal to `op_min`",
        )
        curr_value = control.op_min
    if control.max_delta is None:
        return control.op_min, control.op_max
    lower_bound = max(control.op_min, curr_value - control.max_delta)
    upper_bound = min(control.op_max, curr_value + control.max_delta)
    return lower_bound, upper_bound


def _get_current_value(row: TRowToOptimize, control_name: str) -> float:
    """Get the current value of the optimizable variable from the data."""
    return float(row[control_name])


class OutOfDomainWarning(UserWarning):
    """Thrown when control's current value is outside the operating range"""


def constraint_optimization_range(
    row: TRowToOptimize,
    control: ControlledParameter,
    optimization_range: _TRange,
) -> _TRange:
    """
    Constraints the optimization range of the control based on its current value
    and ``control.direction_bound``.

    Match ``control.direction_bound`` values:

    * "increase" updates the start point of the range
        is updated as ``max(current_value, lower)`` and the end point is left intact
    * "decrease" updates the end point of the range
        is updated as ``min(current_value, upper)`` and the start point is left intact
    * None leaves ``optimization_range`` intact

    Args:
        row: contains current value of the control (column named ``control.name``)
        control: control definition
        optimization_range: the range that needs constraining

    Returns:
        Possibly updated optimization range
    """
    current_value = _get_current_value(row, control.name)
    lower, upper = optimization_range
    if control.direction_bound == "increase":
        return max(current_value, lower), upper
    elif control.direction_bound == "decrease":
        return lower, min(current_value, upper)
    return optimization_range
