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

import numpy as np
from numpy import typing as npt

from recommend.types import TRowToOptimize

from ..controlled_parameters import ControlledParameter
from .base import BaseDomainGenerator
from .utils import (
    constraint_optimization_range,
    get_single_variable_optimization_range,
)

_TFloatArray = npt.NDArray[np.float_]

logger = logging.getLogger(__name__)


class DiscreteGridDomainGenerator(BaseDomainGenerator):
    """
    A discrete grid domain generator.
    Generates evenly spaced real dimension grid utilised by discrete solvers.
    """

    def __init__(
        self,
        controlled_parameters: tp.Mapping[str, ControlledParameter],
        **kwargs: tp.Any,
    ) -> None:
        """
        Inits discrete domain space utilised by discrete solvers.

        Raises:
            ValueError: raised if ``step_size`` for any parameter is missing.

        Args:
            controlled_parameters:
        """
        super().__init__(controlled_parameters, **kwargs)
        self._validate_step_size_is_not_missing()

    @staticmethod
    def generate_for_single_parameter(
        row_to_optimize: TRowToOptimize,
        control_config: ControlledParameter,
    ) -> tp.Set[float]:
        """
        Returns a discrete grid of values as a domain for the optimizable variable.

        Args:
            row_to_optimize: currently optimized row for controlled columns of which
                domain is generated
            control_config: control's config
        """

        opt_range = get_single_variable_optimization_range(
            row=row_to_optimize, control=control_config,
        )
        lower_bound, upper_bound = constraint_optimization_range(
            row=row_to_optimize, control=control_config, optimization_range=opt_range,
        )
        discrete_domain = _generate_discrete_domain_for_single_variable(
            control=control_config,
        )
        discrete_domain = _mask_out_of_bounds_domain(
            discrete_domain=discrete_domain,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        return set(discrete_domain)

    def _validate_step_size_is_not_missing(self) -> None:
        missing_step_sizes = [
            parameter.name
            for parameter in self._controlled_parameters.values()
            if parameter.step_size is None
        ]
        if missing_step_sizes:
            raise ValueError(
                f"Missing `step_size` attribute"
                f" for following parameters: {missing_step_sizes}",
            )


def _mask_out_of_bounds_domain(
    discrete_domain: _TFloatArray,
    lower_bound: float,
    upper_bound: float,
) -> _TFloatArray:
    """
    Keep only domain values between ``lower_bound`` and ``upper_bound``.

    Args:
        discrete_domain: linear space
        lower_bound: lower bound
        upper_bound: upper bound

    Returns:
        Updated linear space.
    """
    is_within_bounds = (
        (lower_bound <= discrete_domain) & (discrete_domain <= upper_bound)
    )
    return tp.cast(_TFloatArray, discrete_domain[is_within_bounds])


def _generate_discrete_domain_for_single_variable(
    control: ControlledParameter, default_step_size: tp.Optional[int] = None,
) -> _TFloatArray:
    """
    Generate a discrete grid domain.

    Notes:
        - If ``step_size`` is not provided, creates an evenly spaced domain of
          ``default_step_size`` points between ``op_min`` and ``op_max``.
        - If ``step_size`` is provided, creates an evenly spaced domain starting from
          ``op_min`` and adding points with step equal to ``step_size``.
          Ensures that ``op_max`` is present in the domain by adding it to the array
          if it is not present (this happens if ``op_max`` - ``op_min`` is not a
          multiple of ``step_size``).
    """
    if control.step_size is None:
        if default_step_size is None:
            raise ValueError(
                f"`step_size` for {control.name} is not provided."
                f" Either provide control-specific step size or `default_step_size`"
                f" argument to the domain generator.",
            )
        lin_space, step_size = np.linspace(
            start=control.op_min,
            stop=control.op_max,
            num=default_step_size,
            retstep=True,
        )
        logger.info(
            f"`step_size` for {control.name} is not provided."
            f" Divided the operational range into steps of size {step_size}."
            f" This domain consists of {default_step_size} points.",
        )
        return tp.cast(_TFloatArray, lin_space)
    lin_space = np.arange(
        start=control.op_min, stop=control.op_max, step=control.step_size,
    )
    if control.op_max not in lin_space:
        lin_space = np.append(lin_space, control.op_max)
    num_points = len(lin_space)
    logger.info(
        f"Created an evenly spaced domain for {control.name}. "
        f"Starting from {control.op_min} with step {control.step_size}. "
        f"This domain consists of {num_points} points.",
    )
    return tp.cast(_TFloatArray, lin_space)
