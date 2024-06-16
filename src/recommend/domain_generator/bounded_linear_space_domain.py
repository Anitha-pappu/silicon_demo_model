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

from recommend.types import TRowToOptimize

from ..controlled_parameters import ControlledParameter
from .base import BaseDomainGenerator
from .utils import (
    constraint_optimization_range,
    get_single_variable_optimization_range,
)


class BoundedLinearSpaceDomain(BaseDomainGenerator):
    """
    A real-dimension domain generator.
    Generates min/max-bounds domain space utilized by continuous solvers.
    """

    @staticmethod
    def generate_for_single_parameter(
        row_to_optimize: TRowToOptimize,
        control_config: ControlledParameter,
    ) -> tp.Tuple[float, float]:
        """
        Returns min/max limits for domain space for the given parameter.

        Args:
            row_to_optimize: currently optimized row for controlled columns of which
                domain is generated
            control_config: control's config
        """
        opt_range = get_single_variable_optimization_range(
            row=row_to_optimize, control=control_config,
        )
        return constraint_optimization_range(
            row=row_to_optimize, control=control_config, optimization_range=opt_range,
        )
