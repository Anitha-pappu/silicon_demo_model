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
from .bounded_linear_space_domain import BoundedLinearSpaceDomain
from .discrete_domain_generator import DiscreteGridDomainGenerator


class MixedSpaceDomain(BaseDomainGenerator):

    @staticmethod
    def generate_for_single_parameter(
        row_to_optimize: TRowToOptimize,
        control_config: ControlledParameter,
    ) -> tp.Union[tp.Tuple[float, float], tp.Set[float]]:
        """
        Redirects call to either ``DiscreteGridDomainGenerator``
        if ``control_config.step_size`` is not None,
        or to ``BoundedLinearSpaceDomain`` otherwise.

        Args:
            row_to_optimize: currently optimized row for controlled columns of which
                domain is generated
            control_config: control's config
        """
        if control_config.step_size is None:
            return BoundedLinearSpaceDomain.generate_for_single_parameter(
                row_to_optimize,
                control_config,
            )
        return DiscreteGridDomainGenerator.generate_for_single_parameter(
            row_to_optimize,
            control_config,
        )
