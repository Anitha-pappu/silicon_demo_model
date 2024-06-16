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
Contains domain generators definitions.
Domain generators produce domains essential for ``Solver``.
This module depends on ``controlled_parameters``.
"""

from .base import BaseDomainGenerator, TDomain
from .bounded_linear_space_domain import BoundedLinearSpaceDomain
from .discrete_domain_generator import DiscreteGridDomainGenerator
from .domain_compatability import get_compatible_domain_generator
from .mixed_domain_generator import MixedSpaceDomain
from .utils import OutOfDomainWarning
