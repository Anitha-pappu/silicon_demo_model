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

from optimizer import solvers

from .base import BaseDomainGenerator
from .bounded_linear_space_domain import BoundedLinearSpaceDomain
from .discrete_domain_generator import DiscreteGridDomainGenerator
from .mixed_domain_generator import MixedSpaceDomain

KNOWN_SOLVERS = frozenset((
    solvers.ContinuousSolver,
    solvers.DiscreteSolver,
    solvers.GeneticAlgorithmSolver,
))


def get_compatible_domain_generator(
    solver_class: tp.Type[solvers.Solver],
) -> tp.Type[BaseDomainGenerator]:
    """Returns solver-compatible domain generator class"""
    if issubclass(solver_class, solvers.DiscreteSolver):
        return DiscreteGridDomainGenerator
    elif issubclass(solver_class, solvers.ContinuousSolver):
        return BoundedLinearSpaceDomain
    elif issubclass(solver_class, solvers.GeneticAlgorithmSolver):
        return MixedSpaceDomain
    solver_class_name = solver_class.__name__
    raise NotImplementedError(
        f"Solver class {solver_class_name = } doesn't have "
        f"any compatible domain generators. "
        f"Please use one of the following base solver classes: {KNOWN_SOLVERS}.",
    )
