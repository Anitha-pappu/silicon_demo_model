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

from __future__ import annotations

import typing as tp
from copy import deepcopy

import pandas as pd

from optimizer.solvers import Solver
from optimizer.stoppers.base import BaseStopper
from recommend.controlled_parameters import ControlledParametersConfig
from recommend.domain_generator import (
    BaseDomainGenerator,
    get_compatible_domain_generator,
)
from recommend.types import TRowToOptimize
from recommend.utils import (
    pformat,
    validate_is_a_single_row_dataframe,
    validate_kwargs,
)
from recommend.warnings_catcher import TWarningsLevelBasic

from ._warnings import OutOfDomainCatcher  # noqa: WPS436

_FORBIDDEN_KEYS = frozenset(("domain", ))

_TKwargs = tp.Optional[tp.Dict[str, tp.Any]]
_TStopperClass = tp.Optional[tp.Type[BaseStopper]]
_TDomainGeneratorBaseClass = tp.Optional[tp.Type[BaseDomainGenerator]]


class SolverFactory(object):
    def __init__(
        self,
        controlled_parameters_config: ControlledParametersConfig,
        solver_class: tp.Type[Solver],
        solver_kwargs: _TKwargs = None,
        stopper_class: _TStopperClass = None,
        stopper_kwargs: _TKwargs = None,
        domain_generator_class: _TDomainGeneratorBaseClass = None,
        domain_generator_kwargs: _TKwargs = None,
    ) -> None:
        self._solver_class = solver_class
        self._solver_kwargs = validate_kwargs(
            solver_kwargs or {},
            # Since we're creating domain via domain generator in
            # ``self.create``, we don't want users to provide domain argument
            forbidden_keys=_FORBIDDEN_KEYS,
            source="solver_kwargs",
        )
        self._stopper_class = stopper_class
        self._stopper_kwargs = stopper_kwargs or {}
        self._domain_generator = self._create_domain_generator(
            controlled_parameters_config,
            domain_generator_class,
            domain_generator_kwargs,
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        solver_class = self._solver_class.__name__
        solver_kwargs = pformat(self._solver_kwargs, indent=8)
        stopper_class = self._stopper_class.__name__ if self._stopper_class else None
        stopper_kwargs = pformat(self._stopper_kwargs, indent=8)
        domain_generator = pformat(self._domain_generator, indent=4)
        return (
            f"{class_name}(\n"
            f"    solver_class={solver_class},\n"
            f"    solver_kwargs={solver_kwargs},\n"
            f"    stopper_class={stopper_class},\n"
            f"    stopper_kwargs={stopper_kwargs},\n"
            f"    domain_generator={domain_generator},\n"
            f")"
        )

    def create(
        self,
        row_to_optimize: TRowToOptimize,
        active_controlled_parameters: tp.List[str],
        warnings_details_level: TWarningsLevelBasic = "aggregated",
    ) -> tp.Tuple[Solver, tp.Optional[BaseStopper]]:
        validate_is_a_single_row_dataframe(row_to_optimize)

        n_active_controls = len(active_controlled_parameters)
        row_index = row_to_optimize.index[0]
        with OutOfDomainCatcher(warnings_details_level, row_index, n_active_controls):
            solver = self._create_solver(row_to_optimize, active_controlled_parameters)

        stopper = (
            self._stopper_class(**deepcopy(self._stopper_kwargs))
            if self._stopper_class
            else None
        )

        return solver, stopper

    def _create_solver(
        self,
        row_to_optimize: pd.DataFrame,
        active_controlled_parameters: tp.List[str],
    ) -> Solver:
        domain = self._domain_generator.generate(
            row_to_optimize, active_controlled_parameters,
        )
        return self._solver_class(domain=domain, **deepcopy(self._solver_kwargs))

    def _create_domain_generator(
        self,
        controlled_parameters_config: ControlledParametersConfig,
        domain_generator_class: tp.Optional[tp.Type[BaseDomainGenerator]],
        domain_generator_kwargs: _TKwargs,
    ) -> BaseDomainGenerator:
        if domain_generator_class is None:
            domain_generator_class = get_compatible_domain_generator(self._solver_class)
        return domain_generator_class(
            controlled_parameters_config, **(domain_generator_kwargs or {}),
        )
