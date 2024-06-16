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

from pydantic.v1 import Field, parse_obj_as, validator
from pydantic.v1.generics import GenericModel

from optimizer.solvers import Solver
from optimizer.stoppers import BaseStopper
from recommend.controlled_parameters import ControlledParametersConfig
from recommend.domain_generator import BaseDomainGenerator
from recommend.problem_factory import ProblemFactoryBase
from recommend.solver_factory import SolverFactory
from recommend.types import Predictor, TProblem
from recommend.utils import load_obj

_Type = tp.TypeVar("_Type", bound=tp.Type[tp.Any])
_TKwargs = tp.Dict[str, tp.Any]


class ObjectSpec(GenericModel, tp.Generic[_Type]):
    """
    This pydantic generic model stores ``type`` and ``kwargs`` attributes.
    Typically one can use those to create an instance of ``type``::
        >>> spec: ObjectSpec[typing.Any]
        >>> object_from_spec = spec.type(**spec.kwargs)

    That is useful when working
    with workflow-control systems like kedro/airflow/etc.
    that typically don't validate args loaded from config files.

    This generic class is typed by the class `_Type` it's expected to describe i.e.
    ``type`` attribute is validated to be a subclass of `_Type`.
    Doing such helps to validate the user has provided a correct ``type``
    (since it can be provided as a string)::
        >>> class A:
        ...     pass
        >>> class B:
        ...     pass
        >>> ObjectSpec[typing.Type[A]](type=B)
        ValidationError: 1 validation error for ObjectSpec[Type[__main__.A]]
        type
          subclass of A expected (type=type_error.subclass; expected_class=A)
    """
    type: tp.Optional[_Type] = None
    kwargs: _TKwargs = Field(default_factory=dict)

    @validator("type", pre=True)
    def _parse_type(
        cls, type_value: tp.Union[_Type, str, None],  # noqa: N805
    ) -> tp.Optional[_Type]:
        """
        Returns a loaded from the path ``type_value`` type if it's a string.
        Otherwise, assumes it's a loaded type already and returns ``type_value``.
        """
        if isinstance(type_value, str):
            return load_obj(type_value)  # type: ignore
        return type_value


_TRawSpec = tp.Union[tp.Dict[str, tp.Any], ObjectSpec[tp.Any]]


def create_problem_factory(
    *,
    controlled_parameters_config: ControlledParametersConfig,
    factory_spec: _TRawSpec,
    problem_spec: _TRawSpec,
    **models: Predictor,
) -> ProblemFactoryBase[TProblem]:
    """
    This function implements a functional interface for creating the problem factory.
    Which is useful in different workflow-control systems like kedro/airflow/etc.

    Parses spec args using ``recommend.functional.ObjectSpec``
    and creates problem factory instance (``factory_spec.type`` instance)
    with the rest of the arguments used in factory's init::
        >>> return factory_spec_.type(
        ...     controlled_parameters_config,
        ...     problem_spec_.type,
        ...     problem_spec_.kwargs,
        ...     models,
        ...     **factory_spec_.kwargs,
        ... )
    """
    factory_spec_ = parse_obj_as(
        ObjectSpec[tp.Type[ProblemFactoryBase[TProblem]]], factory_spec,
    )
    problem_spec_ = parse_obj_as(ObjectSpec[tp.Type[TProblem]], problem_spec)
    if factory_spec_.type is None:
        raise ValueError("Factory type can't be None")
    if problem_spec_.type is None:
        raise ValueError("Problem type can't be None")

    return factory_spec_.type(
        controlled_parameters_config,
        problem_spec_.type,
        problem_spec_.kwargs,
        models,
        **factory_spec_.kwargs,
    )


_DEFAULT_SOLVER_FACTORY_SPEC = ObjectSpec(type=SolverFactory)
_EMPTY_SPEC = ObjectSpec[tp.Any]()


def create_solver_factory(
    *,
    controlled_parameters_config: ControlledParametersConfig,
    factory_spec: _TRawSpec = _DEFAULT_SOLVER_FACTORY_SPEC,
    solver_spec: _TRawSpec,
    stopper_spec: _TRawSpec,
    domain_generator_spec: _TRawSpec = _EMPTY_SPEC,
) -> SolverFactory:
    """
    This function implements a functional interface for creating the solver factory.
    Which is useful in different workflow-control systems like kedro/airflow/etc.

    Parses spec args using ``recommend.functional.ObjectSpec``
    and creates solver factory instance (``factory_spec.type`` instance)
    with the rest of the arguments used in factory's init::
        >>> return factory_spec_.type(
        ...     controlled_parameters_config,
        ...     solver_spec_.type,
        ...     solver_spec_.kwargs,
        ...     stopper_spec_.type,
        ...     stopper_spec_.kwargs,
        ...     domain_generator_spec_.type,
        ...     domain_generator_spec_.kwargs,
        ... )
    """
    factory_spec_ = parse_obj_as(ObjectSpec[tp.Type[SolverFactory]], factory_spec)
    solver_spec_ = parse_obj_as(ObjectSpec[tp.Type[Solver]], solver_spec)
    stopper_spec_ = parse_obj_as(ObjectSpec[tp.Type[BaseStopper]], stopper_spec)
    domain_generator_spec_ = parse_obj_as(
        ObjectSpec[tp.Type[BaseDomainGenerator]], domain_generator_spec,
    )
    if factory_spec_.type is None:
        raise ValueError("Factory type can't be None")
    if solver_spec_.type is None:
        raise ValueError("Solver type can't be None")
    return factory_spec_.type(
        controlled_parameters_config,
        solver_spec_.type,
        solver_spec_.kwargs,
        stopper_spec_.type,
        stopper_spec_.kwargs,
        domain_generator_spec_.type,
        domain_generator_spec_.kwargs,
    )
