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

import abc
import typing as tp

from optimizer import Penalty, Repair

from .controlled_parameters import (
    ControlledParametersConfig,
    get_notnull_controls,
)
from .types import ObjectiveFunction, Predictor, TProblem, TRowToOptimize
from .utils import pformat, validate_is_a_single_row_dataframe, validate_kwargs

_FORBIDDEN_KEYS = frozenset(
    ("objective", "state", "optimizable_columns", "penalties", "repairs"),
)
TModelRegistry = tp.Dict[str, Predictor]


class ProblemFactoryBase(abc.ABC, tp.Generic[TProblem]):
    def __init__(
        self,
        controlled_parameters_config: ControlledParametersConfig,
        problem_class: tp.Type[TProblem],
        problem_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        model_registry: tp.Optional[TModelRegistry] = None,
        **kwargs: tp.Any,
    ) -> None:
        """
        This class introduces the abstraction which is responsible for problem creating.
        Its goal is to create a problem based on input row to optimize. Therefore,
        it contains all the details about objective, repairs, penalties and other things
        one might need when creating a problem.

        Args:
            controlled_parameters_config: list of columns' configs;
                each config must either be an instance if ``ControlledParameterConfig``
                or a dict that can be parsed into ``ControlledParameterConfig``
            problem_class: problem class used to create problem in ``create_problem``
            problem_kwargs: additional problem kwargs passed ``problem_class``
                when creating problem in ``create_problem``
            model_registry: dictionary that maps model's name to its fitted instance
            kwargs: additional keyword args for extending a base class
        """
        self._controlled_parameters_config = controlled_parameters_config
        self._model_registry = _validate_model_registry(model_registry or {})
        self._problem_class = problem_class
        self._problem_kwargs = validate_kwargs(
            kwargs=problem_kwargs or {},
            forbidden_keys=_FORBIDDEN_KEYS,
            source="problem_kwargs",
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        problem_class_name = self._problem_class.__name__
        optimized_columns = pformat(self._controlled_parameters_config, indent=4)
        model_registry = pformat(set(self._model_registry.keys()), indent=8)
        return (
            f"{class_name}(\n"
            f"    optimized_columns={optimized_columns},\n"
            f"    problem_class={problem_class_name},\n"
            f"    problem_kwargs={self._problem_kwargs},\n"
            f"    model_registry={model_registry},\n"
            f")"
        )

    def create(self, row_to_optimize: TRowToOptimize) -> TProblem:
        """
        Returns:
             ``self.problem_class`` instance with following arguments:
            * ``self.problem_kwargs``
            * objective created by ``self._create_objective(row_to_optimize)``
            * penalties created by ``self._create_penalties(row_to_optimize)``
            * repairs created by ``self._create_repairs(row_to_optimize)``
            * state defined as ``row_to_optimize``

        Raises:
            ValueError: if ``row_to_optimize`` is empty or has more than one row
        """
        validate_is_a_single_row_dataframe(row_to_optimize)
        problem: TProblem = self._problem_class(
            objective=self._create_objective(row_to_optimize),
            state=row_to_optimize,
            optimizable_columns=self._get_columns_to_optimize(row_to_optimize),
            penalties=self._create_penalties(row_to_optimize),
            repairs=self._create_repairs(row_to_optimize),
            **self._problem_kwargs,
        )
        return problem  # noqa: WPS331

    @abc.abstractmethod
    def _create_objective(self, row_to_optimize: TRowToOptimize) -> ObjectiveFunction:
        """
        Abstract method that creates an objective.

        Returns:
            an objective function
            that complies with ``recommend.types.ObjectiveFunction`` protocol
            (i.e. a function that takes in ``parameters`` dataframe
            and returns a series of objectives for each row).

        Note:
            The objective must not include any penalties.
            Use ``_create_penalties`` method to create penalties
            included tp optimization.
        """

    @abc.abstractmethod
    def _create_penalties(self, row_to_optimize: TRowToOptimize) -> tp.List[Penalty]:
        """
        Abstract method that creates list of penalties.

        Returns:
            penalties (functions including penalty to the objective)
            that will be plugged-in to the problem's ``penalties``

        Note:
            Q: Should I specify the sign in penalty function or multiplier
            depending on the min/max optimization sense?

            A: No, refer to the penalty as an absolute number.
            Problem class will take care of signs depending on its sense kwarg.
        """

    @abc.abstractmethod
    def _create_repairs(self, row_to_optimize: TRowToOptimize) -> tp.List[Repair]:
        """
        Abstract method that creates list of repairs.

        Returns:
            repairs (functions changing optimization parameters)
            that will be plugged-in to the problem's ``repairs``

        Note:
            Think twice before crating repairs as they slow down the optimization
            process. Specify controlled parameters domain for solver if possible.
        """

    def _get_columns_to_optimize(self, row_to_optimize: TRowToOptimize) -> tp.List[str]:
        """
        Returns columns that can be optimized.

        The criteria for optimization is:
        * controlled parameter's value is not null
        """
        notnull_controls = get_notnull_controls(
            row_to_optimize, list(self._controlled_parameters_config),
        )
        # todo: add dependency graph for controlling on/off logic
        #  based on a dependency graph
        return notnull_controls  # noqa: WPS331


def _validate_model_registry(model_registry: TModelRegistry) -> TModelRegistry:
    non_compliant_models = [
        model_name
        for model_name, model in model_registry.items()
        if not isinstance(model, Predictor)
    ]
    if non_compliant_models:
        raise ValueError(
            f"Models from registry must comply with Predictor protocol. "
            f"Found following non complying models: {non_compliant_models}",
        )
    return model_registry
