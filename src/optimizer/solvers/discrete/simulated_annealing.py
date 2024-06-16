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
Discrete simulated annealing code.
"""

import typing as tp
from copy import deepcopy

import numpy as np

from optimizer.domain import CategoricalDimension, IntegerDimension
from optimizer.exceptions import InitializationError, MaxIterationError
from optimizer.solvers.discrete.base import DiscreteSolver
from optimizer.types import MAXIMIZE, MINIMIZE, Matrix, TSense, Vector


class ExpDecay:
    """Exponential decay iterator class."""

    def __init__(
        self,
        initial_temp: float = 1e8,
        final_temp: float = 0.1,
        exp_const: float = 1e-5,
    ):
        """Init class."""
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.exp_const = exp_const
        self.current_t = 0

    def __iter__(self) -> tp.Iterator[float]:
        """Iter function."""
        return self

    def __next__(self) -> float:
        """Next function."""
        temp: float = self.initial_temp * np.exp(-1.0 * self.exp_const * self.current_t)
        temp = max(self.final_temp, temp)

        self.current_t += 1
        return temp


class DiscreteSimulatedAnnealingSolver(DiscreteSolver):
    """Simulated annealing with discrete options for each row.
    Provided an option set, which for each element of the vector provides the discrete
    values the element may take, this implementation searches for the best combination
    of options.
    """

    def __init__(
        self,
        domain: tp.List[tp.Union[
            tp.List[tp.Any],
            IntegerDimension,
            CategoricalDimension,
        ]],
        sense: TSense = MINIMIZE,
        seed: tp.Optional[int] = None,
        initial_x: tp.Optional[Vector] = None,
        maxiter: int = 100000,
        schedule: tp.Optional[tp.Any] = None,
        initial_temp: float = 1e8,
        final_temp: float = 0.1,
        exp_const: float = 1e-5,
    ):
        """Constructs the solver.

        Args:
            domain: list describing the available choices for each dimension.
            sense: string, optional sense string for "minimize" or "maximize".
            seed: int, optional random seed.
            initial_x: Starting point for the search, if none is given, a random point
                is used.
            maxiter: Maximum number of search iteration
            schedule: Schedule object to decrease temperature, e.g. exponential decay
            initial_temp: float, given to ExpDecay if schedule not provided.
            final_temp: float, given to ExpDecay if schedule not provided.
            exp_const: float, given to ExpDecay if schedule not provided.
        """
        super().__init__(domain=domain, sense=sense, seed=seed)

        schedule = (
            ExpDecay(
                initial_temp=initial_temp, final_temp=final_temp, exp_const=exp_const
            )
            if schedule is None
            else schedule
        )

        for i, dimension in enumerate(self.domain):
            self.domain[i] = set(dimension)

        self.initial_point = (
            initial_x if initial_x is not None else self._initialize_point()
        )
        self.current_point: tp.Optional[Vector] = None
        self.current_objective = np.inf

        self.best_objective = np.inf
        self.best_point: Vector

        self.schedule = schedule
        self.maxiter = maxiter
        self.niters = 0

    def _initialize_point(self) -> Vector:
        """Initializes the current point to some random starting point."""
        self.current_point = np.zeros((len(self.domain)))

        for index, _ in enumerate(self.current_point):
            options = self.domain[index]
            self.current_point[index] = self.rng.choice(list(options))

        return self.current_point

    def _generate_point(self) -> Vector:
        """Generates a new point.
        Does this by choosing a random index and mutating the current
        point there.
        """
        index_to_mutate = self.rng.randint(
            low=0,
            high=len(self.current_point),  # type: ignore
        )
        return self._mutate_single_entry(index_to_mutate)

    def _mutate_single_entry(self, index: int) -> Vector:
        """Mutates entry.
        Changes the indexed value of the current point to some other possible option
        for that point.

        Args:
            index: The index of the entry to mutate

        Returns:
            A mutation of the current point
        """
        new_point = deepcopy(self.current_point)
        options = self.domain[index] - {self.current_point[index]}  # type: ignore

        if len(options) > 0:
            new_item = self.rng.choice(list(options))
            new_point[index] = new_item  # type: ignore

        return new_point

    def ask(self) -> Vector:
        """Get the current point.

        Returns:
            Matrix with a single row of parameters.
        """
        if self.schedule.current_t > self.maxiter:
            raise MaxIterationError(
                f"Simulated annealing cannot exceed its max iteration ({self.maxiter})."
            )

        if self.told and self.current_point is None:
            raise InitializationError(
                "Attempted to generate a new population without evaluating first."
            )

        if not self.told:
            return self.initial_point[np.newaxis, :]

        return self._generate_point()[np.newaxis, :]

    def tell(self, parameters: Matrix, objective_values: Vector) -> None:
        """Update the next point and internal state.

        Args:
            parameters: Matrix with a single row of parameter values.
            objective_values: Vector with a single objective value.
        """
        super().tell(parameters, objective_values)

        parameters, objective_values = self._process_told_parameters_and_objectives(
            parameters, objective_values
        )

        self.told = True
        self.niters += 1

        if parameters.shape != (1, len(self.domain)):
            raise ValueError(
                f"Got parameters shape {parameters.shape}, "
                f"required (1, {len(self.domain)})."
            )

        parameters = parameters.squeeze()
        objective_value = objective_values.item()

        if objective_value <= self.best_objective:
            self.current_objective = objective_value
            self.current_point = deepcopy(parameters)
            self.best_point = deepcopy(parameters)
            self.best_objective = objective_value

        elif self.rng.uniform() > np.exp(self.best_objective - objective_value) / next(
            self.schedule
        ):
            self.current_objective = objective_value
            self.current_point = deepcopy(parameters)

    def stop(self) -> bool:
        """Determine if we should stop iterating.

        Returns:
            Boolean, True if we should stop.
        """
        return self.niters >= self.maxiter

    def best(self) -> tp.Tuple[Vector, float]:
        """Get the best point and objective.

        Returns:
            The best found point and its objective.

        Raises:
            AttributeError: if no point have been evaluated.
        """
        if self.best_point is None:
            raise AttributeError("No points have been evaluated yet.")

        report_objective = (
            -self.best_objective if self.sense == MAXIMIZE else self.best_objective
        )
        return self.best_point, report_objective

    @property
    def _internal_objective_values(self) -> Vector:
        """Return the internal point's objective value.

        Returns:
            The current point's objective value as an array.
        """
        return np.array(self.current_objective)

    @property
    def parameters(self) -> Matrix:
        """Get the current parameter values.

        Returns:
            Matrix (single row) of parameters.
        """
        return self.current_point[np.newaxis, :]  # type: ignore
