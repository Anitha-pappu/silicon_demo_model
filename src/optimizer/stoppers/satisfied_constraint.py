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
Stopper which terminates when population meets constraints.
"""

import typing as tp
import warnings
from copy import deepcopy

import numpy as np

from optimizer.problem.problem import OptimizationProblem
from optimizer.solvers.base import Solver
from optimizer.stoppers.base import BaseStopper
from optimizer.stoppers.utils import top_n_indices


class SatisfiedConstraintsStopper(BaseStopper):
    """
    This class will stop the search if all constraints are met.
    It allows the optimization to stop when the `top_n`
    members of the parameter population satisfy *all* the constraints.

    Args:
        top_n: Number of the best solutions to check for constraint violations
        sense: Whether to "minimize" or "maximize" the objective.

    Examples
    >> stopper = SatisfiedConstraintsStopper(sense='maximize')
    >> while not solver.stop() or stopper.stop():
    >>      parameters = solver.ask()
    >>      objective_values = problem(parameters)
    >>      solver.tell(parameters, objective_values)
    >>      stopper.update(solver, problem)
    """

    def __init__(
        self,
        top_n: int = 1,
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_n = top_n

    def update(
        self,
        solver: tp.Optional[Solver] = None,
        problem: tp.Optional[OptimizationProblem] = None,
    ) -> None:
        """
        Updates current stop state to True when constraints
        for the top_n best performing parameter sets are all met.

        Raises:
            ValueError: if solver is None
            ValueError: if problem is None
            ValueError: is ``problem.penalties`` are empty or None

        Args:
            solver: solver object;
                used for getting current parameters and their objective
            problem: problem object; used to check us all the constraints are satisfied
        """
        if solver is None:
            raise ValueError("Please provide a solver")
        if problem is None:
            raise ValueError("Please provide a problem")

        if problem.sense != solver.sense:
            raise ValueError(
                f"Problem and Solver have different senses: "
                f"{problem.sense = }, {solver.sense = }",
            )

        if not problem.penalties:
            warnings.warn(
                "Provided optimization problem has no constraints to verify. "
                "Stopper will stop instantly.",
            )
            self._stop = True
            return

        # Evaluating the problem on solver.parameters will change the
        # internal state of problem.penalties. Save the existing penalties obj.
        original_penalties = deepcopy(problem.penalties)
        # Evaluate the problem at the current parameters
        problem(solver.parameters)
        top_idx = top_n_indices(solver.objective_values, solver.sense, self.top_n)
        # Make an array where each parameter violates a given constraint.
        is_violated_constraint = np.array(
            [penalty.constraint.violated[top_idx]  # type: ignore
             for penalty
             in problem.penalties]
        ).T
        # Stop when no members of the population of interest/parameters are violating
        self._stop = (is_violated_constraint.sum(axis=1) == 0).all()

        # reset the original penalties
        problem.penalties = original_penalties
