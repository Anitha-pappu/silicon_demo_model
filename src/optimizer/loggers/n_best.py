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
N-Best logger module.
"""

import heapq
import typing as tp
from copy import deepcopy

import numpy as np

from optimizer import OptimizationProblem
from optimizer.loggers.base import LoggerMixin
from optimizer.solvers import Solver
from optimizer.types import MINIMIZE, Sense, TSense, Vector


class NBestLogger(LoggerMixin):
    def __init__(self, n: int, sense: TSense = MINIMIZE):
        """
        Logs the N-best solutions over time.

        Args:
            n: number of solutions to keep track of.
            sense: str, "minimize" or "maximize".
        """
        if n <= 0:
            raise ValueError(f"n must be greater than zero, got {n}.")

        if sense not in Sense:
            raise ValueError(f'sense must be "minimize" or "maximize", got {sense}.')

        self._n = n
        self.log_records: tp.List[tp.Any] = []
        self._sense = sense

    def clone(self) -> 'NBestLogger':
        return NBestLogger(self._n, self._sense)

    def log(
        self,
        *,
        problem: tp.Optional[OptimizationProblem] = None,  # pylint: disable=unused-argument
        solver: tp.Optional[Solver] = None,
    ) -> None:
        """
        Updates the N-best ``solver``'s values.

        Args:
            problem: is not used
            solver: used for solving a problem

        Raises:
            ValueError: if ``solver`` is not provided
        """

        if solver is None:
            raise ValueError("Please provide a solver")

        parameters, objectives = solver.parameters, solver.objective_values

        sign = 1 if self._sense == MINIMIZE else -1
        solutions = [(sign * o.item(), p) for o, p in zip(objectives, parameters)]
        solutions = heapq.nsmallest(self._n, solutions)

        if not self.log_records:
            # We don't have a heap yet, so just get the smallest solutions.
            self.log_records = solutions
            return

        new_heap: tp.List[tp.Any] = []
        # Loop while we have enough solutions to push on the heap and the new
        # heap is less than the desired length.
        while (solutions or self.log_records) and len(new_heap) < self._n:
            if not self.log_records:  # Stored heap is empty.
                heapq.heappush(new_heap, heapq.heappop(solutions))
            elif not solutions:  # Candidate solutions are empty.
                heapq.heappush(new_heap, heapq.heappop(self.log_records))
            elif solutions[0][0] < self.log_records[0][0]:
                heapq.heappush(new_heap, heapq.heappop(solutions))
            else:
                heapq.heappush(new_heap, heapq.heappop(self.log_records))
        self.log_records = new_heap

    @property
    def n_best(self) -> tp.Tuple[Vector, Vector]:
        """Get the N best logged solutions in sorted order and their objective values.

        If M < N values are logged throughout the lifetime of this object, this function
        will return M total values rather than N.

        Returns:
            Tuple of N-best parameters and corresponding objectives.
            The first has shape (N, d), the second (N,).
            Where d is the dimension of the problem being solved.
        """
        heap = deepcopy(self.log_records)

        parameters, objectives = [], []

        sign = 1 if self._sense == MINIMIZE else -1
        while heap:
            solution = heapq.heappop(heap)

            parameters.append(solution[1])
            objectives.append(sign * solution[0])

        return np.array(parameters), np.array(objectives)
