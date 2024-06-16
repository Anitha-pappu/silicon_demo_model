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

import logging
import operator
import typing as tp
import uuid
import warnings
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
import pandas as pd

import optimizer
from recommend.types import TIndexDType

from ._warnings import NotImprovedSolutionWarning  # noqa: WPS436

logger = logging.getLogger(__name__)

_TDataType = tp.Union[str, int, float, pd.Timestamp]

_MAX_KEYS_IN_REPR = 10

_TYPE = "type"


@dataclass(frozen=True)
class ExportColumns(object):
    objective: str = "objective"
    initial: str = "initial"
    optimized: str = "optimized"
    run_id: str = "run_id"
    is_successful_optimization: str = "is_successful_optimization"
    uplift: str = "uplift"


COLUMNS = ExportColumns()


@dataclass
class Solution(object):  # noqa: WPS214
    """
    Class for storing optimizations results

    Attributes:
        loggers: loggers with information about optimization process
        uses_discrete_solver: (deprecated) true if discrete solver was used

    Properties:
        problem: problem definition used in optimization
        row_to_optimize: initial row passed for optimization
        row_after_optimization: row after optimization (with updated controls)
        row_to_optimize_index: row's to optimize index
        run_id: uuid assigned to row (used to distinguish runs for same indexed rows)
        uplift: SIGNED value that stores optimization (after minus before) change
        objective_after_optimization:
        objective_before_optimization:
        is_successful_optimization: true if optimization was sucessful
        control_parameters_before: dict of parameters BEFORE opt
        control_parameters_after: dict of parameters AFTER opt
        context_parameters: parameters that were considered static during opt
        controls_domain: dict containing optimization ranges for each control

    Methods:
        to_series: exports solution into a table format

    """

    def __init__(
        self,
        problem: optimizer.StatefulOptimizationProblem,
        solver: optimizer.solvers.Solver,
        stopper: tp.Optional[optimizer.stoppers.BaseStopper],
        loggers: tp.List[optimizer.loggers.LoggerMixin],
    ) -> None:
        if not isinstance(problem.state, pd.DataFrame):
            raise ValueError("Switch to pd.DataFrame state in problem's definition")

        self._problem = problem
        self._solver = solver
        self._stopper = stopper
        self.loggers = loggers
        self._run_id = str(uuid.uuid4())

        # todo: remove this legacy behaviour on cra side
        self.uses_discrete_solver = isinstance(solver, optimizer.solvers.DiscreteSolver)

        self._controls_domains = dict(
            zip(problem.optimizable_columns, solver.domain),
        )

        self._control_variables = self._problem.optimizable_columns
        self._context_variables = list(
            self.row_to_optimize.columns.difference(self._control_variables),
        )

    @property
    def problem(self) -> optimizer.StatefulOptimizationProblem:
        return self._problem

    @property
    def row_to_optimize(self) -> pd.DataFrame:
        return self._problem.state

    @cached_property
    def row_after_optimization(self) -> pd.DataFrame:
        optimal_controls_with_state = self._problem.substitute_parameters(
            self._optimal_controls.reshape(1, -1),
        )
        optimal_controls_with_state.index = self.row_to_optimize.index
        return optimal_controls_with_state

    @property
    def row_to_optimize_index(self) -> TIndexDType:
        return self.row_to_optimize.index[0]

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def uplift(self) -> float:
        """
        Incremental objective uplift value calculated as
        objective after - objective before the optimization.
        Returns zero if not ``self.is_successful_optimization``.

        Notes:
            uplift is negative when solving a minimization task
            and positive when maximization

        Returns:
            uplift scalar
        """
        uplift = self.objective_after_optimization - self.objective_before_optimization
        return uplift if self.is_successful_optimization else 0.0  # noqa: WPS358

    @property
    def objective_after_optimization(self) -> float:
        """Objective value without penalties included and repairs applied"""
        # transform to single-row matrix since problem works on 2D inputs
        best_controls = self._optimal_controls.reshape(1, -1)
        objective, _ = self._problem(
            best_controls, apply_penalties=False, apply_repairs=False,
        )
        return float(objective)  # extract our single objectives from array

    @property
    def objective_before_optimization(self) -> float:
        """Objective value without penalties included and repairs applied"""
        return float(self._problem.objective(self.row_to_optimize))

    @property
    def is_successful_optimization(self) -> bool:
        worse_comp: tp.Callable[[float, float], bool] = (
            operator.gt if self._problem.sense == "minimize" else operator.lt
        )
        return worse_comp(
            self.objective_before_optimization,
            self.objective_after_optimization,
        )

    @property
    def control_parameters_before(self) -> tp.Dict[str, float]:
        row_to_optimize = self.row_to_optimize.iloc[0]
        controls: tp.Dict[str, float] = (
            row_to_optimize[self._control_variables].to_dict()
        )
        # we're adding typing to avoid extra casts
        return controls  # noqa: WPS331

    @property
    def control_parameters_after(self) -> tp.Dict[str, float]:
        return dict(zip(self._control_variables, self._optimal_controls))

    @cached_property
    def context_parameters(self) -> tp.Dict[str, _TDataType]:
        row_to_optimize = self.row_to_optimize.iloc[0]
        context: tp.Dict[str, _TDataType] = (
            row_to_optimize[self._context_variables].to_dict()
        )
        # we're adding typing to avoid extra casts
        return context  # noqa: WPS331

    @property
    def controls_domain(self) -> tp.Dict[str, tp.Tuple[float, float]]:
        """Returns domain of the control ``control_name``"""
        return self._controls_domains.copy()

    def to_series(self) -> pd.Series:
        """
        Extract comparison of initial and optimized optimizable parameters
        in a table format.

        Returns: a series with comparison of initial and optimized rows
        """
        df_before_after = self._get_before_after_comparison_for_all_columns()
        df_before_after = self._flatten_comparison_table(df_before_after)
        df_before_after[COLUMNS.run_id] = self.run_id
        df_before_after[COLUMNS.is_successful_optimization] = (
            self.is_successful_optimization
        )
        df_before_after[COLUMNS.uplift] = self.uplift
        return df_before_after.iloc[0]  # to series

    @property
    def _optimal_controls(self) -> npt.NDArray["np.generic"]:
        optimal_controls: npt.NDArray["np.generic"]
        optimal_controls, _ = self._solver.best()
        return optimal_controls

    def _get_before_after_comparison_for_all_columns(self) -> pd.DataFrame:
        """
        Warnings:
            NotImprovedSolutionWarning: is thrown when proposed controls
                don't improve objective
        """
        # df with current and with optimal controls
        df_before_after = pd.concat(
            {
                COLUMNS.initial: self.row_to_optimize,
                COLUMNS.optimized: self.row_after_optimization,
            },
            names=[_TYPE, self.row_to_optimize.index.name],
        )
        # Evaluate objective without applying penalties/repairs
        df_before_after[COLUMNS.objective] = self._problem.objective(df_before_after)
        penalty_slack_table = _get_penalty_and_slack(
            df_before_after, self._problem.penalties,
        )
        df_before_after = df_before_after.join(penalty_slack_table, how="left")
        if not self.is_successful_optimization:
            warnings.warn(
                f"Row index {self.row_to_optimize_index}. "
                f"Objective did not improve after optimization "
                f"({self.objective_after_optimization = :0.4f}, "
                f"{self.objective_before_optimization = :0.4f}). "
                f"Initial controls will be returned.",
                category=NotImprovedSolutionWarning,
            )
            self._reset_all_parameters_to_initial(df_before_after)
        return df_before_after

    def _flatten_comparison_table(
        self, df_before_after: pd.DataFrame,
    ) -> pd.DataFrame:
        flat_df_with_comparison_for_all_columns = df_before_after.unstack(_TYPE)
        return flat_df_with_comparison_for_all_columns.drop(
            columns=[
                (column, COLUMNS.optimized)
                for column in self._context_variables
            ],
            errors="ignore",
        )

    def _reset_all_parameters_to_initial(self, df_before_after: pd.DataFrame) -> None:
        """Replaces solutions with objective lower than initial to initial state."""

        row_index = self.row_to_optimize_index
        df_before_after.loc[(COLUMNS.optimized, row_index)] = (
            df_before_after.loc[(COLUMNS.initial, row_index)].copy()
        )


def _get_penalty_and_slack(
    parameters: pd.DataFrame, penalties: tp.List[optimizer.Penalty],  # noqa: WPS110
) -> pd.DataFrame:
    """
    Generates penalties and slack for all constraints.
    Slack is calculated only for inequality constraints.

    Args:
        parameters: parameters table to evaluate penalties on.
        penalties: list of penalties.

    Returns:
        pd.DataFrame: tables with penalties and slacks.
    """
    if not penalties:
        return pd.DataFrame(index=parameters.index)

    for penalty in penalties:  # re-evaluate penalties based on new parameters
        penalty(parameters)

    penalty_table = optimizer.diagnostics.get_penalties_table(penalties)
    slack_table = optimizer.diagnostics.get_slack_table(penalties)
    return penalty_table.join(slack_table, how="left")
