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

import functools
import logging
import typing as tp

import pandas as pd
from joblib import Parallel, delayed

from optimizer.loggers import LoggerMixin
from optimizer.solvers import Solver
from optimizer.stoppers.base import BaseStopper
from optimizer.types import TSense
from recommend.problem_factory import ProblemFactoryBase
from recommend.solution import Solution, Solutions
from recommend.solver_factory import SolverFactory
from recommend.types import TProblem
from recommend.warnings_catcher import (
    TWarningsLevelBasic,
    TWarningsLevelExtended,
)

from ._warnings import (  # noqa: WPS436
    OutOfDomainCatcher,
    get_per_artefact_warning_level,
)

logger = logging.getLogger(__name__)

_SENSE_KEY = "sense"
_TFunction = tp.Callable[..., None]
_TSolutionArtefacts = tp.Tuple[
    TProblem, Solver, tp.Optional[BaseStopper], tp.List[LoggerMixin],
]


class DuplicateIndexError(Exception):
    """Raise when data contains duplicate index."""


class SenseError(Exception):
    """Raise when problem and solver have different sense of optimization."""


def optimize(
    data_to_optimize: pd.DataFrame,
    problem_factory: ProblemFactoryBase[TProblem],
    solver_factory: SolverFactory,
    loggers: tp.Optional[tp.List[LoggerMixin]] = None,
    *,
    n_jobs: float = 1,
    warnings_details_level: TWarningsLevelExtended = "aggregated",
) -> Solutions:
    """
    Run parallel optimization for multiple rows of data.

    Args:
        data_to_optimize: dataset to optimize;
            rows are optimized independently; each row is considered to contain
            a full state of all plant conditions for a given moment of time
        problem_factory: factory producing problems for each row of ``data``
        solver_factory: factory producing solvers for each row of ``data``
        loggers: list of loggers to trigger on each iteration of ask-tell loop of
            optimization; those loggers will be used as prototypes
            i.e., we'll create new logger via `.clone()`
            for each row in ``data_to_optimize``;
            per each solution its own ``loggers`` list will be returned
        n_jobs: the number of cores used for parallel processing
        warnings_details_level: controls warnings' level of verbosity;
            "none" – don't show any warnings,
            "row_aggregated" - show per row aggregated warnings,
            "aggregated" - show overall aggregated warnings,
            "detailed" – show detailed warnings

    Raises:
        DuplicateIndex: if input data contains duplicated index.

    Returns:
        Solutions, a mapping from ``data.index`` into ``Solution`` object
    """
    if data_to_optimize.index.has_duplicates:
        raise DuplicateIndexError("`data_to_optimize` must have unique index.")

    per_artefact_warning_level = get_per_artefact_warning_level(warnings_details_level)
    with OutOfDomainCatcher(warnings_details_level, len(data_to_optimize)):
        solution_artefacts = _prepare_solution_artefacts(
            data_to_optimize,
            problem_factory,
            solver_factory,
            loggers,  # type: ignore
            per_artefact_warning_level,
        )
    backup_problems = [problem for problem, _, _, _ in solution_artefacts]

    # since Parallel doesn't mutate initial objects with some backends,
    # we'll re-collect updated objects
    solution_artefacts = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_solve_problem)(*artefacts) for artefacts in solution_artefacts
    )
    solution_artefacts = _optimize_artifacts_memory_consumption(
        backup_problems, solution_artefacts,
    )
    return Solutions(Solution(*artefacts) for artefacts in solution_artefacts)


def _safe_run(func: _TFunction) -> _TFunction:
    """
    Decorator for handling keyboard interrupt in ``optimize`` function

    Args:
        func: function to decorate
    """

    @functools.wraps(func)
    def wrapper(*args: tp.Any, **kwargs: tp.Any) -> None:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.info(
                msg=f"Execution halted by user, raising KeyboardInterrupt."
                    f"Was executing {args} and {kwargs}",
            )
            return None

    return wrapper


@_safe_run  # type: ignore
def _solve_problem(
    problem: TProblem,
    solver: Solver,
    stopper: tp.Optional[BaseStopper],
    loggers: tp.List[LoggerMixin],
) -> _TSolutionArtefacts[TProblem]:
    """
    Optimizes a single row;
    i.e., updates ``problem``, ``solver``, and ``stopper`` states

    Args:
        problem: problem to solve
        solver: solver to use in optimization
        stopper: optional stopper to use for early stopping

    Raise:
        SenseError: if problem.sense != solver.sense

    Returns:
        tuple of inplace-updated input objects
    """
    while True:
        candidate_parameters = solver.ask()
        obj_vals, fixed_candidate_parameters = problem(candidate_parameters)
        solver.tell(fixed_candidate_parameters, obj_vals)

        for single_logger in loggers:
            single_logger.log(problem=problem, solver=solver)

        if _check_is_finished(problem, solver, stopper):
            break

    return problem, solver, stopper, loggers


def _check_is_finished(
    problem: TProblem, solver: Solver, stopper: BaseStopper,
) -> bool:
    """
    Checks if solver has run out of iterations (``solver.stop()``)
    or stopper criteria are triggered (``stopper.stop()``)
    """
    solver_finished = solver.stop()
    if stopper is None:
        return tp.cast(bool, solver_finished)
    stopper.update(solver, problem)
    any_finished = stopper.stop() | solver_finished
    return tp.cast(bool, any_finished)


def _validate_senses_match(problem_sense: TSense, solver_sense: TSense) -> None:
    if problem_sense != solver_sense:
        raise ValueError(
            f"Problem's sense = {problem_sense} doesn't match "
            f"solver's sense = {solver_sense}",
        )


def _prepare_solution_artefacts(
    data: pd.DataFrame,
    problem_factory: ProblemFactoryBase[TProblem],
    solver_factory: SolverFactory,
    loggers: tp.List[LoggerMixin],
    per_artefact_warning_level: TWarningsLevelBasic,
) -> tp.List[_TSolutionArtefacts[TProblem]]:
    solution_artefacts = []
    logger.info("Creating problem, solver, and stopper for each row")
    for index in data.index:
        row = data.loc[[index]]
        problem = problem_factory.create(row)
        solver, stopper = solver_factory.create(
            row, problem.optimizable_columns, per_artefact_warning_level,
        )
        _validate_senses_match(problem.sense, solver.sense)
        loggers = (
            [single_logger.clone() for single_logger in loggers]
            if loggers is not None
            else []
        )
        solution_artefacts.append((problem, solver, stopper, loggers))
    return solution_artefacts


def _optimize_artifacts_memory_consumption(
    problems: tp.List[TProblem],
    solution_artefacts: tp.List[_TSolutionArtefacts[TProblem]],
) -> tp.List[_TSolutionArtefacts[TProblem]]:
    """
    When joblib runs optimization in parallel, each artifact is copied
    (this is how multiprocessing in python works).

    Each problem from artifacts tuple stores a reference to a model
    from model registry (if you use one to evaluate an objective/penalty/repair/etc.).
    So once a parallel process is created, a model is copied
    and the returned artifacts tuple contains a copy of a model
    instead of a reference to the model registry.

    This leads to excessive RAM consumption and memory when you try to persist it.
    A simple sklearn pipeline model consumes ~0.5M space, imagine what happens when you
    persist 1000 rows where each contains at least one model copy.
    That will already be a 0.5G file.

    To avoid this issue, we refer to the fact that each
    problem is stateless.
    So we can reuse a problem object created before optimization
    which stores a reference, not a copy of a model.
    """
    return [
        (problem, solver, stopper, logger_)
        for problem, (_, solver, stopper, logger_) in zip(problems, solution_artefacts)
    ]
