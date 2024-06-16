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
Problem set-up diagnostics functions
"""

import typing as tp
from operator import le, lt

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from optimizer import Penalty
from optimizer.constraint import InequalityConstraint
from optimizer.problem.problem import OptimizationProblem
from optimizer.solvers.continuous.base import to_limits
from optimizer.types import (
    PenaltyCompatible,
    ReducesMatrixToSeries,
    StoresPenaltyValues,
    TBoundsList,
    Vector,
)
from optimizer.utils.functional import get_penalty_name

_NON_CONVEX_EPS = 1e-6


def evaluate_on_historic_data(
    data: pd.DataFrame,
    problem: OptimizationProblem,
    bounds: TBoundsList,
) -> pd.DataFrame:
    """
    Evaluates the problem statement on historical data. This is
    helpful to sanity check the constraints and problem set-up.

    Args:
        data: input data
        problem: Optimization Problem
        bounds: List of bounds for each variables in the input data. It is expected
        that the bounds are ordered tuples (i.e. (min, max) and is provided for
        each column in the input data.)

    Returns:
        A dataframe with input data along with the evaluated penalties,
        slack, bounds check and objective values. If the inequality constraint
        is violated, the slack is NaN.

    Raises:
        ValueError: if number of columns in data does not match
        the number of bounds

    """
    if len(data.columns) != len(bounds):
        raise ValueError("Bounds list length do not match the number of columns")

    return _evaluate_on_historic_data(data, problem, bounds)


def _evaluate_on_historic_data(
    data: pd.DataFrame, problem: OptimizationProblem, bounds: TBoundsList,
) -> pd.DataFrame:
    data = data.copy()
    check_bounds = _check_bounds(bounds, data)
    # we re-run penalties on the user provided data;
    # i.e., we don't want to perform a state substitution
    for p in problem.penalties:
        p(data)
    penalty_table = get_penalties_table(problem.penalties)
    objective_values = _get_objective_value(data, problem)
    slack_table = get_slack_table(problem.penalties)
    return (
        data.join(check_bounds, how="left")
        .join(penalty_table, how="left")
        .join(slack_table, how="left")
        .join(objective_values, how="left")
    )


def _get_objective_value(
    data: pd.DataFrame, problem_stmt: OptimizationProblem,
) -> pd.DataFrame:
    """
    Computes the objective value with and without penalty.
    Here, we pass the inputs to the "objective" function of the problem class, so that
    the states are not replaced to the user provided values.
    Similarly, the "apply_penalty" function is used to account for both maximization
    and minimization use-case.
    """

    objective_value = pd.Series(
        problem_stmt.objective(data), name="objective_value", index=data.index
    )

    # Note that we are adding penalty separately to the objective value rather
    # than calling problem_stmt(data), so that the states are historical values
    # from the data and not those provided by user
    objective_value_with_penalty = pd.Series(
        objective_value + problem_stmt.calculate_penalty(data),
        name="objective_value_with_penalty",
    )

    return pd.concat([objective_value, objective_value_with_penalty], axis=1)


def _check_bounds(bounds: TBoundsList, data: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if each column is within the specified bounds. It is expected
    that the bounds are ordered tuples (i.e. (min, max) and provided for each column
    in the dataset and in the same order as the dataset.)
    Returns a dataframe with boolean flags.
    """
    bounds_stack = pd.DataFrame(
        np.stack(bounds), columns=["min", "max"], index=data.columns
    )
    bounds_table = pd.DataFrame(
        np.stack(
            [
                (bounds_stack.loc[col, "min"] <= data[col])
                & (data[col] <= bounds_stack.loc[col, "max"])
                for col in data.columns
            ],
            axis=1,
        ),
        columns=["within_bounds_" + col_name for col_name in data.columns],
        index=data.index,
    )
    return bounds_table


def get_penalties_table(penalties: tp.Sequence[PenaltyCompatible]) -> pd.DataFrame:
    """
    Computes absolute values for all penalties on the given data.
    Returns a dataframe with calculated penalties.

    Args:
        penalties: PRE-EXECUTED penalty-constraints (since we have only
            optimal parameters in our solutions, that won't include state
            for ContextfulOptimizationProblem, we ask users to evaluate penalties
            on their own using the problem's state if needed)

    Returns:
        Dataframe with penalties for each constraint
    """
    if not penalties:
        return pd.DataFrame()

    penalties = [p for p in penalties if isinstance(p, PenaltyCompatible)]
    _validate_penalties_were_executed(penalties)

    penalty_matrix = np.stack(
        [p.calculated_penalty for p in penalties],  #type: ignore
        axis=1,
    )
    penalty_names = [
        get_penalty_name(p, f"penalty_{index}") for index, p in enumerate(penalties)
    ]
    index = _resolve_index(penalties)
    return pd.DataFrame(penalty_matrix, columns=penalty_names, index=index)


def get_slack_table(
    penalties: tp.Sequence[PenaltyCompatible],
    mask_negative_slack: bool = True,
) -> pd.DataFrame:
    """
    Extracts slack (distance between function and boundary) for the
    ``InequalityConstraints`` in the penalties list.
    The value denotes the amount of slack left before the constraint is violated.

    If the penalties are not named, their slack values are returned as
    ``[f"penalty_{i}_slack" for i in range(penalties)]``

    Args:
        penalties: PRE-EXECUTED penalty-constraints (since we have only
            optimal parameters in our solutions, that won't include state
            for ContextfulOptimizationProblem, we ask users to evaluate penalties
            on their own using the problem's state if needed)
        mask_negative_slack: since slack is positive amount of space left for opt.
            all negative values are considered as no slack left; hence are masked to nan
            if `True` otherwise negative values are returned.

    Returns:
        Dataframe with a slack for each inequality penalty constraint

    """
    inequality_penalties: tp.Sequence[Penalty] = [
        p
        for p in penalties
        if isinstance(p, Penalty) and isinstance(p.constraint, InequalityConstraint)
    ]

    if not inequality_penalties:
        return pd.DataFrame()

    _validate_penalties_were_executed(inequality_penalties)

    # negative distance is the space left => negate distances matrix
    slack_matrix = -np.stack([p.distances for p in inequality_penalties], axis=1)  # pylint: disable=(invalid-unary-operand-type
    if mask_negative_slack:
        slack_matrix = np.where(slack_matrix < 0, np.NaN, slack_matrix)
    columns = [
        get_penalty_name(
            penalty, f"penalty_{index}_slack",
        ).replace("_penalty", "_slack")
        for index, penalty in enumerate(inequality_penalties)
    ]
    slack_table = pd.DataFrame(
        slack_matrix, columns=columns, index=_resolve_index(penalties),
    )
    return slack_table


def check_is_non_convex(  # pylint: disable=too-many-locals
    objective: ReducesMatrixToSeries,
    domain: TBoundsList,
    n_samples: int = 10000,
    n_points_on_line: int = 10,
    strict: bool = False,
    seed: tp.Optional[int] = None,
    return_hits: bool = False,
) -> tp.Union[bool, tp.Tuple[bool, int]]:
    """
    Performs a simple test for non-convexity exploiting the definition
    of a convex function.

    In summary, this function randomly samples `n_samples` pairs of points and draws a
    line segment between them. It then tests `n_points_on_line` values on the line using
    the definition of convexity. If the test

    f(t * x_1 + (1 - t) * x_2) <= t * f(x_1) + (1 - t) * f(x_2)

    fails for any of the samples, we can conclude `objective` is non-convex. The t
    values will be generated at even intervals on the line between x_1 and x_2.

    See here for more on convex functions:
        https://en.wikipedia.org/wiki/Convex_function

    ** Note: this test returning a False result does NOT prove convexity.
    ** Note: this function involves 3 calls to the objective function using matrices
    with dimension (`n_samples` * `n_points_on_line`, len(domain))
    ** Note: by definition, a function defined on an integer domain is not convex.
    Consider converting your discrete domain to a continuous one when using this
    function.
    ** Note: when using this test for an optimization problem, an objective may seem
    convex, but still be part of a non-convex optimization problem if the constraints
    are non-convex.

    Args:
        objective: callable, the function to test for non-convexity.
        domain: list of tuples, the boundaries of the objective function.
        n_samples: number of pairs of points to sample.
        n_points_on_line: number of points to test on the line segment formed between
            the two randomly sampled points.
        strict: boolean, False to allow for equality in the non-convexity test.
        seed: random seed.
        return_hits: boolean, True to return the number of times the test failed.

    Returns:
        True if the `objective` is non-convex.
        If `return_hits` is true, the number of times the test failed will be returned.
    """
    rng = check_random_state(seed)
    comparison = lt if strict else le

    limits = to_limits(domain)

    line_segment_samples = np.linspace(0, 1, num=n_points_on_line, endpoint=False)[1:]

    # Sample from [0, 1) and convert to the desired domain.
    sample = rng.random_sample(size=(n_samples * 2, len(domain)))
    sample = limits[0] + (limits[1] - limits[0]) * sample

    sample_x1, sample_x2 = sample[:n_samples], sample[n_samples:]

    # In order to make the testing operation fast, we repeat each sample for each
    # point on the line we'd like to test.
    sample_x1 = np.repeat(sample_x1, len(line_segment_samples), axis=0)
    sample_x2 = np.repeat(sample_x2, len(line_segment_samples), axis=0)

    # Similar to the above operation, we tile the line segment t values to obtain a
    # different t for each repeated sample.
    tiled_segment_samples = np.tile(line_segment_samples, n_samples)

    lhs = objective(
        tiled_segment_samples.reshape(-1, 1) * sample_x1
        + (1 - tiled_segment_samples.reshape(-1, 1)) * sample_x2
    )
    rhs = (
        tiled_segment_samples * objective(sample_x1)
        + (1 - tiled_segment_samples) * objective(sample_x2)
    )

    tests = comparison(lhs, rhs + (_NON_CONVEX_EPS if not strict else 0))
    test_result = not all(tests)

    if return_hits:
        return test_result, np.sum(~tests).item()

    else:
        return test_result


def estimate_smoothness(  # pylint: disable=too-many-locals
    objective: ReducesMatrixToSeries,
    domain: TBoundsList,
    n_samples: tp.Union[int, Vector] = 10000,
    seed: tp.Optional[int] = None,
    return_list: bool = False,
) -> tp.Union[float, tp.Tuple[float, tp.List[float]]]:
    """Estimate the smoothness (condition number) of a function.

    This implements an estimate of the relative condition number for a random sample of
    points from the domain.

    See here for more on the condition number for several variables:
        https://en.wikipedia.org/wiki/Condition_number#Several_variables

    For a simplified formula, also see here:
        https://math.stackexchange.com/q/736022

    Note: this estimate is meant to compare functions on similar ranges. Comparing
    functions with very different ranges may lead to incorrect conclusions.

    Args:
        objective: callable, function to test smoothness.
        domain: list of tuples, the boundaries of the objective function.
        n_samples: number of points to estimate the relative condition number or a
            numpy array of points to calculate condition numbers at.
        seed: random seed.
        return_list: boolean, True to return the list of all estimated condition numbers

    Returns:
        float, the mean relative condition number. If ``return_list`` is true, all
        estimates will also be returned.
    """
    rng = check_random_state(seed)
    limits = to_limits(domain)

    if isinstance(n_samples, int):
        samples = limits[0] + (limits[1] - limits[0]) * rng.rand(n_samples, len(domain))
    else:
        samples = np.array(n_samples, dtype=float)
        n_samples = int(samples.shape[0])

        if len(domain) != samples.shape[1]:
            raise ValueError(
                f"Given domain has {len(domain)} dimensions, but provided samples "
                f"have {samples.shape[1]} dimensions. Must be equal."
            )

    x_normed = np.linalg.norm(samples, ord=2, axis=1)
    fx_abs = np.abs(objective(samples))

    #
    # See here for an explanation on the arithmetic below in 1D: https://w.wiki/4Uuw
    # Here, matrices are constructed to vectorize the Jacobian estimate rather than
    # looping over the samples matrix and computing one at a time.
    #

    # Repeat the samples rowwise, once for each dimension.
    samples_repeated = np.repeat(samples, len(domain), axis=0)

    # Tile the identity matrix and multiply by the repeated samples to get only the
    # diagonals the of the repeated samples.
    tiled_eye = np.tile(np.eye(len(domain)), (n_samples, 1))

    # Compute the steps in each direction using an appropriate step size based on dtype.
    # See page 7 of:
    #   http://paulklein.ca/newsite/teaching/Notes_NumericalDifferentiation.pdf
    eps = np.power(np.finfo(samples.dtype).eps, 1 / 3)
    steps = (np.maximum(np.abs(samples_repeated), 1) * tiled_eye) * eps

    # Compute how far we stepped directly rather than reusing the above epsilon.
    forward_step = samples_repeated + steps
    backward_step = samples_repeated - steps
    dx = forward_step - backward_step

    # Get the diagonal of each stacked dx matrix.
    dx = dx[
        np.arange(dx.shape[0]),
        np.tile(np.arange(dx.shape[1]), n_samples),
    ]

    # Estimate the Jacobian and take the L2 norm.
    jacobian_norm = np.linalg.norm(
        (  # Center point method for each partial derivative in the Jacobian.
            (objective(forward_step) - objective(backward_step)) / dx
        ).reshape(
            samples.shape
        ),  # Reshape to take the Jacobian across rows.
        ord=2,
        axis=1,
    )

    # Calculate the array of relative condition numbers.
    relative_condition_numbers = (x_normed * jacobian_norm) / fx_abs
    mean_condition: float = np.mean(relative_condition_numbers).item()

    if return_list:
        return mean_condition, list(relative_condition_numbers)

    else:
        return mean_condition


def _validate_penalties_were_executed(
    penalties: tp.Sequence[StoresPenaltyValues],
) -> None:
    not_executed_penalties = tuple(
        index for index, penalty in enumerate(penalties)
        if penalty.calculated_penalty is None
    )
    if not_executed_penalties:
        raise ValueError(
            f"Penalties with indices {not_executed_penalties} were not evaluated. "
            f"Please evaluate them by calling `for penalty in penalties: "
            f"penalty(data)` or `problem(parameters)` "
            f"if you are extracting penalties from problem.",
        )


def _resolve_index(
    penalties: tp.Sequence[StoresPenaltyValues],
) -> tp.Optional[tp.Iterable[tp.Any]]:
    for penalty in penalties:
        if (
            isinstance(penalty, StoresPenaltyValues)
            and isinstance(penalty.calculated_penalty, pd.Series)
        ):
            return tp.cast(tp.Iterable[tp.Any], penalty.calculated_penalty.index)
    return None
