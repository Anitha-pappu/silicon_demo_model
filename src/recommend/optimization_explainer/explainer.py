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

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure as PlotlyFigure

from optimizer.constraint.handler import BaseHandler
from optimizer.types import ReducesMatrixToSeries

from ..solution import Solution
from .extraction import (  # noqa: WPS450
    OptimizationExplanation,
    _Dependency,
    _Point,
    create_repair_violations,
    extract_dependent_function_value,
    get_problem_objective,
)
from .grid import create_grid_for_optimizable_parameter
from .layout import create_layout_for_optimization_explainer
from .trace import (
    add_optimization_limits,
    add_vertical_line,
    create_initial_position_scatter_trace,
    create_line_trace,
    create_optimized_position_scatter_trace,
    create_repairs_scatter_traces_from_violations,
)


def create_optimization_explainer_plot(
    optimizable_parameter: str,
    solution: Solution,
    dependent_function: ReducesMatrixToSeries = None,
    state: tp.Optional[pd.DataFrame] = None,
    n_points_in_grid: tp.Optional[int] = None,
    x_axis_name: tp.Optional[str] = None,
    y_axis_name: tp.Optional[str] = None,
) -> PlotlyFigure:
    """
    Create plot that explains why optimized
    value is better than initial value of the parameter.

    Shows the dependencies between different values
    of ``optimizable_parameter`` and ``dependent_function(optimizable_parameter)``.
    Highlights areas constrained with repairs
    and marks initial and optimized values for ``optimizable_parameter``.

    Args:
        dependent_function: Function used for Y-axis that maps
         row with parameters into number.
         Can be optimization objective, penalty,
         or anything else aligned with the API.
         If None, then objective with penalties is considered.
        optimizable_parameter: Optimization parameter name involved in optimization
        solution: solution produced by the Optimizer class
        state: Values used for other parameters needed
         to calculate ``dependent_function(optimizable_parameter)``;
         if None, ``solution.row_after_optimization`` is considered
        n_points_in_grid: number of points to use
         in grid if grid can not be produced from problem
        x_axis_name: Name for X-axis to use. Might be useful
         when parameters involved into optimization don't have human-readable name
        y_axis_name: Name for X-axis to use. By 'dependent_function' class name
         is used by default. Hence, this parameter is useful,
         when default class name is not intuitive
    Returns:
        plotly.Figure with optimization explanation
    """
    optimization_explanation = create_optimization_explanation(
        optimizable_parameter,
        solution,
        dependent_function,
        state=state,
        n_points_in_grid=n_points_in_grid,
        dependent_function_name=None,
    )
    return create_plot_from_optimization_explanation(
        optimization_explanation,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
    )


# WPS210 is silenced to avoid having too heavy dataclass definition
def create_optimization_explanation(  # noqa: WPS210
    optimizable_parameter: str,
    solution: Solution,
    dependent_function: ReducesMatrixToSeries = None,
    dependent_function_name: str | None = None,
    state: tp.Optional[pd.DataFrame] = None,
    n_points_in_grid: tp.Optional[int] = None,
) -> OptimizationExplanation:
    """
    Create optimization explanation dataclass for
    the input optimizable parameter using the `Solution` object provided.

    Args:
        optimizable_parameter: Optimization parameter name involved in optimization
        solution: solution produced by the Optimizer class
        dependent_function: Function used for Y-axis that maps
         row with parameters into number.
         Can be optimization objective, penalty,
         or anything else aligned with the API.
         If None, then objective with penalties is considered.
        dependent_function_name: Name for dependent function that will be used.
         If None, will be extracted from dependent_function
        state: Values used for other parameters needed
         to calculate ``dependent_function(optimizable_parameter)``;
         if None, ``solution.row_after_optimization`` is considered
        n_points_in_grid: number of points to use
         in grid if grid can not be produced from problem

    Returns:
        OptimizationExplanation object that stores
        all the information needed for explanation.
    """
    if state is None:
        state = solution.row_after_optimization
    if dependent_function is None:
        dependent_function = get_problem_objective(
            solution.problem, apply_penalties=True, apply_repairs=False,
        )
    dependent_function_name = (
        dependent_function_name if dependent_function_name is not None
        else _extract_name(dependent_function)
    )
    initial_parameter_position = (
        solution.control_parameters_before[optimizable_parameter]
    )
    initial_dependent_function_value = extract_dependent_function_value(
        dependent_function=dependent_function,
        state=state,
        optimizable_parameter_value=initial_parameter_position,
        optimizable_parameter=optimizable_parameter,
    )
    optimized_parameter_position = (
        solution.control_parameters_after[optimizable_parameter]
    )
    optimized_dependent_function_value = extract_dependent_function_value(
        dependent_function=dependent_function,
        state=state,
        optimizable_parameter_value=optimized_parameter_position,
        optimizable_parameter=optimizable_parameter,
    )
    bounds = solution.controls_domain[optimizable_parameter]
    parameter_grid = create_grid_for_optimizable_parameter(
        repairs=solution.problem.repairs,
        optimizable_parameter=optimizable_parameter,
        bounds=bounds,
        initial_position=initial_parameter_position,
        optimized_position=optimized_parameter_position,
        n_points_in_grid=n_points_in_grid,
    )
    state_with_grid = _fill_state_with_grid(
        parameter_grid=parameter_grid,
        state=state,
        optimizable_parameter=optimizable_parameter,
    )
    dependent_function_values = dependent_function(state_with_grid)
    repair_violations = create_repair_violations(
        repairs=solution.problem.repairs,
        state_with_grid=state_with_grid,
    )
    return OptimizationExplanation(
        explained_parameter=optimizable_parameter,
        initial_point=_Point(
            x_axis=initial_parameter_position,
            y_axis=initial_dependent_function_value,
        ),
        problem_sense=solution.problem.sense,
        optimized_point=_Point(
            x_axis=optimized_parameter_position,
            y_axis=optimized_dependent_function_value,
        ),
        dependency=_Dependency(
            dependent_function_name=dependent_function_name,
            x_axis=parameter_grid,
            y_axis=dependent_function_values,
        ),
        opt_bounds=bounds,
        repair_violations=repair_violations,
    )


def create_plot_from_optimization_explanation(
    explanation: OptimizationExplanation,
    x_axis_name: str | None = None,
    y_axis_name: str | None = None,
) -> go.Figure:
    """
    Create optimization explanation plot from the existing explanation object.

    If y_axis_name is provided, then it will be used as a name for the Y-axis
    and explanation.dependency.dependency_function_name will be ignored.
    """
    repair_violations = explanation.repair_violations
    initial_point = explanation.initial_point
    optimized_point = explanation.optimized_point
    dependency = explanation.dependency
    fig = go.Figure(
        data=[
            *create_repairs_scatter_traces_from_violations(
                parameter_grid=dependency.x_axis,
                repair_violations=repair_violations,
                initial_dependent_function_value=initial_point.y_axis,
                dependent_function_values=dependency.y_axis,
            ),
            create_line_trace(
                parameter_grid=dependency.x_axis,
                dependent_function_values=dependency.y_axis,
            ),
            create_initial_position_scatter_trace(
                initial_position=initial_point.x_axis,
                initial_dependent_function_value=initial_point.y_axis,
            ),
            create_optimized_position_scatter_trace(
                position=optimized_point.x_axis,
                dependent_function_value=optimized_point.y_axis,
            ),
        ],
        layout=create_layout_for_optimization_explainer(
            explanation.explained_parameter,
            dependency.y_axis,
            initial_point.y_axis,
            dependency.dependent_function_name,
            x_axis_name,
            y_axis_name,
        ),
    )
    add_vertical_line(
        fig,
        x_axis_coordinate=initial_point.x_axis,
        y_axis_coordinate=initial_point.y_axis,
        text=f"Initial: {initial_point.x_axis:0.2f}",
        annotation_position="top",
    )
    add_vertical_line(
        fig,
        x_axis_coordinate=optimized_point.x_axis,
        y_axis_coordinate=optimized_point.y_axis,
        text=f"Optimized: {optimized_point.x_axis:0.2f}",
        annotation_position="bottom",
    )
    add_optimization_limits(fig, bounds=explanation.opt_bounds)
    return fig


def _fill_state_with_grid(
    parameter_grid: tp.List[float],
    state: pd.DataFrame,
    optimizable_parameter: str,
) -> pd.DataFrame:
    state_with_grid = pd.concat(
        [state] * len(parameter_grid),  # noqa: WPS435 (we copy values explicitly)
        copy=True,
    )
    state_with_grid[optimizable_parameter] = parameter_grid
    return state_with_grid


def _extract_name(dependent_function: ReducesMatrixToSeries) -> str:
    if isinstance(dependent_function, BaseHandler):
        return str(dependent_function.name)
    return str(dependent_function)
