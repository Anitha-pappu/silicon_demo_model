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
from plotly import graph_objects as go
from plotly.express.colors import qualitative as colors
from plotly.graph_objs import Figure as PlotlyFigure

from optimizer import Repair, UserDefinedRepair
from optimizer.types import Vector

from .extraction import _RepairViolation  # noqa: WPS450
from .layout import calculate_y_range_limits

REPAIR_OPACITY = 0.2
SCATTER_MARKER_SIZE_INITIAL_POSITION = 10
SCATTER_MARKER_SIZE_OPTIMIZED_POSITION = 10
VERTICAL_LINE_ANNOTATION_TEXT_ANGLE = 0
VERTICAL_LINE_FONT_SIZE = 12
VERTICAL_LINE_OPACITY = 0.75
ANNOTATION_COLOR = "grey"
MAIN_COLOR = "rgb(43, 75, 110)"


_TViolations = tp.List[tp.Iterable[bool]]


def create_line_trace(
    parameter_grid: tp.Iterable[float],
    dependent_function_values: tp.Iterable[float],
) -> go.Scatter:
    """
    Create line+markers scatter plot of the
    ``dependent_function(optimizable_parameter)`` vs ``optimizable_parameter``
    """
    return go.Scatter(
        x=parameter_grid,
        y=dependent_function_values,
        mode="lines+markers",
        showlegend=False,
        marker={
            "color": MAIN_COLOR,
            "size": 5,
            "line": {
                "color": MAIN_COLOR,
                "width": 1,
            },
        },
        hovertemplate="(%{x}, %{y})<extra></extra>",
    )


def create_initial_position_scatter_trace(
    initial_position: float,
    initial_dependent_function_value: float,
) -> go.Scatter:
    """
    Create single points scatter trace representing
    the initial state of the ``optimizable_parameter``.
    """
    return go.Scatter(
        x=[initial_position],
        y=[initial_dependent_function_value],
        fillcolor=MAIN_COLOR,
        mode="markers",
        marker={"size": SCATTER_MARKER_SIZE_OPTIMIZED_POSITION, "color": MAIN_COLOR},
        showlegend=False,
        hovertemplate="(%{x}, %{y})<extra></extra>",
    )


def create_optimized_position_scatter_trace(
    position: float,
    dependent_function_value: float,
) -> go.Scatter:
    """
    Create single points scatter trace representing
    the optimized state of the ``optimizable_parameter``.
    """
    return go.Scatter(
        x=[position],
        y=[dependent_function_value],
        fillcolor=MAIN_COLOR,
        mode="markers",
        marker={"size": SCATTER_MARKER_SIZE_OPTIMIZED_POSITION, "color": MAIN_COLOR},
        showlegend=False,
        hovertemplate="(%{x}, %{y})<extra></extra>",
    )


def _create_repair_violations_to_plot(
    repairs: tp.List[Repair],
    state_with_grid: pd.DataFrame,
) -> tp.Tuple[tp.List[Repair], _TViolations]:
    """
    Return user defined repairs and points of the grid where they are violated.
    """
    user_defined_repairs = [
        repair
        for repair in repairs
        if isinstance(repair, UserDefinedRepair)
    ]
    repair_violations = []
    for repair in user_defined_repairs:
        repair(state_with_grid)
        repair_violations.append(repair.constraint.violated)
    return user_defined_repairs, repair_violations


def create_repairs_scatter_traces_from_violations(
    parameter_grid: tp.List[float],
    repair_violations: tp.Iterable[_RepairViolation],
    initial_dependent_function_value: float,
    dependent_function_values: Vector,
) -> tp.List[go.Scatter]:
    """
    Create scatter traces that are filled inside
    that highlight regions constrained with ``UserDefinedRepairs``.

    Notes:
        All regions related to one repair have
        same color and item in legend of the plot.
    """
    scatters = []
    for repair_violation, color in zip(repair_violations, colors.Set1):
        # All regions related to one repair have same color and item in legend
        # This is done through manipulating ``showlegend`` kw argument.
        # See "Grouped Legend Items" section at plotly.com/python/legend/
        # for more details.
        showlegend = True
        for grid_index in range(len(parameter_grid)):  # noqa: WPS518
            if not repair_violation.violations[grid_index]:
                continue
            displayed_name = repair_violation.name.replace("_user_defined_repair", "")
            scatters.append(
                _create_single_repair_scatter_trace(
                    dependent_function_values,
                    initial_dependent_function_value,
                    grid_index=grid_index,
                    parameter_grid=parameter_grid,
                    color=color,
                    scatter_group_name=displayed_name,
                    showlegend=showlegend,
                ),
            )
            showlegend = False
    return scatters


def _create_single_repair_scatter_trace(
    dependent_function_values: Vector,
    initial_dependent_function_value: float,
    grid_index: int,
    parameter_grid: tp.List[float],
    color: str,
    scatter_group_name: str,
    showlegend: bool,
) -> go.Scatter:
    if grid_index == 0:
        lhs_coordinate = parameter_grid[0]
    else:
        lhs_coordinate = (
            parameter_grid[grid_index]
            - (parameter_grid[grid_index] - parameter_grid[grid_index - 1]) / 2
        )
    if grid_index == len(parameter_grid) - 1:
        rhs_coordinate = parameter_grid[-1]
    else:
        rhs_coordinate = (
            parameter_grid[grid_index]
            + (parameter_grid[grid_index + 1] - parameter_grid[grid_index]) / 2
        )
    lower_coordinate, upper_coordinate = calculate_y_range_limits(
        list(dependent_function_values) + [initial_dependent_function_value],
    )
    return go.Scatter(
        x=[
            lhs_coordinate,
            lhs_coordinate,
            rhs_coordinate,
            rhs_coordinate,
        ],
        y=[
            lower_coordinate,
            upper_coordinate,
            upper_coordinate,
            lower_coordinate,
        ],
        mode="lines",
        line={"color": "rgba(0,0,0,0)"},
        fill="toself",
        legendgroup=scatter_group_name,
        name=scatter_group_name,
        fillcolor=color,
        showlegend=showlegend,
        opacity=REPAIR_OPACITY,
        hoverinfo='skip',
    )


def add_vertical_line(
    fig: go.Figure,
    x_axis_coordinate: float,
    y_axis_coordinate: float,
    annotation_position: str,
    text: str,
) -> None:
    _, y_max = tuple(fig.layout.yaxis.range)
    fig.add_shape(
        type="line",
        x0=x_axis_coordinate,
        x1=x_axis_coordinate,
        y0=y_axis_coordinate,
        y1=y_max,
        line={"color": ANNOTATION_COLOR, "width": 1},
        opacity=VERTICAL_LINE_OPACITY,
    )
    fig.add_annotation(
        showarrow=True,
        text=text,
        x=x_axis_coordinate,
        xanchor="left",
        y=1,
        yref="paper",
        yanchor=annotation_position,
        textangle=VERTICAL_LINE_ANNOTATION_TEXT_ANGLE,
        font={"size": VERTICAL_LINE_FONT_SIZE, "color": ANNOTATION_COLOR},
        arrowcolor=ANNOTATION_COLOR,
        opacity=VERTICAL_LINE_OPACITY,
    )


def add_optimization_limits(fig: PlotlyFigure, bounds: tp.Tuple[float, float]) -> None:
    left_bound, right_bound = bounds
    fig.add_vline(x=left_bound, line_dash="dash", line_color="red")
    fig.add_vline(x=right_bound, line_dash="dash", line_color="red")
