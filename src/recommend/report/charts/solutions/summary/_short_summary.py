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
from collections import Counter

import numpy as np
import plotly.graph_objects as go

from recommend.solution import Solutions

_PIE_ROTATION = 180

_PIE_HOLE_SIZE = 0.3

_FONT_COLOR = "#2b3f5c"
_ANNOTATION_MARGIN = 0.05
_DEFAULT_PLOT_TITLE_SIZE = 20

UPLIFT_NUMBER_DOMAIN = (0.525, 1)
PIE_DOMAIN = (0.025, 0.425)


def plot_summary(
    solutions: Solutions,
    objective_unit: tp.Optional[str],
) -> go.Figure:
    fig = go.Figure()
    _add_uplift_number(fig, solutions, objective_unit)
    _add_pie_with_overall_rows_optimized(fig, solutions)
    return fig


def _add_uplift_number(
    fig: go.Figure, solutions: Solutions, objective_unit: tp.Optional[str],
) -> None:
    _validate_all_solutions_have_same_sense(solutions)

    mean_objective = None
    mean_objective_with_uplift = None
    if solutions:
        mean_uplift = np.mean([solution.uplift for solution in solutions.values()])
        mean_objective = np.mean([
            solution.objective_before_optimization
            for solution in solutions.values()
        ])
        mean_objective_with_uplift = mean_objective + mean_uplift

    inverse_color_mapping = (
        {"increasing": {"color": "red"}, "decreasing": {"color": "green"}}
        if mean_objective_with_uplift is not None and mean_objective_with_uplift > 0
        else {}
    )

    suffix = f" {objective_unit}" if objective_unit is not None else None
    fig.add_indicator(
        mode="number+delta",
        value=mean_objective_with_uplift,
        delta={"reference": mean_objective, "suffix": suffix, **inverse_color_mapping},
        domain_x=UPLIFT_NUMBER_DOMAIN,
        number_suffix=suffix,
    )
    fig.add_annotation(
        text="<b>Mean Objective<br>After Optimization</b><br>",
        font=dict(size=_DEFAULT_PLOT_TITLE_SIZE, color=_FONT_COLOR),
        x=sum(UPLIFT_NUMBER_DOMAIN) / 2,
        xref="x domain",
        xanchor="center",
        y=1 - _ANNOTATION_MARGIN,
        yref="y domain",
        yanchor="top",
        showarrow=False,
        borderpad=0,
        borderwidth=0,
    )


def _validate_all_solutions_have_same_sense(solutions: Solutions) -> None:
    unique_senses = Counter(solution.problem.sense for solution in solutions.values())
    if len(unique_senses) > 1:
        raise ValueError(f"Expected same sense solutions found: {unique_senses}")


def _add_pie_with_overall_rows_optimized(fig: go.Figure, solutions: Solutions) -> None:
    total_recs = len(solutions)
    n_successful = sum(
        [solution.is_successful_optimization for solution in solutions.values()],
    )
    n_skipped = total_recs - n_successful
    optimized_share = n_successful / total_recs if n_successful else 0

    chart_tp_bottom_margin = 0.32
    fig.add_pie(
        values=[n_successful, n_skipped],
        labels=["<b>Successful</b>", "<b>Skipped</b>"],
        textinfo="label",
        textposition="outside",
        showlegend=False,
        hole=_PIE_HOLE_SIZE,
        insidetextorientation="horizontal",
        rotation=_PIE_ROTATION,
        marker=dict(
            colors=["#91C491", "grey"],
            line=dict(color="#000000", width=2),
        ),
        automargin=False,
        domain=dict(
            x=PIE_DOMAIN, y=[chart_tp_bottom_margin, 1 - chart_tp_bottom_margin],
        ),
    )

    fig.add_annotation(
        text=(
            "<b>Points optimized:</b><br>"
            f"<sup>{n_successful}/{total_recs} ({optimized_share:.0%})</sup>"
        ),
        font=dict(size=_DEFAULT_PLOT_TITLE_SIZE, color=_FONT_COLOR),
        x=sum(PIE_DOMAIN) / 2,
        xref="x domain",
        xanchor="center",
        y=1 - _ANNOTATION_MARGIN,
        yref="y domain",
        yanchor="top",
        showarrow=False,
        borderpad=0,
        borderwidth=0,
    )
