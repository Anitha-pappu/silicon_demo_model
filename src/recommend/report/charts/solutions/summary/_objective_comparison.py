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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from reporting.types import TColumn

FEATURE_COMPARISON_LEGEND_OFFSET_X = 1.0
FEATURE_COMPARISON_LEGEND_OFFSET_Y = 1.15

TYPE = "type"
INITIAL = "initial"
OPTIMIZED = "optimized"
OBJECTIVE = "objective"
COLOR_MAP = {INITIAL: "grey", OPTIMIZED: "green"}  # noqa: WPS407

HIST_NORM = "probability density"


def plot_objective_comparison(
    df_before_after: pd.DataFrame,
    objective_before_column: TColumn = INITIAL,
    objective_after_column: TColumn = OPTIMIZED,
) -> go.Figure:
    """
    Creates before vs. after an objectives comparison figure

    Args:
        df_before_after: data frame with objectives before and after validation
        objective_before_column: column with objective before optimization
        objective_after_column: column with objective after optimization

    Returns:
        Objective comparison histogram figure
    """

    df_before_after = _unstack_data(
        df_before_after, objective_before_column, objective_after_column,
    )

    fig = px.histogram(
        df_before_after,
        x="objective",
        color=TYPE,
        color_discrete_map=COLOR_MAP,
        marginal="box",
        hover_data=df_before_after.columns,
        opacity=0.5,
        barmode="overlay",
        histnorm=HIST_NORM,
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
    )
    fig.update_yaxes(
        title="Probability Density<br>Histogram",
        hoverformat=".2f",
        row=1,
    )
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        hoverformat=".2f",
    )
    fig.update_layout(
        title="Objective Comparison<br><sup>Before and After Optimization</sup>",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            title_text=None,
            orientation="h",
            x=FEATURE_COMPARISON_LEGEND_OFFSET_X,
            y=FEATURE_COMPARISON_LEGEND_OFFSET_Y,
            xanchor="right",
            yanchor="top",
        ),
    )
    return fig


def _unstack_data(
    df_before_after: pd.DataFrame,
    objective_before_column: TColumn,
    objective_after_column: TColumn,
) -> pd.DataFrame:
    """
    Transforms flat form data::

        | ``objective_before_column`` | ``objective_after_column`` |
        |----------------------------:|---------------------------:|
        |                     14.42   |                    10.7975 |
        |                     14.9194 |                    10.8225 |
        |                     13.9857 |                    10.8283 |
        |                     11.3064 |                    10.7298 |
        |                     ...     |                    ...     |

    to long form data required by plotly express::

        | type      |   objective |
        |:----------|------------:|
        | initial   |     14.42   |
        | initial   |     14.9194 |
        | initial   |     13.9857 |
        | initial   |     11.3064 |
        | ...       |     ...     |
        | optimized |     10.7975 |
        | optimized |     10.8225 |
        | optimized |     10.8283 |
        | optimized |     10.7298 |
        | ...       |     ...     |

    Args:
        df_before_after:
        objective_after_column:
        objective_before_column:

    Returns: unstacked version of the ``df_df_before_after``
    """
    return (
        # select in order
        df_before_after[[objective_before_column, objective_after_column]]
        .set_axis([INITIAL, OPTIMIZED], axis=1)  # remove multi-indexing (if exists)
        # unstack & cast to frame with type and objective columns
        .unstack().reset_index(level=0).set_axis([TYPE, OBJECTIVE], axis=1)
    )
