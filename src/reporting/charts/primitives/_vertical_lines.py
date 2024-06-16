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

import plotly.graph_objects as go

_DEFAULT_VERTICAL_LINE_ANNOTATION_TEXT_ANGLE = -15


def add_vertical_lines(
    fig: go.Figure, vertical_line_locations: tp.Optional[tp.Dict[str, int]] = None,
) -> None:
    if vertical_line_locations is None:
        return

    for line_text, line_loc in vertical_line_locations.items():
        _add_vertical_line(fig, x=line_loc, text=line_text)


def _add_vertical_line(fig: go.Figure, x: float, text: str) -> None:  # noqa: WPS111
    yref = "paper"
    y_min, y_max = _find_limits_across_all_y_axes(fig)

    fig.add_shape(
        type="line",
        xref="x",
        yref=yref,
        x0=x,
        x1=x,
        y0=y_min,
        y1=y_max,
        line=dict(color="Black", width=1),
    )

    fig.add_annotation(
        go.layout.Annotation(
            showarrow=True,
            text=text,
            x=x,
            xref="x",
            xanchor="left",
            y=y_max,
            yanchor="top",
            yref=yref,
            textangle=_DEFAULT_VERTICAL_LINE_ANNOTATION_TEXT_ANGLE,
        ),
    )


def _find_limits_across_all_y_axes(fig: go.Figure) -> tp.Tuple[float, float]:
    y_axes_with_domain = list(
        fig.select_yaxes(lambda x: x.domain is not None),  # noqa: WPS111
    )
    y_min, _ = min(axis.domain for axis in y_axes_with_domain)
    _, y_max = max(axis.domain for axis in y_axes_with_domain)
    return y_min, y_max
