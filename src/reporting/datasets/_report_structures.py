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

from __future__ import annotations

import copy

import plotly.graph_objects as go

from reporting.rendering.types import TRenderingAgnosticDict


def _create_happy_face(face_type: int) -> go.Figure:
    """
    Create a scatter plot resembling a happy face.

    Different variations can be selected by specifying the desired ``face_type``.

    Args:
        face_type: int that determines which variation of the figure is returned.
        Takes either ``1`` or ``2`` as a value

    Returns:
        A scatter plot as a ``go.Figure`` object
    """
    if face_type == 1:
        x_axis = [1, 3, 5, 7, 9, None, 2, 3, 4, 3, 2, None, 6, 7, 8, 7, 6]
        y_axis = [2, 1.2, 1, 1.2, 2, None, 4, 3, 4, 5, 4, None, 4, 3, 4, 5, 4]
    else:
        x_axis = [1, 3, 5, 7, 9, None, 2, 4, 4, 2, 2, None, 6, 8, 8, 6, 6]
        y_axis = [2, 1.2, 1, 1.2, 2, None, 3, 3, 4.5, 4.5, 3, None, 3, 3, 4.5, 4.5, 3]
    return go.Figure().add_trace(
        go.Scatter(
            x=x_axis,
            y=y_axis,
            mode="lines",
            line={'width': 4},
        ),
    ).update_xaxes(scaleanchor="y", scaleratio=1)


def create_basic_report_structure() -> TRenderingAgnosticDict:
    """

    Create a report structure by putting two figures in a
    dictionary with the keys becoming headers

    """

    figure1 = _create_happy_face(1)

    figure2 = _create_happy_face(2)

    return {
        "A happy face": figure1,
        "Another": figure2,
    }


def create_advanced_report_structure() -> TRenderingAgnosticDict:
    """

    Create a report with two figures in the same section by using a list to indicate
    that a section contains multiple objects.
    When the report is rendered the figures in the list are under the same section.

    """

    figure1 = _create_happy_face(1)

    figure2 = _create_happy_face(2)

    figure3 = copy.copy(figure2)
    figure3.data[0].line.color = "red"

    return {
        "A happy face": figure1,
        "Two happy faces": [figure2, figure3],
    }


def create_multilevel_report_structure() -> TRenderingAgnosticDict:
    """

    Create a report with sections and subsections with several levels of headers
    and sub headers.
    When the report is rendered, the sections are nested inside each other
    in the same way as the dictionaries are nested within the report structure.

    """

    face1 = _create_happy_face(1)

    face2 = _create_happy_face(2)

    figure1 = copy.copy(face1)
    figure1.data[0].line.color = "orange"

    figure2 = copy.copy(face1)
    figure2.data[0].line.color = "cyan"

    figure3 = copy.copy(face1)
    figure3.data[0].line.color = "navy"

    figure4 = copy.copy(face2)
    figure4.data[0].line.color = "navy"

    return {
        "An orange face": figure1,
        "Blue faces": {
            "Cyan blue": figure2,
            "Navy blue": [figure3, figure4],
        },
    }
