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
from textwrap import wrap

import plotly.graph_objects as go

_TEXT_BOX_MARGIN_ABS = 60
_INTERLINE_MARGIN_MULTIPLIER = 0.3
_DEFAULT_LEFT_MARGIN = 20

EOL_TAG = "<br>"


def plot_string(
    text: str,
    title: str = "",
    text_size: int = 14,
    title_size: int = 24,
    max_characters_per_text_line: tp.Optional[int] = 80,
    font_color: str = "#2b3f5c",
    left_margin: float = _DEFAULT_LEFT_MARGIN,
) -> go.Figure:
    """
    Generate a text plot from any data that can be converted to string

    Args:
        text: main text to show
        title: title; located above the text
        text_size:
        title_size:
        max_characters_per_text_line: used to wrap text and title
         to ``max_characters_per_text_line``; if ``None``, then no wrapping applied
        font_color: text and title font color
        left_margin: (default: ``_DEFAULT_LEFT_MARGIN``) text margin from the left
         border of the chart

    Returns: plot of plain string
    """

    title = _prep_title(title, text_size, title_size, max_characters_per_text_line)
    text = _prep_text(text, max_characters_per_text_line)
    height, center_y_location = _evaluate_plot_height(
        text, text_size, title, title_size,
    )
    fig = go.Figure(
        layout=go.Layout(
            xaxis={
                "fixedrange": True,
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
                "range": (0, 1),
            },
            yaxis={
                "fixedrange": True,
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
                "range": (0, 1),
            },
            paper_bgcolor="white",
            plot_bgcolor="rgba(0,0,0,0)",
            height=height,
            margin=dict(l=left_margin, r=0, t=0, b=0),
        ),
    )
    canvas_y_center = 0.5
    fig.add_annotation(
        x=0,
        y=(center_y_location if text else canvas_y_center),
        text=f"<b>{title}</b>",  # make title bold
        font=dict(size=title_size, color=font_color),
        showarrow=False,
        align="left",  # h-alignment
        valign="middle",  # v-alignment
        # box anchor point used for positioning
        xanchor="left",
        yanchor=("bottom" if text else "middle"),
    )
    fig.add_annotation(
        x=0,
        y=(center_y_location if title else canvas_y_center),
        text=text,
        font=dict(size=text_size, color=font_color),
        showarrow=False,
        align="left",  # h-alignment
        valign="middle",  # v-alignment
        # box anchor point used for positioning
        xanchor="left",
        yanchor=("top" if title else "middle"),
    )
    fig.update_annotations(borderpad=0)  # removes extra margin
    return fig


def _evaluate_plot_height(
    text: str, text_size: int, title: str, title_size: int,
) -> tp.Tuple[float, float]:
    n_lines_title = title.count(EOL_TAG) + 1 if title else 0
    n_lines_text = text.count(EOL_TAG) + 1 if text else 0
    interline_margin = _INTERLINE_MARGIN_MULTIPLIER * title_size
    total_height_title = n_lines_title * (title_size + interline_margin)
    total_height_text = n_lines_text * (text_size + interline_margin)
    total_height_whole_box = total_height_text + total_height_title
    plot_height = total_height_whole_box + 2 * _TEXT_BOX_MARGIN_ABS
    # center is relative
    center = (total_height_text + _TEXT_BOX_MARGIN_ABS) / plot_height
    return plot_height, center


def _prep_text(text: str, max_characters_per_text_line: tp.Optional[int]) -> str:
    """
    Replaces all pythonic EOL to HTML EOL;
    wraps text to `max_characters_per_text_line` if one is provided
    """
    lines = text.split("\n")  # Helps preserve the user's line breaks
    text_lines = []
    for line in lines:
        if max_characters_per_text_line is not None:
            text_lines.extend(wrap(line, width=max_characters_per_text_line))
        else:
            text_lines.append(line)
    return EOL_TAG.join(text_lines)


def _prep_title(
    title: str,
    text_size: int,
    title_size: int,
    max_characters_per_text_line: tp.Optional[int],
) -> str:
    """
    Preps title using `_prep_text` with scaling of `max_characters_per_text_line`.
    Scaling is defined as ratio of text to title.
    """
    if max_characters_per_text_line is None:
        title_width = None
    else:
        body_to_title_size_ratio = text_size / title_size
        title_width = int(max_characters_per_text_line * body_to_title_size_ratio)
    return _prep_text(title, title_width)
