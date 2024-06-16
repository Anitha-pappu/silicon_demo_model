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


from dataclasses import dataclass

DEFAULT_TITLE = ""
DEFAULT_TEXT_SIZE = 14
DEFAULT_TITLE_SIZE = 24
DEFAULT_MAX_CHARS_PER_LINE = 80
DEFAULT_FONT_COLOR = "#2b3f5c"
DEFAULT_LEFT_MARGIN = 20


@dataclass(frozen=True)
class Text(object):
    """
    Rendering-agnostic text identifier

    Args:
        text: main text to show
        title: title; located above the text
        text_size: font size of the text
        title_size: font size of the title
            max_characters_per_text_line: used to wrap text and title
            to ``max_characters_per_text_line``; if ``None``, then no wrapping applied
        font_color: text and title font color
        left_margin: margin from the left border
    """

    text: str
    title: str = DEFAULT_TITLE
    text_size: int = DEFAULT_TEXT_SIZE
    title_size: int = DEFAULT_TITLE_SIZE
    max_characters_per_text_line: int | None = DEFAULT_MAX_CHARS_PER_LINE
    font_color: str = DEFAULT_FONT_COLOR
    left_margin: float = DEFAULT_LEFT_MARGIN
