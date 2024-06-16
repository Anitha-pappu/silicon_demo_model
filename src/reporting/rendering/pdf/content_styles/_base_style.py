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

import types
import typing as tp
from abc import ABC, abstractmethod

from reportlab.lib.styles import ParagraphStyle

from reporting.rendering.pdf.fonts import register_fonts

register_fonts()  # Consider moving to pdf report generation

NORMAL_TEXT_SIZE = 14
FOOTER_TEXT_SIZE = 8

# `leading`` set to 1.2 * fontSize following here
#  https://stackoverflow.com/questions/21753218/reportlab-overlapping-words
#  in order to avoid overlapping words
DEFAULT_LEADING_FACTOR = 1.2
DEFAULT_SPACER_FACTOR = 1.2

DEFAULT_SPACER_HEIGHT = DEFAULT_SPACER_FACTOR * NORMAL_TEXT_SIZE

DEFAULT_TITLES_STYLES = types.MappingProxyType({
    0: ParagraphStyle(
        name="title_0",
        fontSize=40,  # noqa: WPS432
        fontName="PTSans-Bold",
        spaceBefore=2 * DEFAULT_SPACER_HEIGHT,
        spaceAfter=2 * DEFAULT_SPACER_HEIGHT,
        leading=DEFAULT_LEADING_FACTOR * 40,  # noqa: WPS432
    ),
    1: ParagraphStyle(
        name="title_1",
        fontSize=22,  # noqa: WPS432
        fontName="PTSans-Regular",
        leading=DEFAULT_LEADING_FACTOR * 22,  # noqa: WPS432
        spaceBefore=DEFAULT_SPACER_HEIGHT,
        spaceAfter=DEFAULT_SPACER_HEIGHT,
    ),
    2: ParagraphStyle(
        name="title_3",
        fontSize=18,  # noqa: WPS432
        fontName="PTSans-Regular",
        leading=DEFAULT_LEADING_FACTOR * 18,  # noqa: WPS432
        spaceBefore=DEFAULT_SPACER_HEIGHT,
        spaceAfter=DEFAULT_SPACER_HEIGHT,
    ),
    3: ParagraphStyle(
        name="title_3",
        fontSize=16,  # noqa: WPS432
        fontName="PTSans-Regular",
        leading=DEFAULT_LEADING_FACTOR * 16,  # noqa: WPS432
        spaceBefore=DEFAULT_SPACER_HEIGHT,
        spaceAfter=DEFAULT_SPACER_HEIGHT,
    ),
})
DEFAULT_STYLE_TITLE = DEFAULT_TITLES_STYLES[3]

DEFAULT_STYLE_PARAGRAPH = ParagraphStyle(
    name='Normal',
    fontSize=NORMAL_TEXT_SIZE,
    leading=DEFAULT_LEADING_FACTOR * NORMAL_TEXT_SIZE,
)

DEFAULT_STYLE_FOOTER = ParagraphStyle(
    name='Footer',
    fontSize=FOOTER_TEXT_SIZE,
    leading=DEFAULT_LEADING_FACTOR * FOOTER_TEXT_SIZE,
)

DEFAULT_CODE_PARAGRAPH_STYLE = ParagraphStyle(
    name='Normal',
    fontSize=NORMAL_TEXT_SIZE,
    leading=DEFAULT_LEADING_FACTOR * NORMAL_TEXT_SIZE,
    fontName="PTMono-Regular",
    textColor="#000000",
    backColor="#f9f9f9",
    borderPadding=(5, 5, 5, 5),
    spaceBefore=DEFAULT_SPACER_HEIGHT,
    spaceAfter=DEFAULT_SPACER_HEIGHT,
)


class ContentStyleAbstract(ABC):
    """Abstract class defining the required attributes of a content style"""
    @property
    @abstractmethod
    def titles_styles(self) -> tp.Dict[int, ParagraphStyle]:
        """A Mapping of heading level to heading style

        Heading level is an integer greater or equal than zero.
        The more important the heading, the lower the heading level.
        """

    @property
    @abstractmethod
    def default_style_title(self) -> ParagraphStyle:
        """Default Heading style.

        To be used with headings with unspecified heading level
        """

    @property
    @abstractmethod
    def paragraph_style(self) -> ParagraphStyle:
        """Paragraph style"""

    @property
    @abstractmethod
    def footer_style(self) -> ParagraphStyle:
        """Footer style"""

    @property
    @abstractmethod
    def code_style(self) -> ParagraphStyle:
        """Code style

        In the original idea, code is intended to be a monospaced font, and the
        paragraph box could padding and have a different background.
        """

    @classmethod
    def header_style(cls: ContentStyleBase, header_level: int):
        """Returns the specified style for a heading of a given ``heading_level``"""
        style = cls.titles_styles.get(header_level, None)
        if style is None:
            return cls.default_style_title
        return style

    @classmethod
    def text_style(cls: ContentStyleBase, hierarchy_level):
        """ Chooses the style for text.

        Assuming if hierarchy is ``None``, then it is a paragraph, otherwise a header.
        Hierarchy is intended to be an integer that depends on the position of the
        text in the nested document structure.
        The idea is that the hierarchy of an element can be used to determine the
        importance of a text element, and use it to assign a heading level.
        """
        if hierarchy_level is None:
            return cls.paragraph_style
        return cls.header_style(header_level=hierarchy_level)


class ContentStyleBase(ContentStyleAbstract):
    """Base class setting defaults for content style classes"""

    titles_styles: tp.Dict[int, ParagraphStyle] = DEFAULT_TITLES_STYLES
    default_style_title: ParagraphStyle = DEFAULT_STYLE_TITLE
    paragraph_style: ParagraphStyle = DEFAULT_STYLE_PARAGRAPH
    footer_style: ParagraphStyle = DEFAULT_STYLE_FOOTER
    code_style: ParagraphStyle = DEFAULT_CODE_PARAGRAPH_STYLE
