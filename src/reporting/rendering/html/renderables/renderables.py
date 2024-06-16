
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

import base64
import io
import re
import typing as tp
from abc import ABC, abstractmethod
from functools import cached_property

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from reporting.charts.primitives import plot_string
from reporting.rendering.html.renderables.protocols import (
    THtmlRenderableContent,
    THtmlRenderableDictKey,
)
from reporting.rendering.types import (
    SavefigCompatible,
    ToHtmlCompatible,
    identifiers,
)

from ._code import plot_code  # noqa: WPS436
from ._table import plot_table  # noqa: WPS436

TNumericPrefix = tp.Tuple[int, ...]


_CSS_SECTION = "section"
_CSS_SECTION_CONTENT = "section-content"
_CSS_SECTION_TITLE = "section-title"
_CSS_FIGURE_IMG = "figure-img"
_CSS_FIGURE_HTML = "figure-html"


INITIAL_LEVEL_HEADER = 1

_DEFAULT_IMAGE_PADDING_INCH = 0.7


# TODO: Update docstring and error message
def convert_object_into_renderable(fig: tp.Any) -> HtmlRenderableFigure:
    """
    Renders a figure based on its type:
        * plotly.Figure -> HtmlRenderableToHtmlCompatible
        * matplotlib.Figure -> RenderableSavefigCompatible
    """
    fig = _map_into_compatible(fig)

    if isinstance(fig, ToHtmlCompatible):
        return HtmlRenderableToHtmlCompatible(fig)
    elif isinstance(fig, SavefigCompatible):
        return HtmlRenderableSavefigCompatible(fig)
    fig_type = type(fig)
    raise NotImplementedError(
        f"Current implementation supports only "
        f"`SavefigCompatible` and `ToHtmlCompatible` protocols. "
        f"Passed figure type `{fig_type}` does not support any of those.",
    )


def _map_into_compatible(fig: tp.Any) -> tp.Any:
    if isinstance(fig, pd.DataFrame):
        fig = plot_table(data=fig)
    if isinstance(fig, identifiers.Text):
        fig = plot_string(
            text=fig.text,
            title=fig.title,
            text_size=fig.text_size,
            title_size=fig.title_size,
            max_characters_per_text_line=fig.max_characters_per_text_line,
            font_color=fig.font_color,
            left_margin=fig.left_margin,
        )
    if isinstance(fig, str):
        fig = plot_string(text=fig)
    if isinstance(fig, identifiers.Code):
        fig = plot_code(code=fig.formatted_code, language=fig.language)
    if isinstance(fig, identifiers.Table):
        fig = plot_table(
            data=fig.table,
            columns=fig.columns,
            precision=fig.precision,
            title=fig.title,
            columns_filters_position=fig.columns_filters_position,
            columns_to_color_as_bars=fig.columns_to_color_as_bars,
            width=fig.width,
            table_alignment=fig.table_alignment,
            sort_by=fig.sort_by,
            show_index=fig.show_index,
        )
    return fig


class HtmlRenderableObject(ABC):
    @abstractmethod
    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        """ Is visible method"""

    @abstractmethod
    def to_html(
        self,
        previous_item: tp.Optional[HtmlRenderableObject] = None,
        next_item: tp.Optional[HtmlRenderableObject] = None,
    ) -> str:
        """ to_html method"""


class HtmlRenderableFigure(HtmlRenderableObject, ABC):
    def __init__(self, figure: THtmlRenderableContent) -> None:
        self._figure = figure

    @property
    def figure(self):
        return self._figure

    def __eq__(self, other: tp.Any) -> bool:
        """Helps to compare rendered objects; especially during tests"""
        if isinstance(other, HtmlRenderableFigure):
            return self._figure == other.figure
        return False

    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        return True

    def to_html(
        self,
        previous_item: tp.Optional[HtmlRenderableObject] = None,
        next_item: tp.Optional[HtmlRenderableObject] = None,
    ) -> str:
        encoded_fig = self._encode_to_str(self._figure)
        figure_html = self._wrap_src_to_html(encoded_fig)
        prefix = (
            "" if isinstance(previous_item, HtmlRenderableFigure)
            else
            f'<div class="{_CSS_SECTION_CONTENT}">'
        )
        suffix = "" if isinstance(next_item, HtmlRenderableFigure) else "</div></div>"
        return f"{prefix}{figure_html}{suffix}"

    @staticmethod
    @abstractmethod
    def _encode_to_str(figure: THtmlRenderableContent) -> str:
        """ encode_to_str method"""

    @staticmethod
    @abstractmethod
    def _wrap_src_to_html(encoded_src: str) -> str:
        """ wrap_src_to_html method"""


class HtmlRenderableToHtmlCompatible(HtmlRenderableFigure):
    @staticmethod
    def _encode_to_str(figure: ToHtmlCompatible) -> str:
        if isinstance(figure, go.Figure):
            html = figure.to_html(include_plotlyjs=False, full_html=False)
            return re.sub("<div>(.*)</div>", r"\1", html)  # removing wrapping div
        return figure.to_html()

    @staticmethod
    def _wrap_src_to_html(encoded_src: str) -> str:
        return f'<div class="{_CSS_FIGURE_HTML}">{encoded_src}</div>'


class HtmlRenderableSavefigCompatible(HtmlRenderableFigure):
    @staticmethod
    def _encode_to_str(figure: SavefigCompatible) -> str:
        buf = io.BytesIO()
        if isinstance(figure, plt.Figure):
            figure.savefig(
                buf,
                format="png",
                pad_inches=_DEFAULT_IMAGE_PADDING_INCH,
                bbox_inches='tight',
            )
        else:
            figure.savefig(buf, format="png")
        fig_bytes = buf.getvalue()
        return base64.b64encode(fig_bytes).decode()

    @staticmethod
    def _wrap_src_to_html(encoded_src: str) -> str:
        return (
            f'<img class="{_CSS_FIGURE_IMG}" alt="graph" '
            f'src="data:image/png;base64,{encoded_src}">'
        )


class HtmlRenderableHeader(HtmlRenderableObject):
    # highest header level starts from 2 since h1 is reserved for report's title
    _INITIAL_H_TAG_LEVEL = 2

    def __init__(
        self,
        level: int,
        text: THtmlRenderableDictKey,
        unique_prefix: TNumericPrefix,
        description: tp.Optional[str],
    ) -> None:
        self._level = level
        self._text = str(text)
        self._description = description
        self._unique_prefix = "-".join(
            # Ok ignoring WPS111 for simple list/tuple/dict/set comprehension
            str(x) for x in unique_prefix  # noqa: WPS111
        )

    def __eq__(self, other: tp.Any) -> bool:
        """Helps to compare rendered objects; especially during tests"""
        if isinstance(other, HtmlRenderableHeader):
            return self.id == other.id
        return False

    def to_html(
        self,
        previous_item: tp.Optional[HtmlRenderableObject] = None,
        next_item: tp.Optional[HtmlRenderableObject] = None,
    ) -> str:
        h_tag = f"h{self._h_tag_level}"
        default = (
            f'<{h_tag} class="{_CSS_SECTION_TITLE}" id="header-{self.id}">'
            f"{self.text}"
            f"</{h_tag}>"
        )
        prefix = (
            "" if isinstance(next_item, HtmlRenderableHeader)
            else f'<div class="{_CSS_SECTION}">'
        )
        if self.description is None:
            return f"{prefix}{default}"

        info_section = (
            f'\n<span data-text="{self.description}" class="tooltip">'
            f'<span class="info-icon"></span></span>'
        )
        return f'{prefix}\n<div class="inline">{default}{info_section}</div>'

    @property
    def level(self) -> int:
        return self._level

    @property
    def text(self) -> str:
        return self._text

    @property
    def description(self) -> tp.Optional[str]:
        return self._description

    @cached_property
    def id(self) -> str:
        normalized_text = re.sub('[^0-9a-zA-Z]+', '-', self.text)
        return f"{self._unique_prefix}-{self.level}-{normalized_text}"

    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        return self._is_visible(self.level, visibility_level)

    @property
    def _h_tag_level(self) -> int:
        """
        Evaluates level for <h> as diff between initial header level
        and initial <h> tag level
        """
        level_diff = self._INITIAL_H_TAG_LEVEL - INITIAL_LEVEL_HEADER
        return self.level + level_diff

    @staticmethod
    def _is_visible(object_level: int, visibility_level: tp.Optional[int]) -> bool:
        if visibility_level is None:
            return True
        return object_level <= visibility_level


class HtmlRenderableTocElement(HtmlRenderableObject):
    def __init__(
        self,
        reference_header: HtmlRenderableHeader,
        children: tp.Iterable[HtmlRenderableTocElement] = (),
    ) -> None:
        self._reference_header = reference_header
        self._children: tp.List[HtmlRenderableTocElement] = list(children)

    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, HtmlRenderableTocElement):
            return False

        if self.reference_header_id != other.reference_header_id:
            return False

        if self.level != other.level:
            return False

        return self.children == other.children

    @property
    def text(self) -> str:
        return self._reference_header.text

    @property
    def reference_header(self) -> HtmlRenderableHeader:
        return self._reference_header

    @property
    def reference_header_id(self) -> str:
        return self._reference_header.id

    @property
    def level(self) -> int:
        return self._reference_header.level

    def is_visible(self, visibility_level: tp.Optional[int]) -> bool:
        if visibility_level is None:
            return True
        return (
            self.level <= visibility_level
            and self._reference_header.is_visible(visibility_level)
        )

    @property
    def children(self) -> tp.List[HtmlRenderableTocElement]:
        return self._children

    def add_child(self, toc_element: HtmlRenderableTocElement) -> None:
        self._children.append(toc_element)

    # todo
    def to_html(
        self,
        previous_item: tp.Optional[HtmlRenderableObject] = None,
        next_item: tp.Optional[HtmlRenderableObject] = None,
    ) -> str:
        """TODO: To be implemented"""
