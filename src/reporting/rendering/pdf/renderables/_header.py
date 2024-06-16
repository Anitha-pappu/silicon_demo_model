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

import typing as tp

from reportlab.platypus import Paragraph

from ._base import (  # noqa: WPS436
    ElementInfo,
    ElementType,
    PdfRenderableElement,
    PdfRenderableReportProtocol,
    TUid,
)


class PdfRenderableHeader(PdfRenderableElement):
    """ Takes care of headers

        When creating the ``reportlab`` object, sets the property ``.keepWithNext`` to
        ``True``. This keeps the element together with the next element in the story, if
        possible.

        This is done so that page breaks do not happen in places that make the document
        look ugly, e.g. between title and subtitle, or between title and figure, etc..
        Using the property ``.keepWithNext`` was chosen over using ``KeepTogether``
        objects because the latter did not work satisfactorily while exploring the best
        way to keep objects together.

        From `this StackOverflow post <https://stackoverflow.com/questions/4797536/
        how-to-group-objects-in-reportlab-so-that-they
        -stay-together-across-new-pages#answer-4885328>`_


        The header text should be a string, but for compatibility perhaps other
        formats should be supported (e.g. int (float?) for compatibility with the
        batch plot)
        At the moment ``text`` is just converted into a string using ``str``
    """

    # TODO: Add validation/warning if text is not text
    #  And/or explicitly support other formats
    def __init__(
        self,
        header_text,
        header_level: tp.Optional[int] = None,
        uid: tp.Optional[TUid] = None,
    ) -> None:
        super().__init__(uid=uid)
        header_text = str(header_text)  # This could be implemented in a more stable way
        self._header_text = header_text
        self._header_level = header_level

    @property
    def header_level(self):
        return self._header_level

    def to_reportlab(self, style=None):
        """
        Also sets the .keepWithNext attribute to True so titles are not separated
        from either the first element of the section or from the next (sub)title.
        """
        reportlab_header = Paragraph(text=self._header_text, style=style)
        reportlab_header.keepWithNext = True
        return reportlab_header

    @property
    def element_type(self) -> ElementType:
        return ElementType.HEADER

    def __repr__(self) -> str:
        header_text_repr = repr(self._header_text)
        input_element_repr = f"header_text={header_text_repr}"
        return self._basic_repr(input_element_repr=input_element_repr)

    @property
    def _element_info_class(self) -> tp.Type[HeaderElementInfo]:
        return HeaderElementInfo

    def _get_kwargs_to_reportlab(
        self,
        renderable_report: PdfRenderableReportProtocol,
    ) -> tp.Dict[str, tp.Any]:
        content_style = renderable_report.content_style
        header_level = self.header_level
        header_style = content_style.header_style(header_level=header_level)
        return dict(style=header_style)


class HeaderElementInfo(ElementInfo):
    """Store info about a header element that can be used when building the pdf"""

    def _store_info(self, element: PdfRenderableHeader):
        self.header_level = element.header_level
