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


class PdfRenderableText(PdfRenderableElement):
    """ Takes care of text"""

    def __init__(self, text, uid: tp.Optional[TUid] = None) -> None:
        super().__init__(uid=uid)
        self._text = text

    def to_reportlab(self, style=None):
        return Paragraph(text=self._text, style=style)

    @property
    def element_type(self) -> ElementType:
        return ElementType.TEXT

    def __repr__(self) -> str:
        text_repr = repr(self._text)
        input_element_repr = f"text={text_repr}"
        return self._basic_repr(input_element_repr=input_element_repr)

    @property
    def _element_info_class(self) -> tp.Type[TextElementInfo]:
        return TextElementInfo

    def _get_kwargs_to_reportlab(
        self,
        renderable_report: tp.Optional[PdfRenderableReportProtocol],
    ) -> tp.Dict[str, tp.Any]:
        """ ``None`` is the hierarchy level for text"""
        content_style = renderable_report.content_style
        text_style = content_style.text_style(hierarchy_level=None)
        return dict(style=text_style)


class TextElementInfo(ElementInfo):
    """Store info about a flowable image that can be used when building the pdf"""

    def _store_info(self, element: PdfRenderableText):
        """ Does not store any additional info"""
