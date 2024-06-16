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

from reporting.rendering.types import identifiers

from ._base import (  # noqa: WPS436
    ElementInfo,
    ElementType,
    PdfRenderableElement,
    PdfRenderableReportProtocol,
    TUid,
)


class PdfRenderableCode(PdfRenderableElement):

    def __init__(
        self,
        input_code: identifiers.Code,
        uid: tp.Optional[TUid] = None,
    ) -> None:
        super().__init__(uid=uid)
        self._code = _code_from_code_identifier(input_code)

    def to_reportlab(self, style=None):
        return Paragraph(text=self._code, style=style)

    @property
    def element_type(self) -> ElementType:
        return ElementType.CODE

    def __repr__(self) -> str:
        code_repr = repr(self._code)
        input_element_repr = f"input_code={code_repr}"
        return self._basic_repr(input_element_repr=input_element_repr)

    @property
    def _element_info_class(self) -> tp.Type[CodeElementInfo]:
        return CodeElementInfo

    def _get_kwargs_to_reportlab(
        self,
        renderable_report: tp.Optional[PdfRenderableReportProtocol],
    ) -> tp.Dict[str, tp.Any]:
        if renderable_report is None:
            return {}
        content_style = renderable_report.content_style
        code_style = content_style.code_style
        return dict(style=code_style)


def _code_from_code_identifier(code_identifier: identifiers.Code):
    """
    Notes:
        As of now

        - replaces ``\n`` with ``<br />`` in order to use the format from
        the code formatter
        - replaces four consecutive spaces with a space and 4 non-breakable spaces
            - this is to enable nicer indents in python code
        - it could be that this is a bit hacky and works with python code, but does
        the job for the moment
    """
    # TODO: Consider using the following trick only for python
    #  And/or consider making a function that can handle it for each of the
    #  supported code formats
    four_nbsp = "&nbsp;" * 4
    space_and_four_nbsp = f" {four_nbsp}"
    return (
        code_identifier.formatted_code
        .replace("\n", "<br />")
        .replace(" " * 4, space_and_four_nbsp)
    )


class CodeElementInfo(ElementInfo):
    """Store info about a code element that can be used when building the pdf"""

    def _store_info(self, element: PdfRenderableCode):
        """ Does not store any additional info"""
