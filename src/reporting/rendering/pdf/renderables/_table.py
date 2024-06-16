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


class PdfRenderableTable(PdfRenderableElement):

    _TABLE_NOT_SUPPORTED_MESSAGE = "TABLE NOT SUPPORTED"

    def __init__(self, table, uid: tp.Optional[TUid] = None) -> None:
        super().__init__(uid=uid)
        self._table = None

    def to_reportlab(self):
        return Paragraph(text=self._TABLE_NOT_SUPPORTED_MESSAGE, style=None)

    @property
    def element_type(self) -> ElementType:
        return ElementType.TABLE

    def __repr__(self) -> str:
        table_repr = repr(self._table)
        input_element_repr = f"table={table_repr}"
        return self._basic_repr(input_element_repr=input_element_repr)

    @property
    def _element_info_class(self) -> tp.Type[TableElementInfo]:
        return TableElementInfo

    def _get_kwargs_to_reportlab(
        self,
        renderable_report: tp.Optional[PdfRenderableReportProtocol],
    ) -> tp.Dict[str, tp.Any]:
        return {}


class TableElementInfo(ElementInfo):
    """Store info about a flowable image that can be used when building the pdf"""

    def _store_info(self, element: PdfRenderableTable):
        """ Does not store any additional info"""
