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

from reportlab.platypus import Flowable

from ._base import (  # noqa: WPS436
    ElementInfo,
    ElementType,
    PdfRenderableElement,
    PdfRenderableReportProtocol,
    TUid,
)


class PdfRenderableFlowable(PdfRenderableElement):
    """ Wrapper for ``reportlab.platypus.Flowables``

    This class exists only to have a uniform internal api to the stage just before
    the rendering into a pdf.
    It helps mostly by providing a uniform terminology to describe all the pre-pdf
    object (they can be collectively referred to as "pre-pdf" objects instead of having
    to call them "pre-pdf objects and Flowables").
    It is not supposed to do anything more than this.

    ``Flowables`` are just used as-is.
    """
    def __init__(self, flowable: Flowable, uid: tp.Optional[TUid] = None) -> None:
        super().__init__(uid=uid)
        self._flowable = flowable

    def to_reportlab(self):
        return self._flowable

    @property
    def element_type(self) -> ElementType:
        return ElementType.FLOWABLE

    def __repr__(self) -> str:
        flowable_repr = repr(self._flowable)
        input_element_repr = f"flowable={flowable_repr}"
        return self._basic_repr(input_element_repr=input_element_repr)

    @property
    def _element_info_class(self) -> tp.Type[FlowableElementInfo]:
        return FlowableElementInfo

    def _get_kwargs_to_reportlab(
        self,
        renderable_report: tp.Optional[PdfRenderableReportProtocol],
    ) -> tp.Dict[str, tp.Any]:
        return {}


class FlowableElementInfo(ElementInfo):
    """Store info about a flowable element that can be used when building the pdf"""

    def _store_info(self, element: PdfRenderableFlowable):
        """ Does not store any additional info"""
