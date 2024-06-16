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

""" Prereportlab classes

Used as interface between (supported) generic objects and reportlab.
"""
from ._base import ElementInfo, ElementType, PdfRenderableElement, TUid
from ._code import CodeElementInfo, PdfRenderableCode
from ._flowable import FlowableElementInfo, PdfRenderableFlowable
from ._header import HeaderElementInfo, PdfRenderableHeader
from ._image import ImageElementInfo, PdfRenderableImage
from ._table import PdfRenderableTable, TableElementInfo
from ._text import PdfRenderableText, TextElementInfo
from .into_renderable_converter import convert_recursively_into_pdf_renderable
