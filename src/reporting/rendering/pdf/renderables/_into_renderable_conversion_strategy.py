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

"""
This module defines the conversion strategy used.

This conversion strategy is the one used in ``into_renderable_converter.py`` to
determine what pdf-renderable class is used when the objects are encountered in a
report structure.
It was defined there, but too many imports from ``.protocols`` were needed, so this
definition was moved to this dedicated file.
"""
from ._code import PdfRenderableCode  # noqa: WPS436
from ._flowable import PdfRenderableFlowable  # noqa: WPS436
from ._image import PdfRenderableImage  # noqa: WPS436
from ._table import PdfRenderableTable  # noqa: WPS436
from ._text import PdfRenderableText  # noqa: WPS436
from .protocols import (
    POTENTIAL_IMAGES,
    POTENTIAL_TABLE,
    POTENTIAL_TEXT,
    SURE_CODE,
    SURE_FLOWABLE,
    SURE_IMAGES,
    SURE_TABLE,
)

# todo: switch to function that does mapping
APPROPRIATE_CONVERSION_CLASS_SEARCH_STRATEGY = (
    (PdfRenderableImage, SURE_IMAGES),
    (PdfRenderableCode, SURE_CODE),
    (PdfRenderableTable, SURE_TABLE),
    (PdfRenderableImage, POTENTIAL_IMAGES),
    (PdfRenderableText, POTENTIAL_TEXT),
    (PdfRenderableTable, POTENTIAL_TABLE),
    (PdfRenderableFlowable, SURE_FLOWABLE),
)
