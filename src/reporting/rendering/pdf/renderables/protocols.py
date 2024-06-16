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

import typing as tp

import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure as PltFigure
from reportlab.platypus import Flowable

from reporting.rendering.types import (
    SavefigCompatible,
    ToImageCompatible,
    identifiers,
)

SURE_IMAGES = (go.Figure, PltFigure)
SURE_CODE = (identifiers.Code,)
SURE_TABLE = (identifiers.Table,)
POTENTIAL_IMAGES = (ToImageCompatible, SavefigCompatible)
POTENTIAL_TEXT = (str, )
# Consider adding pd.Series,
#  after PdfRenderableTable will be able to change series into dataframes
POTENTIAL_TABLE = (pd.DataFrame, )
SURE_FLOWABLE = (Flowable, )

PDF_CONTENT_RENDERABLE_TYPES = (
    *SURE_IMAGES,
    *SURE_CODE,
    *SURE_TABLE,
    *POTENTIAL_IMAGES,
    *POTENTIAL_TEXT,
    *SURE_FLOWABLE,
)

TPdfRenderableContent = tp.Union[  # These are the `PDF_CONTENT_RENDERABLE_TYPES`
    go.Figure,
    PltFigure,
    identifiers.Code,
    identifiers.Table,
    ToImageCompatible,
    SavefigCompatible,
    str,
    pd.DataFrame,
    Flowable,
]
# TODO: Is the int dict key supported by a pdf report? Check!
# TODO: Remember to add support for a `SectionHeader` object in the html report as well,
#  after confirming that it applies there

# ``TReportStructureHeader`` and ``TPdfRenderableDictKey`` are the same, but they are
# seen from a different perspective:
#  - ``TReportStructureHeader`` is what is meant to be a report title
#  - ``TPdfRenderableDictKey`` is what the pdf renderer can handle as key in a dict
TReportStructureHeader = tp.Union[str, int, identifiers.SectionHeader]
TPdfRenderableDictKey = TReportStructureHeader

TPdfRenderableDictValue = tp.Union[  # These are actual content elements
    TPdfRenderableDictKey,  # TODO: Check why is correct that the dict key is here
    # Seems yes because it used to be defined like this in rendering
    TPdfRenderableContent,
    tp.List[TPdfRenderableContent],
]
TPdfRenderableDict = tp.Dict[
    TPdfRenderableDictKey,
    tp.Union[TPdfRenderableDictValue, "TPdfRenderableDict"],
]
