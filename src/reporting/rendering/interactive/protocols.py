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

from reporting.rendering.types import ReprHtmlCompatible

TInteractiveRenderableDictKey = tp.Union[str, int]
TInteractiveRenderableDictValue = tp.Union[
    TInteractiveRenderableDictKey, ReprHtmlCompatible, tp.List[ReprHtmlCompatible],
]
TInteractiveRenderableDict = tp.Dict[
    TInteractiveRenderableDictKey, tp.Union[
        TInteractiveRenderableDictValue, "TInteractiveRenderableDict",
    ],
]
TDictWithFloatsOrStr = tp.Dict[str, tp.Union[float, str]]

# As long as they have a _repr_html_, they are directly renderable
# This list is here just to present the python types that natively have a _repr_html_
# The list includes only the common types that are also `SUPPORTED_COMMON_TYPES`
DIRECTLY_INTERACTIVE_RENDERABLE_COMMON_TYPES = (
    pd.DataFrame,
    # TODO: Add pd.Series when it will be officially supported
)
DIRECTLY_INTERACTIVE_RENDERABLE_TYPES = (ReprHtmlCompatible,)
TInteractiveRenderableFlatDict = tp.Dict[
    TInteractiveRenderableDictKey, ReprHtmlCompatible,
]
