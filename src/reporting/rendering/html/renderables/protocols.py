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

from reporting.rendering.types import (
    SavefigCompatible,
    ToHtmlCompatible,
    identifiers,
)

DIRECTLY_HTML_RENDERABLE_TYPES = (ToHtmlCompatible, SavefigCompatible)
THtmlRenderableContent = tp.Union[ToHtmlCompatible, SavefigCompatible]
THtmlRenderableDictKey = tp.Union[str, int, identifiers.SectionHeader]
THtmlRenderableDictValue = tp.Union[  # These are actual content elements
    THtmlRenderableDictKey,
    THtmlRenderableContent,
    tp.List[THtmlRenderableContent],
]
THtmlRenderableDict = tp.Dict[
    THtmlRenderableDictKey,
    tp.Union[THtmlRenderableDictValue, "THtmlRenderableDict"],
]
