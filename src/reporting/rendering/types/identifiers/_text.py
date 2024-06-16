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

_TCallableFormatter = tp.Callable[[str], str]


@tp.runtime_checkable
class Text(tp.Protocol):
    text: str
    title: str
    text_size: int
    title_size: int
    max_characters_per_text_line: int | None
    font_color: str
    left_margin: float
