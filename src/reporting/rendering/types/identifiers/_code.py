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

""" Module for classes that provide ways to identify if some objects need to be rendered
 in a special way """

import typing as tp
from functools import cached_property


@tp.runtime_checkable
class Code(tp.Protocol):
    """Wraps string containing code"""
    code: str | None
    language: str | None

    @cached_property
    def formatted_code(self) -> str:
        """Returns formatted code"""
