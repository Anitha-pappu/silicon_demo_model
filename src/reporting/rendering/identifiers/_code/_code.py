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

from dataclasses import dataclass
from functools import cached_property

from ._formatter import TCodeFormatter, format_code  # noqa: WPS436


@dataclass(frozen=True)
class Code(object):
    """
    Code identifier

    Attributes:
        code: code text
        code_formatter: code formatter used to prettify code.
            It Has to be either callable that converts str to formatted str
            or one of the built-in formatters:
            {'py-autopep8', 'py-black', 'json', 'js-jsbeautifier'}.
            For user defined formatters note
            that in case of unsuccessful formatting,
            it must raise ``CodeFormattingError`` to allow catching those exceptions.
        language: this will be used in representation to highlight the code.
            Use this for setting proper code highlighting in html repr in case
            auto recognized language is incorrect.
            Most common args are {'python', 'sql', 'json', 'js', 'java', 'c++'}.
            See ``https://github.com/EnlighterJS/EnlighterJS#languages``
            for the full list of available languages.
    """

    code: str
    code_formatter: TCodeFormatter = None
    language: str | None = None

    @cached_property
    def formatted_code(self) -> str:
        return format_code(code=self.code, code_formatter=self.code_formatter)

    # todo: use the same _repr_html_ as the code
    def _repr_html_(self) -> str:
        """Returns HTML representation for code block"""
        return self.formatted_code
