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

"""Contains functionality for formatting code."""

import logging
import typing as tp
from functools import partial

from ._available_formatters import (  # noqa: WPS436
    CodeFormattingError,
    apply_no_formatting,
    format_js,
    format_json,
    format_using_auto_pep8,
    format_using_black,
)


class _TCallableFormatter(tp.Protocol):
    def __call__(self, code: str, *args: tp.Any, **kwargs: tp.Any) -> str:
        """
        Formats input code

        Raises:
            ``CodeFormattingError`` - if formatting raised any exception
        """


TCodeFormatter = str | _TCallableFormatter | None


AVAILABLE_FORMATTERS: tp.Dict[str | None, _TCallableFormatter] = {  # noqa: WPS407
    "py-autopep8": partial(format_using_auto_pep8, prettify_default_repr=True),
    "py-black": partial(format_using_black, prettify_default_repr=True),
    "json": partial(format_json, indent=2),
    "js-jsbeautifier": partial(format_js),
    None: apply_no_formatting,
}

logger = logging.getLogger(__name__)


def format_code(code: str, code_formatter: TCodeFormatter = None) -> str:
    """
    Format ``code`` using the provided ``code_formatter``

    Args:
        code: code to format
        code_formatter: code formatter used to prettify code.
            It has to be either callable that converts str to formatted str
            or one of the built-in formatters from ``AVAILABLE_FORMATTERS``:
            {'py-autopep8', 'py-black', 'json', 'js-jsbeautifier'}.
            For user defined formatters note
            that in case of unsuccessful formatting,
            it must raise `CodeFormattingError` to allow catching those exceptions.
    """
    code_formatter = _parse_formatter(code_formatter)
    try:
        return code_formatter(code)
    except CodeFormattingError as exc:
        logger.warning(
            f"Code formatting failed, initial code will be used.\n"
            f"Code formatter: `{code_formatter}`.\n"
            f"Code: `{code}`.\n"
            f"Error message: `{exc}`.",
        )
    return code


def _parse_formatter(code_formatter: TCodeFormatter) -> _TCallableFormatter:
    if callable(code_formatter):
        return code_formatter
    elif code_formatter not in AVAILABLE_FORMATTERS:
        raise KeyError(
            f"Unknown formatter alias passed, "
            f"please consider one of known: {AVAILABLE_FORMATTERS}",
        )
    return AVAILABLE_FORMATTERS[code_formatter]
