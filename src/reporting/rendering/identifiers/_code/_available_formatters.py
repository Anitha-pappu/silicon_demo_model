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

import json
import re
import typing as tp

import autopep8
import black
import jsbeautifier


class CodeFormattingError(Exception):
    """ A class for code-formatting errors"""


def format_using_black(code: str, prettify_default_repr: bool = True) -> str:
    if prettify_default_repr:
        code = _replace_default_reprs_with_class_name(code)
    try:
        return black.format_str(code, mode=black.FileMode())  # type: ignore
    except black.InvalidInput as exc:
        raise CodeFormattingError from exc


def format_using_auto_pep8(code: str, prettify_default_repr: bool = True) -> str:
    if prettify_default_repr:
        code = _replace_default_reprs_with_class_name(code)
    return tp.cast(str, autopep8.fix_code(code))


def apply_no_formatting(code: str) -> str:
    return code


def format_json(code: str, indent: int = 2) -> str:
    try:
        json_object = json.loads(code)
    except json.JSONDecodeError as exc:
        raise CodeFormattingError from exc
    return json.dumps(json_object, indent=indent)


def format_js(code: str) -> str:
    return tp.cast(str, jsbeautifier.beautify(code))


def _replace_default_reprs_with_class_name(object_repr: str) -> str:
    """
    Replaces inputs of objects with default repr to pythonic object creation

    Examples::
        >>> _replace_default_reprs_with_class_name(
        ...     "<catboost.core.CatBoostRegressor object at 0x7f7cf1104b50>"
        ...  )
        "catboost.core.CatBoostRegressor(...)"

    """
    return re.sub("<([._a-zA-Z0-9]+) object at .*>", r"\1(...)", object_repr)
