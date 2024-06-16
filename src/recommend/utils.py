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
Recommend pipeline utils.
"""

import importlib
import logging
import typing as tp
from pprint import pformat as pprint_format

import pandas as pd
from pydantic import TypeAdapter
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

_TKwargs = tp.Dict[str, tp.Any]
_TExpectedType = tp.TypeVar("_TExpectedType")
_TSchema = TypedDict(
    "_TSchema", {"required": tp.List[str], "properties": tp.Iterable[str]},
)
_TRecords = tp.List[tp.Dict[str, tp.Any]]


class ForbiddenKwargs(ValueError):
    """Raised when forbidden keys are passed in kwargs"""


def load_obj(obj_path: str, default_obj_path: str = "") -> tp.Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path. In the case this is provided, `obj_path`
        must be a single name of the object being imported.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.

    Examples:
        Importing an object::

            >>> load_obj("sklearn.linear_model.Ridge")

        Importing using `default_obj_path`::

            >>> load_obj("Ridge", default_obj_path="sklearn.linear_model")
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    loaded_object = getattr(module_obj, obj_name, None)
    if loaded_object is None:
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`.",
        )
    return loaded_object


def validate_kwargs(
    kwargs: _TKwargs, forbidden_keys: tp.Iterable[str], source: str,
) -> _TKwargs:
    """
    Raises:
        ForbiddenKwargs:
            with ``source`` message if ``kwargs`` contain any ``forbidden_keys``

    Returns:
        Copy of valid kwargs
    """
    found_forbidden_keys = set(forbidden_keys).intersection(kwargs)
    if found_forbidden_keys:
        raise ForbiddenKwargs(
            f"Found forbidden keyword arguments in {source}: {found_forbidden_keys}. "
            f"The error appears since those kwargs are populated programmatically.",
        )
    return kwargs.copy()


def load_type(
    raw_object: tp.Any,
    expected_type: tp.Type[_TExpectedType],
) -> tp.Type[_TExpectedType]:
    parsed_type: tp.Type[_TExpectedType] = (
        raw_object
        if issubclass(raw_object, expected_type)
        else load_obj(raw_object)
    )
    if not issubclass(raw_object, expected_type):
        raise ValueError(
            f"Parsed type `{parsed_type}` "
            f"is not a subclass of expected type `{expected_type}`.",
        )
    return parsed_type


def validate_is_a_single_row_dataframe(row_to_optimize: tp.Any) -> None:
    """
    Raises:
        ValueError: if got other than pd.DataFrame type
        ValueError: if row_to_optimize shape != 1
    """
    if isinstance(row_to_optimize, pd.Series):
        raise ValueError("Expected pd.DataFrame type for row_to_optimize.")
    n_rows, _ = row_to_optimize.shape
    if n_rows != 1:
        raise ValueError(
            f"Expected single-row dataframe. Got {n_rows} rows.",
        )


def parse_records_from_dataframe_by_schema(
    df: pd.DataFrame,
    schema: tp.Dict[str, tp.Any],
    columns_to_fields_mapping: tp.Optional[tp.Mapping[str, tp.Any]],
) -> _TRecords:
    """
    Retrieves records from the data-frame where
    each record is a dict defining an entity based on the ``schema``

    Args:
        df: dataframe to parse schema from
        schema: dict with two keys 'required' (specifies required columns)
            and 'properties' (list of all properties: required and optional)
        columns_to_fields_mapping: is used to rename ``df`` columns
            into ``schema`` fields; must contain all fields of schema

    Returns: extracted list of dicts aligned with provided schema

    Examples:

        Dataframe contsins extra columns::
            >>> df = pd.DataFrame(
            ...     {'tag_type': ['control', 'state'], 'unit': ['cm', 'mm']},
            ... )
            >>> schema = {'required': ['tag_type'], 'properties': ['tag_type']}
            >>> parse_records_from_dataframe_by_schema(df, schema, None)
            [{'tag_type': 'control'}, {'tag_type': 'state'}]


        Schema contains some missing optional columns::
            >>> df = pd.DataFrame(
            ...     {'tag_type': ['control', 'state'], 'unit': ['cm', 'mm']},
            ... )
            >>> schema = {
            ...     'required': ['tag_type'],
            ...     'properties': ['tag_type', 'unit', 'area'],
            ... }
            >>> parse_records_from_dataframe_by_schema(df, schema, None)
            [{'tag_type': 'control', 'unit': 'cm'}, {'tag_type': 'state', 'unit': 'mm'}]


        Exatrction with renaming::
            >>> df = pd.DataFrame(
            ...     {'tag_type': ['control', 'state'], 'unit': ['cm', 'mm']},
            ... )
            >>> schema = {
            ...     'required': ['tag_type'],
            ...     'properties': ['tag_type', 'unit', 'area']
            ... }
            >>> rename = {
            ...     'TAG_TYPE': 'tag_type',
            ...     'tag_type': 'tag_type',
            ...     'unit': 'unit',
            ...     'area': 'area',
            ... }
            >>> parse_records_from_dataframe_by_schema(df, schema, rename)
            [{'tag_type': 'control', 'unit': 'cm'}, {'tag_type': 'state', 'unit': 'mm'}]


        Exatrction with all required columns in renaming::
            >>> df = pd.DataFrame(
            ...     {'TAG_TYPE': ['control', 'state'], 'unit': ['cm', 'mm']},
            ... )
            >>> schema = {
            ...     'required': ['tag_type'],
            ...     'properties': ['tag_type', 'unit', 'area'],
            ... }
            >>> rename = {'TAG_TYPE': 'tag_type'}
            >>> parse_records_from_dataframe_by_schema(df, schema, rename)
            [{'tag_type': 'control'}, {'tag_type': 'state'}]

    """
    if columns_to_fields_mapping:
        df = df[columns_to_fields_mapping.keys()].rename(
            columns=columns_to_fields_mapping,
        )

    schema_parsed = TypeAdapter(_TSchema).validate_python(schema)

    required_columns = schema_parsed['required']
    missing_columns = set(required_columns).difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing following required columns: {missing_columns}")
    optional_columns = (
        set(schema_parsed['properties'])
        .difference(required_columns)
        .intersection(df.columns)
    )
    columns_to_select = [*required_columns, *optional_columns]
    records = df.loc[:, columns_to_select].to_dict(orient="records")
    return tp.cast(_TRecords, records)


def pformat(object_to_format: tp.Any, indent: int = 0) -> str:
    """
    Applies ``pprint.pformat`` to ``object_to_format``.
    Compared to ``pprint.pformat``, this function correctly deals with
    indentations and new lines.

    Args:
        object_to_format: object to create pretty formatted repr for
        indent: number of spaces to indent for each level of nesting

    Returns:
        formatted object_to_format repr
    """
    object_repr = pprint_format(
        object_to_format, indent=indent, sort_dicts=False, compact=True,
    )
    # detect correct brackets
    if object_repr.startswith("{"):
        # false flake8 alert
        brackets = "{}"  # noqa: P103
    elif object_repr.startswith("("):
        brackets = "()"
    elif object_repr.startswith("["):
        brackets = "[]"
    else:
        # object doesn't have brackets to split across lines, just indent it
        return _align_indent(object_repr, indent)
    return _split_brackets_between_lines(
        object_repr,
        indent,
        tuple(brackets),  # type: ignore
    )


def _split_brackets_between_lines(
    object_repr: tp.Any,
    indent: int,
    brackets: tp.Tuple[str, str],
) -> str:
    open_bracket, close_bracket = brackets
    object_repr = object_repr.strip(open_bracket).strip(close_bracket).strip(" ")
    pre_dict_indent = " " * max(indent, 4)
    post_dict_indent = " " * max(indent - 4, 0)
    return (
        f"{open_bracket}\n"
        f"{pre_dict_indent}{object_repr},\n"
        f"{post_dict_indent}{close_bracket}"
    )


def _align_indent(object_repr: str, indent: int) -> str:
    str_indent = " " * indent
    return object_repr.replace("\n", f"\n{str_indent}")
