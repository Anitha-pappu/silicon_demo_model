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
    SUPPORTED_RENDERING_AGNOSTIC_TYPES as _RENDERING_AGNOSTIC_TYPES,
)
from reporting.rendering.types.identifiers import SectionHeader

# Ok to ignore S101 in this module since the functions in this module are used in
#  connection with testing


def check_is_valid_renderable_structure(
    object_to_check: tp.Any,
    acceptable_rendering_types: tuple[tp.Type[tp.Any]] = _RENDERING_AGNOSTIC_TYPES,
) -> None:
    """
    Raises an ``AssertionError`` if ``object_to_check`` contains unexpected types.
    What is being checked:
        - for ``dict``:
            1. each key is a non-empty string
            2. each value is a valid renderable (checked through recursive call of
               ``check_nested_dict_has_expected_types`` on each element)
        - for ``list``: each value are valid renderables
            (checked through recursive call of ``check_nested_dict_has_expected_types``
            on each element)
        - other types: an object must be valid renderable
    """
    if isinstance(object_to_check, dict):
        for header_section, dict_element in object_to_check.items():
            _validate_name(header_section)
            check_is_valid_renderable_structure(object_to_check=dict_element)
        return
    if isinstance(object_to_check, list):
        for list_element in object_to_check:
            check_is_valid_renderable_structure(object_to_check=list_element)
        return
    if isinstance(object_to_check, acceptable_rendering_types):
        return
    obj_type = type(object_to_check)
    raise TypeError(f"Incompatible type found {obj_type}")


def _validate_name(section_header: tp.Any) -> None:
    if isinstance(section_header, SectionHeader):
        return
    if isinstance(section_header, str) and section_header:
        return
    raise TypeError(
        "Every dict key must be an instance"
        " of ``SectionHeader`` or a non-empty string",
    )
