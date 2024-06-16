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

_TRenderableKey = tp.TypeVar("_TRenderableKey")
_TRenderableValue = tp.TypeVar("_TRenderableValue")
_TRenderableDict = tp.Dict[_TRenderableKey, _TRenderableValue]
_TRenderableElement = tp.TypeVar(
    "_TRenderableElement",
    _TRenderableDict,
    _TRenderableValue,
)


@tp.runtime_checkable
class SectionHeader(tp.Hashable, tp.Protocol):
    """Identifier that indicates the section header.
    Hashable to be used as a key in a dictionary
    """
    header_text: str
    description: str | None


def remove_section_description_from_structure(
    structure: _TRenderableElement,
) -> _TRenderableElement:
    """Remove descriptions from the report structure.

    Notes:
        This is done to avoid having the descriptions in the structure,
        since this functionality is not supported by multiple renderers
    """
    if isinstance(structure, dict):
        structure_without_descriptions = {}
        for section_header, section_content in structure.items():
            if isinstance(section_header, SectionHeader):
                section_header = section_header.header_text
            structure_without_descriptions[section_header] = (
                remove_section_description_from_structure(section_content)
            )
        return structure_without_descriptions
    if isinstance(structure, list):
        return [
            remove_section_description_from_structure(section_content)  # noqa: WPS441
            for section_content in structure
        ]
    return structure
