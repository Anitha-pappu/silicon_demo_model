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

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from reporting.rendering.html.renderables.protocols import (
    THtmlRenderableDict,
    THtmlRenderableDictKey,
)
from reporting.rendering.types import identifiers

from .renderables import (
    INITIAL_LEVEL_HEADER,
    HtmlRenderableHeader,
    HtmlRenderableObject,
    HtmlRenderableTocElement,
    TNumericPrefix,
    convert_object_into_renderable,
)

_TFigReference = tp.Tuple[THtmlRenderableDictKey, ...]
TSectionDescription = tp.Dict[_TFigReference, tp.Union[str, "TSectionDescription"]]


@dataclass
class Rendering(object):
    rendering_content: tp.List[HtmlRenderableObject]
    table_of_content: tp.List[HtmlRenderableTocElement]


def render_report(
    figures: THtmlRenderableDict,
    sections_description: tp.Optional[TSectionDescription],
    max_table_of_content_depth: tp.Optional[int],
    max_level_of_header: tp.Optional[int],
) -> Rendering:
    """
    Renders plot

    Args:
        figures: mapping from header to section content;
            contains one or several plots or another section.
        sections_description: maps section into its description;
            key is a path to the section, value is a description. See further example.
        max_table_of_content_depth: max header level to show in the table of content;
            level indexing starts from _INITIAL_LEVEL_HEADER
        max_level_of_header: all headers after this level will be hidden;
            level indexing starts from _INITIAL_LEVEL_HEADER

    Example:
        Consider the report structure defined by following report data::
            report_data = {
                'Model Report': {'Validation Period': {'Train': ..., 'Test': ...,}}
            }

        So this example contains a section 'Model Report' on the first level
        with 'Validation Period' as its subsection.
        And on the final level we have two sections 'Train' and 'Test'.
        Let's assume we want to provide a description for each section.
        We can do so using the following structure::

            section_descriptions = {
                ('Model Report', ): '...',
                ('Model Report', 'Validation'): '...',
                ('Model Report', 'Validation', 'Train'): '...',
                ('Model Report', 'Validation', 'Test'): '...',
            }

    Returns: rendering that contains rendered objects and first level of TOC elements
    """

    rendered_section = _convert_section_into_renderables(
        section_data=figures,
        section_descriptions=sections_description or {},
        header_level=INITIAL_LEVEL_HEADER,
        section_prefix=(),
        section_numeric_prefix=(),
    )
    rendered_section = _hide_headers_based_on_visibility_level(
        rendered_section, max_level_of_header,
    )
    table_of_content = _extract_table_of_content(
        rendered_section, max_table_of_content_depth,
    )
    return Rendering(rendered_section, table_of_content)


def _convert_section_into_renderables(
    section_data: THtmlRenderableDict,
    section_descriptions: TSectionDescription,
    header_level: int,
    section_prefix: _TFigReference,
    section_numeric_prefix: TNumericPrefix,
) -> tp.List[HtmlRenderableObject]:
    """
    Renders section:
        * creates ``HtmlRenderableHeader`` from keys
        * if content is a nested section,
         calls ``_convert_section_into_renderables`` recursively,
            creates (list of) `HtmlRenderableFigure` otherwise
    """
    rendered_objects = []
    for header_index, (header, section_content) in enumerate(section_data.items()):
        current_prefix = (*section_prefix, header)
        current_numeric_prefix = (*section_numeric_prefix, header_index)
        _handle_section_header(
            rendered_objects=rendered_objects,
            header_element=header,
            header_level=header_level,
            current_prefix=current_prefix,
            current_numeric_prefix=current_numeric_prefix,
            section_descriptions=section_descriptions,
        )
        if isinstance(section_content, dict):
            rendered_objects.extend(
                _convert_section_into_renderables(
                    section_data=section_content,
                    section_descriptions=section_descriptions,
                    header_level=header_level + 1,
                    section_prefix=current_prefix,
                    section_numeric_prefix=current_numeric_prefix,
                ),
            )
        elif isinstance(section_content, list):
            rendered_objects.extend(
                (convert_object_into_renderable(fig) for fig in section_content),
            )
        else:
            rendered_objects.append(convert_object_into_renderable(section_content))
    return rendered_objects


def _hide_headers_based_on_visibility_level(
    rendered_objects: tp.List[HtmlRenderableObject],
    max_level_of_header: tp.Optional[int],
) -> tp.List[HtmlRenderableObject]:
    return [
        rendered_obj
        for rendered_obj in rendered_objects
        if rendered_obj.is_visible(max_level_of_header)
    ]


def _extract_table_of_content(
    rendered_objects: tp.List[HtmlRenderableObject],
    max_table_of_content_depth: tp.Optional[int],
) -> tp.List[HtmlRenderableTocElement]:
    """
    Selects only headers that are visible at given level of `max_table_of_content_depth`
    """
    flat_rendered_toc = [
        HtmlRenderableTocElement(rendered_object)
        for rendered_object in rendered_objects
        if isinstance(rendered_object, HtmlRenderableHeader)
    ]

    visible_toc_elements = [
        toc_element
        for toc_element in flat_rendered_toc
        if toc_element.is_visible(max_table_of_content_depth)
    ]

    if not visible_toc_elements:
        return []

    root = HtmlRenderableTocElement(HtmlRenderableHeader(-1, "", (), None))
    root.add_child(visible_toc_elements.pop(0))
    visit_stack = [root]
    for toc_element in visible_toc_elements:
        parent = visit_stack[-1]
        sibling = parent.children[-1]
        if toc_element.level > sibling.level:  # stepping down
            visit_stack.append(sibling)
            parent = sibling
        # if toc_element.level == sibling.level: then just adding to the same level
        elif toc_element.level < sibling.level:
            while toc_element.level < sibling.level:  # stepping up until found sibling
                parent = visit_stack.pop()
                sibling = parent.children[-1]
            # return parent after finding right insertion place
            visit_stack.append(parent)
        parent.add_child(toc_element)
    return root.children


def prune_to_level(
    toc: tp.List[HtmlRenderableTocElement], max_level: int,
) -> tp.List[HtmlRenderableTocElement]:
    if max_level is None:
        return toc
    return [
        HtmlRenderableTocElement(
            reference_header=toc_element.reference_header,
            children=prune_to_level(toc_element.children, max_level),
        )
        for toc_element in toc
        if toc_element.level <= max_level
    ]


# TODO: consider moving this to the renderables
#  Analogously to how the ``convert_object_into_renderable`` function is defined in
#  renderables.
def _handle_section_header(
    rendered_objects: tp.List,
    header_element,
    header_level: int,
    current_prefix: _TFigReference,
    current_numeric_prefix: TNumericPrefix,
    section_descriptions: TSectionDescription,
) -> None:
    """``current_prefix``, ``current_numeric_prefix`` and ``section_description`` are
    the legacy implementation.
    The new ``SectionHeader`` class should make these obsolte and make it simpler and
    more easily customizable to add descriptions to the code.

    Notice that if a description is provided within the section header, this will be
    used instead of whatever is provided in the "current description"
    """
    if isinstance(header_element, (str, int)):
        rendered_objects.append(
            HtmlRenderableHeader(
                level=header_level,
                text=header_element,
                unique_prefix=current_numeric_prefix,
                description=section_descriptions.get(current_prefix),
            ),
        )
        return
    if isinstance(header_element, identifiers.SectionHeader):
        description = _retrieve_description(
            section_header=header_element,
            section_descriptions=section_descriptions,
            current_prefix=current_prefix,
        )
        rendered_objects.append(
            HtmlRenderableHeader(
                level=header_level,
                text=header_element.header_text,
                unique_prefix=current_numeric_prefix,
                description=description,
            ),
        )
        return
    header_element_class = header_element.__class__.__name__
    raise NotImplementedError(
        f"Instances of {header_element_class} are not supported as section headers",
    )


def _retrieve_description(
    section_header: identifiers.SectionHeader,
    section_descriptions: TSectionDescription,
    current_prefix: _TFigReference,
) -> str:
    """Determines if and which description should be used between the one in a
    ``SectionHeader`` element (chosen if existing) or the one provided among the
    ``section_descriptions.``
    """
    if section_header.description is not None:
        return section_header.description
    return section_descriptions.get(current_prefix)
