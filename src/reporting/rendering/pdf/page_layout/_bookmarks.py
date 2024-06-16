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
This function is meant to be used as ``afterFlowable``.

The functions to be used as afterflowable should be

``fn(doc: SimpleDocTemplate, flowable: Flowable, **kwargs)``
The stylable functions to be passed to the renderable report should be

``fn(doc: SimpleDocTemplate, flowable: Flowable, content_style= page_layout=...)``
In case content style or page layout are not needed, they should support **kwargs,
so in case content_style and/or page_layout are passed, they do not raise errors.
``fn(doc: SimpleDocTemplate, flowable: Flowable, **kwargs)``
"""

import typing as tp

from reportlab.platypus import SimpleDocTemplate

from reporting.rendering.pdf.page_layout.page_layout import PageLayout
from reporting.rendering.pdf.protocols import EnhancedFlowable
from reporting.rendering.pdf.renderables import ElementType, HeaderElementInfo

TBookmarkHeaderLevels = tp.Tuple[int]


def create_header_bookmark(
    doc: SimpleDocTemplate,
    flowable,
    page_layout: PageLayout,
    **kwargs,
):
    """Create pdf bookmarks.

    Very helpful when navigating the content.
    """
    return _create_header_bookmark(
        doc=doc,
        flowable=flowable,
        bookmark_header_levels=page_layout.bookmark_header_levels,
        force_zero_level_bookmark=page_layout.bookmark_force_zero_level,
        zero_level_bookmark_text=page_layout.bookmark_zero_level_text,
    )


# TODO: Add a header level equivalence and a way to reorder the headers
#  So it is guaranteed that the header hierarchy at most increases by 1, even if
#  custom header levels are used
def _create_header_bookmark(
    doc: SimpleDocTemplate,
    flowable: EnhancedFlowable,
    bookmark_header_levels: TBookmarkHeaderLevels = (),
    force_zero_level_bookmark: bool = True,
    zero_level_bookmark_text: str = "REPORT",
) -> None:
    """Add bookmark for headers.

    Also takes care of applying rules for deciding whether the header should be
    bookmarked or not.

    ``force_zero_level_bookmark`` and ``zero_level_bookmark_text`` are related to fixing
    some strange behavior of ``reportlab`` when it comes to the bookmark with hierarchy
    level zero (this is somehow considered a hack to fix the strange behavior).
    """
    if not getattr(flowable, "reporting_info", None):
        # Ensuring that the reporting info are available
        return
    fri = flowable.reporting_info
    if not fri.element_type == ElementType.HEADER:
        return
    if not isinstance(fri, HeaderElementInfo):
        return
    if fri.header_level not in bookmark_header_levels:
        return

    header_level = fri.header_level
    header_hierarchy = _convert_header_level_to_header_hierarchy(
        bookmark_header_levels=bookmark_header_levels,
        level=header_level,
    )
    if force_zero_level_bookmark:
        _force_show_zero_level_bookmark(doc=doc, bookmark_text=zero_level_bookmark_text)

    header_hierarchy = _adjust_header_hierarchy(
        doc=doc,
        header_hierarchy=header_hierarchy,
    )
    flowable_id = fri.uid_string
    bookmark_text = flowable.getPlainText()
    doc.canv.bookmarkPage(flowable_id)
    doc.canv.addOutlineEntry(
        bookmark_text,
        flowable_id,
        level=header_hierarchy,
        closed=None,
    )
    doc.notify("TOCEntry", (header_hierarchy, bookmark_text, doc.page, flowable_id))


def _convert_header_level_to_header_hierarchy(
    bookmark_header_levels: TBookmarkHeaderLevels,
    level: int,
):
    """
    Convert header level to header hierarchy

    ``header_level`` is the style, header hierarchy is used to determine the position
    in the bookmark hierarchy. Typically expected them to be equivalent.
    Header hierarcy must start from 0 (zero) at the top level.
    """
    sorted_unique_levels = sorted(set(bookmark_header_levels))
    level_hierarchy_mapping = {
        level: hierarchy
        for hierarchy, level in enumerate(sorted_unique_levels)
    }
    return level_hierarchy_mapping.get(level)


def _adjust_header_hierarchy(doc: SimpleDocTemplate, header_hierarchy: int):
    """Adjust the header hierarchy so that bookmark creation does not fail.

    Bookmarks can be created at most one level deeper that the current outline level.
    This adjusts the hierarchy so that this is always the case.
    Will result in header being "promoted" to a more important hierarchy than the one
    they are supposed to have (decided that this is acceptable and preferrable compared
    to failing to generate a report).
    """
    current_outline_hierarchy_level = doc.canv._doc.outline.currentlevel  # noqa: WPS437
    max_acceptable_hierarchy_level = current_outline_hierarchy_level + 1
    return min(header_hierarchy, max_acceptable_hierarchy_level)


def _force_show_zero_level_bookmark(doc: SimpleDocTemplate, bookmark_text: str) -> None:
    """ Forces the level-0 bookmarks to appear

    Notes:

    - Adds a dummy bookmark pointing to the first bookmark

      - the text for this dummy bookmark is ``bookmark_text``
    - This is to be consider a hack to deal with strange/unexplaned/ununderstood
    behavior from ``reportlab``. For some reason bookmark with level 0 (zero) sometimes
    is not created as expected.
    """
    if doc.canv._doc.outline.currentlevel != -1:  # noqa: WPS437
        # then it is not the first bookmark being defined, no need to do anything
        return
    dummy_bookmark_id = "__dummy_header_zero"
    doc.canv.bookmarkPage(dummy_bookmark_id)
    doc.canv.addOutlineEntry(
        bookmark_text,
        dummy_bookmark_id,
        level=0,
        closed=None,
    )
    doc.notify("TOCEntry", (0, bookmark_text, doc.page, dummy_bookmark_id))
