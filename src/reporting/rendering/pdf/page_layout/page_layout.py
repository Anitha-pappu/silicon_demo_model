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

from reportlab.lib.pagesizes import A4, landscape, portrait

_PAGE_SIZE = A4
_PAGE_ORIENTATION_LANDSCAPE = False

_MARGIN_LEFT = 10
_MARGIN_RIGHT = 10
_MARGIN_TOP = 10
_MARGIN_BOTTOM = 50

BOOKMARK_HEADER_LEVELS = (0, 1, 2)
_BOOKMARK_FORCE_ZERO_LEVEL = True
_BOOKMARK_ZERO_LEVEL_TEXT = "Report"

_FOOTER_SHOW = True
_FOOTER_TEXT = "report footer text"
_FOOTER_ON_FIRST_PAGE = False


class PageLayout(object):
    """Base class that takes care of
    - exposing more pythonic names for input arguments
    - collecting the page-related part of the configuration
    """

    def __init__(  # noqa: WPS211
        self,
        page_size=_PAGE_SIZE,
        page_orientation_landscape=_PAGE_ORIENTATION_LANDSCAPE,
        margin_left=_MARGIN_LEFT,
        margin_right=_MARGIN_RIGHT,
        margin_top=_MARGIN_TOP,
        margin_bottom=_MARGIN_BOTTOM,
        bookmark_header_levels=BOOKMARK_HEADER_LEVELS,
        bookmark_force_zero_level=_BOOKMARK_FORCE_ZERO_LEVEL,
        bookmark_zero_level_text=_BOOKMARK_ZERO_LEVEL_TEXT,
        footer_show=_FOOTER_SHOW,
        footer_text=_FOOTER_TEXT,
        footer_on_first_page=_FOOTER_ON_FIRST_PAGE,
    ) -> None:
        self._page_orientation_landscape = page_orientation_landscape
        self._page_size = page_size
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.bookmark_header_levels = bookmark_header_levels
        self.bookmark_force_zero_level = bookmark_force_zero_level
        self.bookmark_zero_level_text = bookmark_zero_level_text
        self.footer_show = footer_show
        self.footer_text = footer_text
        self.footer_on_first_page = footer_on_first_page

    @property
    def pagesize(self):
        """A tuple with the dimensions of the page.

        This is either the same as the size or the size with the values exchanged,
        according to what produces the desired orientation.

        Notice that ``pagesize`` is the property meant to be accessed directly,
        not ``_page_size`` nor ``_page_orientation_landscape``
        """
        if self._page_orientation_landscape:
            return landscape(self._page_size)
        return portrait(self._page_size)
