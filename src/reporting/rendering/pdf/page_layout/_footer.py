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

""" Basic functionality to customize a page

These functions are meant to be used in the ``afterPage`` method of a
``SimpleDocTemplate``, after ``content_style`` and ``page_style`` have been applied

These functions who do not need a ``content_style`` or a ``page_style`` kwarg, take in
``**kwargs``, so that the ``content_style`` and ``page_style`` parameter can be passed
without causing an error.
This enables implementing a common interface for all the stylable functions.
"""

import typing as tp

from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate

from reporting.rendering.pdf.content_styles import ContentStyleBase
from reporting.rendering.pdf.page_layout import PageLayout


def write_page_number(doc: SimpleDocTemplate, **kwargs):
    """Write the page number at the bottom of the page"""
    page_number = doc.canv.getPageNumber()
    page_number_text = str(page_number)
    bottom_coord = doc.frame.bottomPadding
    top_coord = bottom_coord + doc.bottomMargin
    middle_coord = 0.5 * (bottom_coord + top_coord)
    left_coord = doc.frame.leftPadding + doc.leftMargin
    right_coord = left_coord + doc.width
    center_coord = 0.5 * (left_coord + right_coord)
    doc.canv.drawCentredString(x=center_coord, y=middle_coord, text=page_number_text)


def write_page_footer(doc, content_style: ContentStyleBase, page_layout: PageLayout):
    footer_text = page_layout.footer_text if page_layout.footer_show else None
    # TODO: Check - this return is likely not needed
    #  Just calling _write_page_footer should be enough. Verify if true and change if
    #  needed.
    return _write_page_footer(
        doc=doc,
        text=footer_text,
        footer_style=content_style.footer_style,
        also_on_first_page=page_layout.footer_on_first_page,
    )


def _write_page_footer(
    doc: SimpleDocTemplate,
    text: tp.Union[str, None],
    footer_style: tp.Optional[ParagraphStyle] = None,
    also_on_first_page: bool = True,
):
    """ Writes the page footer.

    If ``text`` is ``None``, then does not write the footer.

    Solution following the example here:
    https://stackoverflow.com/questions/8827871/a-multilineparagraph-footer-and-header-in-reportlab
    """
    if text is None:
        return
    if not also_on_first_page:
        page_number = doc.canv.getPageNumber()
        if page_number == 1:
            return
    doc.canv.saveState()
    left_margin_right = doc.frame.leftPadding + doc.leftMargin
    drawable_width = doc.width - doc.frame.leftPadding - doc.frame.rightPadding
    footer_paragraph = Paragraph(text=text, style=footer_style)
    _footer_paragraph_width, footer_paragraph_height = footer_paragraph.wrap(
        drawable_width, doc.bottomMargin,
    )
    footer_paragraph.drawOn(doc.canv, left_margin_right, footer_paragraph_height)
    doc.canv.restoreState()
