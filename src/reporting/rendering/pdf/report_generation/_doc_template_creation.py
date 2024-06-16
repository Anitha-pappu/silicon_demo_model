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

import types
import typing as tp

from reportlab.platypus import SimpleDocTemplate

from reporting.rendering.pdf.content_styles import ContentStyleBase
from reporting.rendering.pdf.page_layout import PageLayout
from reporting.rendering.pdf.page_layout.protocols import (
    TAfterFlowable,
    TAfterPage,
    TStyledAfterFlowable,
    TStyledAfterPage,
)


def create_doc_template(
    file_pdf: str,
    content_style: ContentStyleBase,
    page_layout: PageLayout,
    after_page: tp.Optional[TAfterPage] = None,
    after_flowable: tp.Optional[TAfterFlowable] = None,
    digital_title: tp.Optional[str] = None,
) -> SimpleDocTemplate:
    """Creates the doc template as desired.

    This function takes care of

    - creating the actual doc template
    - (if needed) overriding the the ``afterPage`` and ``afterFlowable`` methods with
    the desired ones in the doc template if needed

    It also takes care of customizing the behavior of ``afterPage`` and/or
    ``afterFlowable`` in case the desired behavior depends on content style and/or page
    layout.
    """
    if after_page is not None:
        after_page = _use_style(
            after_page,
            page_layout=page_layout,
            content_style=content_style,
        )
    if after_flowable is not None:
        after_flowable = _use_style(
            after_flowable,
            page_layout=page_layout,
            content_style=content_style,
        )

    the_doc = SimpleDocTemplate(
        file_pdf,
        pagesize=page_layout.pagesize,
        leftMargin=page_layout.margin_left,
        rightMargin=page_layout.margin_right,
        topMargin=page_layout.margin_top,
        bottomMargin=page_layout.margin_bottom,
        title=digital_title,
    )
    return _override_methods(
        doc_template=the_doc,
        after_page=after_page,
        after_flowable=after_flowable,
    )


@tp.overload
def _use_style(
    fn: TAfterFlowable,
    content_style: ContentStyleBase = None,
    page_layout: PageLayout = None,
    content_style_arg_name: str = "content_style",
    page_layout_arg_name: str = "page_layout",
) -> TStyledAfterFlowable:
    """For typing in case of an after-flowable function"""


@tp.overload
def _use_style(  # noqa: WPS440
    fn: TAfterPage,
    content_style: ContentStyleBase = None,
    page_layout: PageLayout = None,
    content_style_arg_name: str = "content_style",
    page_layout_arg_name: str = "page_layout",
) -> TStyledAfterPage:
    """For typing in case of an after-page function"""


def _use_style(  # noqa: WPS440
    fn: tp.Union[TAfterFlowable, TAfterPage],
    content_style: ContentStyleBase = None,
    page_layout: PageLayout = None,
    content_style_arg_name: str = "content_style",
    page_layout_arg_name: str = "page_layout",
) -> tp.Union[TStyledAfterFlowable, TStyledAfterPage]:
    """Use the desired style in the functions"""
    content_style_kwarg = (
        {} if content_style is None else {content_style_arg_name: content_style}
    )
    page_layout_kwarg = (
        {} if page_layout is None else {page_layout_arg_name: page_layout}
    )
    style_kwargs = {
        **content_style_kwarg,
        **page_layout_kwarg,
    }

    def styled_fn(*args, **kwargs):  # noqa: WPS430
        return fn(*args, **style_kwargs, **kwargs)
    return styled_fn


def _override_methods(
    doc_template: SimpleDocTemplate,
    after_page: tp.Optional[TStyledAfterPage] = None,
    after_flowable: tp.Optional[TStyledAfterFlowable] = None,
) -> SimpleDocTemplate:
    """Overwrite selected methods of ``doc_template`` with other desired methods.

    These selected methods of the ``SimpleDocTemplate`` which can be overridden are:

    - ``afterPage``
    -  ``afterFlowable``

    The ``afterPage`` method is called after a page is drawn, and can be useful for
    customizing the page appearance (e.g. by drawing header, footer, page numbers, etc.)
    The ``afterFlowable`` method is called after a flowable is drawn, and can be useful
    for adding custom elements that rely on the flowable position after it has been
    drawn (e.g. bookmarks that link to a header can be created using a reference to the
    flowable after it has been drawn)
    """

    if after_page is not None:
        doc_template.afterPage = types.MethodType(
            after_page,
            doc_template,
        )
    if after_flowable is not None:
        doc_template.afterFlowable = types.MethodType(
            after_flowable,
            doc_template,
        )
    return doc_template
