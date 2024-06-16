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
from pathlib import Path

# TODO: Move the metadata to common (when common will exist)
from reporting.rendering.html.report_generation import (
    ReportMetaData,
    TMetaData,
)
from reporting.rendering.pdf.content_styles import (
    ContentStyleBase,
    DefaultContentStyle,
)
from reporting.rendering.pdf.page_layout.page_layout import PageLayout
from reporting.rendering.pdf.protocols import EnhancedFlowable
from reporting.rendering.pdf.renderables import (
    PdfRenderableElement,
    convert_recursively_into_pdf_renderable,
)
from reporting.rendering.pdf.renderables.protocols import TPdfRenderableDict
from reporting.rendering.types.identifiers import (
    remove_section_description_from_structure,
)

from ._doc_template_creation import create_doc_template  # noqa: WPS436


# TODO: Add unit tests that check that when you build the doc,
#  the drawable area is what is expected
class PdfRenderableReport(object):
    """ Built to guarantee that the drawable area is what we expect.

    Notes:
        This class is defined in order to provide the information of the drawable area
        _before_ the doc is build, so that this can be used later on for

        - setting page breaks where necessary
        - resizing components if needed


        The following is a detailed explanation of how things work in reportlab and why
        this class exists.

        When

        - a ``SimpleDocTemplate`` is instantiated as
          ``doc = SimpleDocTemplate(<path/to/file>)``

        - the doc is built with ``doc.build([<non empty list of story elements>])``
        then the available height where one can draw on on the page is (from top to
        bottom):

          - ``doc.topMargin``
          - `doc.frame.topPadding`
          - (drawable)
          - ``doc.frame.bottomPadding``
          - ``doc.bottomMargin``


        Also

        - ``doc.pagesize`` is a tuple with the width and height of the "paper".
        - ``doc.height`` is what you get after removing the margins, so

            - ``doc.height =  doc.pagesize[1] - doc.topMargin - doc.bottomMargin``
        - (it should work in a similar way for ``doc.width``


        Since the space where you can draw is what you get after removing the padding,
        you can get the available height for drawing as:
        ``doc.height - doc.frame.topPadding - doc.frame.bottomPadding``

        The issue is that ``doc.frame`` seems to be defined only after one draws the pdf
        with ``doc.build([<non empty list of story elements>])``.
        This means that the top and bottom padding are not available.

        Since they depend on the settings of how the frame is build in
        ``SimpleDocTemplate`` (or another DocTemplate), they _can_ actually be known in
        advance.
        Creating this class that is aware of the size of the drawable area, solves these
        issues.
    """

    TOP_PADDING = 6  # default top padding in SimpleDocTemplate = 6
    BOTTOM_PADDING = 6  # default bottom padding in SimpleDocTemplate = 6
    LEFT_PADDING = 6  # default left padding in SimpleDocTemplate = 6
    RIGHT_PADDING = 6  # default right padding in SimpleDocTemplate = 6

    DEFAULT_CONTENT_STYLE = DefaultContentStyle
    _MAIN_TITLE_LEVEL_HEADER = 0
    _DEFAULT_INITIAL_LEVEL_HEADER = 1

    def __init__(
        self,
        file_pdf: tp.Union[str, Path],
        page_layout: tp.Optional[PageLayout] = None,
        content_style: ContentStyleBase = DEFAULT_CONTENT_STYLE,
        after_page=None,
        after_flowable=None,
        digital_title=None,
    ) -> None:
        """ ``file_pdf`` is where the pdf file will be saved"""
        self._file_pdf = Path(file_pdf)
        self._content_style = content_style
        self._page_layout = PageLayout() if page_layout is None else page_layout
        self._doc = create_doc_template(
            file_pdf=str(self._file_pdf),
            content_style=content_style,
            page_layout=self._page_layout,
            after_page=after_page,
            after_flowable=after_flowable,
            digital_title=digital_title,
        )

    @property
    def content_style(self):
        """The style of the content elements in the document.

        Contains info about the style of the content.
        """
        return self._content_style

    @property
    def drawable_height(self):
        """ The height of the drawable area in the document.

        Notes:
            Defined here so it is available before build.
        """
        return (self._doc.height - self.TOP_PADDING - self.BOTTOM_PADDING)

    @property
    def drawable_width(self):
        """ The width of the drawable area in the document.

        Notes:
            Defined here so it is available before build.
        """
        return (self._doc.width - self.LEFT_PADDING - self.RIGHT_PADDING)

    # TODO: Decide what to do with ``_INITIAL_LEVEL_HEADER = 1``, which originally came
    #  from the html renderer. Could likely be kept equal to 1, and level zero could be
    # reserved for the report title (that can be provided as a keyword argument to the
    # report)
    def create_report(self, report_structure, title=None):
        """Build a pdf file and saves it.

        The steps are

        - convert the input ``report_structure`` into a ``renderable_story``, which is a
        list of ``PdfRenderableElement``s
        - convert the ``renderable_story`` into a ``reportlab_story``, where the
        ``PdfRenderableElement``s are converted into ``Flowable``s (from
        ``reportlab.platypus``)
        - build and save a pdf from the ``reportlab_story`` by leveraging the
        ``reportlab`` functionality

        Notes:
            The file is saved using the ``file_path`` provided when instantiating the
            ``PdfRenderableReport``
        """
        renderable_story = self._prepare_title(
            report_structure=report_structure,
            title=title,
        )
        initial_level_header = (
            self._MAIN_TITLE_LEVEL_HEADER if title
            else self._DEFAULT_INITIAL_LEVEL_HEADER
        )
        renderable_story = convert_recursively_into_pdf_renderable(
            section_data=report_structure,
            header_level=initial_level_header,
        )
        reportlab_story = [
            self._convert_to_reportlab_object(element)
            for element in renderable_story
        ]
        self._build_pdf(story=reportlab_story)

    def _convert_to_reportlab_object(
        self,
        element: PdfRenderableElement,
    ) -> EnhancedFlowable:
        """ Converts a pdf-renderable ``element`` into an enhanced reportlab flowable.

        Notes:
            On the enhancement
            After converting the pdf-renderable element into a reportlab ``Flowable``,
            it "enhances" it by adding a ``.reporting_info`` attribute that contains
            information that can be leveraged later when rendering the pdf.

            On the conversion
            When the conversion happens, it also makes sure that the resulting object
            either:

            - fits in the page
            - can be broken down so that it fits in the page
            - can be made smaller so that it fits in the page


            Most of the above is accomplished almost natively by the reportlab objects,
            provided that they are aware of the size of the drawable area.

            This is why this method needs to exist within this class, where the size of
            the page (and of the drawable area) is known and the information available.
        """
        reportlab_flowable = element.to_reportlab_from_renderable_report(
            renderable_report=self,
        )
        enhanced_flowable = reportlab_flowable
        enhanced_flowable.reporting_info = element.get_info()
        return enhanced_flowable

    def _prepare_title(
        self,
        report_structure: TPdfRenderableDict,
        title: tp.Optional[str] = None,
    ):
        """Take care of the report title.

        - if a ``title`` is provided, wraps the whole report in a dict, where ``title``
        is the key, so it will become the main report title
        - otherwise returns the report structure unchanged
        """
        # TODO: Distinguish title from digital title and have the main title of the
        #  report as main title in the report_structure. So content and metainfo are
        #  separated clearly
        if title:
            report_structure = {title: report_structure}
        return report_structure

    def _build_pdf(self, story):
        """Builds the pdf

        Also creates the necessary folders in case some folder in the path to
        ``self._file_pdf`` does not exist

        Uses ``self._doc.multiBuild(story)`` instead of ``self._doc.build(story)``.
        This is helpful when some pdf builds require multiple passes at the document,
        and it does not seem to be harmful when only one pass is enough. Therefore it
        was decided to keep it
        """
        output_folder_path = Path(self._file_pdf).parent
        output_folder_path.mkdir(parents=True, exist_ok=True)
        self._doc.multiBuild(story)


def generate_pdf_report(
    report_structure: TPdfRenderableDict,
    render_path: tp.Union[str, Path],
    report_meta_data: tp.Optional[TMetaData] = None,
    **kwargs,
):
    """Generate pdf report.

    Notes:

        - ``kwargs`` are passed to the report template
        - ``report_meta_data`` is used to

          - retrieve ``title``, passed to the ``.create_report`` method of the report
          template and used as main title of the pdf
          - retrieve ``digital_title``, passed to ``PdfRenderableReport`` and displayed
          at the top of the tab when seen on the screen
        - if ``digital_title`` is missing in the metadata, then ``title`` is used as the
        digital title
    """
    report_structure = remove_section_description_from_structure(report_structure)
    report_meta_data = ReportMetaData.from_input(report_meta_data)

    # TODO: Handle metadata for both pdf and html in one place
    # TODO: handle title properly
    # TODO: Remove this try-except by handling metadata properly and taking care of
    #  digital_title there
    title = report_meta_data.title
    try:
        digital_title = report_meta_data.digital_title
    except AttributeError:
        digital_title = title

    report_template = PdfRenderableReport(
        file_pdf=render_path,
        digital_title=digital_title,
        **kwargs,
    )
    report_template.create_report(
        report_structure=report_structure,
        title=title,
    )
