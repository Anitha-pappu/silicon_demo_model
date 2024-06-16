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
This subpackage implements report generation.
It is done in several steps:
    * template loading; done by ``report.templates``
    * input rendering; done by ``report.rendering``
    * generation report file; done by ``report.report_generation``
"""
from .renderables import (
    InteractiveHtmlContentBase,
    InteractiveHtmlRenderableCode,
    InteractiveHtmlRenderableTable,
    ReprImplementationError,
    plot_code,
    plot_table,
)
from .report_generation import generate_html_report
from .templates import TemplateBase, TemplateWithCodeHighlighter
