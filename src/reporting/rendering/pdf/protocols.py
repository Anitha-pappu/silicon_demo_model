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

from reportlab.platypus import Flowable

from reporting.rendering.pdf.renderables import ElementInfo


class HasSetReportingInfo(tp.Protocol):

    @property
    def reporting_info(self) -> ElementInfo:
        """``self.reporting_info`` attribute"""


class EnhancedFlowable(Flowable, HasSetReportingInfo):
    """
    Check that has both ``Flowable`` methods and the ``reporting_info`` attribute.
    """
