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

from modeling.api import (
    SupportsEvaluateMetrics,
    SupportsModel,
    SupportsShapFeatureImportance,
)


class SupportsModelAndEvaluateMetrics(
    SupportsModel,
    SupportsEvaluateMetrics,
    tp.Protocol,
):
    """This protocol describes the model that is able to produce metrics"""


class Model(
    SupportsModel,
    SupportsEvaluateMetrics,
    SupportsShapFeatureImportance,
    tp.Protocol,
):
    """
    This is a Protocol that unifies following protocols:
        * ``SupportsModel``
        * ``SupportsEvaluateMetrics`.`
        * ``SupportsShapFeatureImportance`.`

    See parental protocols for details.
    """
