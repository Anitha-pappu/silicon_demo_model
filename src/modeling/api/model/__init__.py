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


from ._cross_validatable import TCrossValidatableModel
from ._estimator import Estimator
from ._evaluates_metrics import SupportsEvaluateMetrics
from ._factory import SupportsModelFactory
from ._model import ContainsEstimator, SupportsModel
from ._produces_shap_importances import (
    ShapExplanation,
    SupportsShapFeatureImportance,
)
from ._tuner import SupportsModelTuner
