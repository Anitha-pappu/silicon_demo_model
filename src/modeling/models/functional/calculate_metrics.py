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

import pandas as pd

from ..model_base import EvaluatesMetrics


def calculate_metrics(
    data: pd.DataFrame,
    model: EvaluatesMetrics,
    **predict_kwargs: tp.Any,
) -> tp.Dict[str, float]:
    """
    Calculate a standard set of model's regression metrics for given data.
    Automatically infers actual target values (based on target column name specified
    in model), generates predictions, and uses the two to calculate performance metrics.

    Args:
        data: to calculate metrics for.
        model: trained instance of ModelBase.
        predict_kwargs: keyword arguments to ``.predict()`` method.

    Returns:
        A dictionary of metrics names and values.
    """
    return model.evaluate_metrics(data, **predict_kwargs)
