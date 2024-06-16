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

import numpy as np
import pandas as pd
from numpy import typing as npt

from modeling.api import SupportsModel

_P = tp.TypeVar("_P", bound=npt.NBitBase)  # noqa: WPS111
_TNumericNDArray = npt.NDArray[np.number[_P]]
_TRange = tp.Tuple[float, float]

_N_SAMPLES_TO_CALCULATE_PREDICTIONS = 10
_DEFAULT_RANDOM_STATE_FOR_SAMPLING = 42


# variables introduced to improve readability
def _create_model_predictions_for_grid(  # noqa: WPS210
    model: SupportsModel,
    feature_name: str,
    feature_grid: _TNumericNDArray[_P],
    data: pd.DataFrame,
    n_sample_to_calculate_predictions: int = _N_SAMPLES_TO_CALCULATE_PREDICTIONS,
    random_state: int | None = _DEFAULT_RANDOM_STATE_FOR_SAMPLING,
) -> tp.List[_TNumericNDArray[_P]]:
    sampled_data = data.sample(
        n=n_sample_to_calculate_predictions,
        # replace=False,
        replace=True,
        random_state=random_state,
    )
    n_values_in_grid = len(feature_grid)
    rows_with_grid_values = []
    for _, row in sampled_data.iterrows():
        row_with_grid_values = pd.DataFrame(
            data=[row for _ in range(n_values_in_grid)],
        ).assign(**{feature_name: feature_grid})
        rows_with_grid_values.append(row_with_grid_values)
    rows_with_grid_values = pd.concat(
        rows_with_grid_values,
        axis=0,
        ignore_index=True,
    )
    raw_predictions = model.predict(rows_with_grid_values)
    predictions = []
    for plot_idx in range(n_sample_to_calculate_predictions):
        idx_subplot_start = plot_idx * n_values_in_grid
        idx_subplot_end = (plot_idx + 1) * n_values_in_grid - 1
        predictions.append(raw_predictions[idx_subplot_start: idx_subplot_end])
    # Ignoring types because of too generic
    # output types in ModelBase
    return predictions  # type: ignore
