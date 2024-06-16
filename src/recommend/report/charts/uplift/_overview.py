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

import pandas as pd

from reporting.rendering.identifiers import Text

_TITLE_SIZE = 16
_LEFT_MARGIN = 50


def plot_impact_summary(
    data: pd.DataFrame,
    impact: pd.DataFrame,
) -> Text:
    """
    Generates a list of figures with impact overview.

    Args:
        data: Dataframe with baseline, optimal and actual values
        impact: Dataframe with impact analysis

    Returns:
        List of impact overview figures

    """
    start = data["timestamp"].min()
    end = data["timestamp"].max()
    n_observations = data.shape[0]
    impact = impact[impact["group"] == "all_data"]["uplift"].iloc[0]
    return Text(
        text=f"start: {start}\n"
        f"end : {end}\n"
        f"observations: {n_observations}\n"
        f"impact: {impact:.0f}",
        title="Timeframe and impact overview",
        title_size=_TITLE_SIZE,
        left_margin=_LEFT_MARGIN,
    )
