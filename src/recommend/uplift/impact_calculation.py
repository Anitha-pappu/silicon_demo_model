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

import numpy as np
import pandas as pd

from .uplifts import BaselineUplifts


def get_impact_estimation(
    uplifts: BaselineUplifts,
    annualize: bool = False,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Calculate impact estimation from uplifts. First, uplifts are added to find the
    impact and reported by the name 'all_data'. Then, if ``by_group`` is True, uplifts
    are also added by group and reported under the group name. Finally, if annualize
    is True, the impact is extrapolated to a year period.

    Args:
        uplifts: Uplifts to calculate impact estimation for.
        annualize: Whether to annualize the impact estimation. If True, the impact found
            in the period of time between the start and end of the uplifts will be
            extrapolated to a year period, assuming that the impact per time unit is
            constant.
        timestamp_col: Name of the timestamp column in ``uplift`` data.

    Returns:
        Dataframe with impact estimation.

    """

    impact = pd.DataFrame(
        {
            "group": uplifts.group_names,
            "uplift": np.nan,
        },
    )
    for group, uplifts_group in uplifts.group_iterator():
        impact.loc[impact["group"] == group, "uplift"] = uplifts_group["uplift"].sum()

    if annualize:
        min_timestamp = uplifts.data[timestamp_col].min()
        max_timestamp = uplifts.data[timestamp_col].max()
        if min_timestamp == max_timestamp:
            raise ValueError(
                "Cannot annualize impact estimation when all uplifts are on the same "
                "timestamp.",
            )
        impact["uplift"] = impact["uplift"] / (
            (max_timestamp - min_timestamp) / pd.Timedelta("365D")
        )

    return impact
