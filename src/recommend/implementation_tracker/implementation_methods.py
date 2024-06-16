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


def deviation_implementation_status(
    implementation_data: pd.DataFrame,
    sensitivity: float | dict[str, float],
    sensitivity_type: tp.Literal["rel", "abs"],
) -> pd.DataFrame:
    """
    Calculates implementation status by assessing the percentage of time a sensor is
    within a range of its target value: If no data is available for a sensor, percentage
    is set to nan.

    Args:
        implementation_data: data ready for implementation status calculations.
        sensitivity: allowed deviation from target value. It can be a float (value is
            used for all controls) or a mapping from control to float (per control
            specific deviation).
        sensitivity_type: if "rel", ``sensitivity`` represents the percentage
            deviation allowed. If "abs", it represents the absolute one.

    Returns:
        Dataframe with one row per recommendation and run id with the deviation
        implementation status percentage

    Raises:
        ValueError: if ``sensibility`` is not a float or a dict or ``sensibility_type``
            is not "rel" or "abs"
    """
    implementation_data = implementation_data.copy()
    if isinstance(sensitivity, dict):
        sensitivity_values = implementation_data['tag_id'].map(sensitivity)
    elif isinstance(sensitivity, float):
        sensitivity_values = sensitivity
    else:
        raise ValueError("sensibility must be a float or a dictionary")

    if sensitivity_type == "rel":
        implementation_data['implementation_perc'] = (
            implementation_data['current_value'].between(
                implementation_data['recommended_value'] * (1 - sensitivity_values),
                implementation_data['recommended_value'] * (1 + sensitivity_values),
            )
        ).astype("float")
    elif sensitivity_type == "abs":
        implementation_data['implementation_perc'] = (
            implementation_data['current_value'].between(
                implementation_data['recommended_value'] - sensitivity_values,
                implementation_data['recommended_value'] + sensitivity_values,
            )
        ).astype("float")
    else:
        raise ValueError("sensibility_type must be 'rel' or 'abs'")

    # Implementation percentage cannot be calculated if current value is nan, so it
    # is assigned to nan
    implementation_data['implementation_perc'] = np.where(
        implementation_data['current_value'].isna(),
        np.nan,
        implementation_data['implementation_perc'],
    )

    return implementation_data[["id", "tag_id", "run_id", "implementation_perc"]]


def progress_implementation_status(
    implementation_data: pd.DataFrame,
    clip: bool = False,
) -> pd.DataFrame:
    """
    Calculates implementation status defined as the ratio
        actual change / suggested change
    where
        actual change = actual value - original value
        suggested change = recommended value - original value.

    Example::

        >>> imp_data = pd.DataFrame(
        ...     [
        ...         ["tag_1", "run_1", 100, 50, "id_1", 75],
        ...         ["tag_1", "run_2", 100, 50, "id_2", 25],
        ...         ["tag_1", "run_3", 100, 50, "id_3", 125],
        ...     ],
        ...     columns=["tag_id", "run_id", "recommended_value", "before_recs_value",
        ...     "id", "current_value"]
        ...)
        >>> progress_implementation_status(imp_data)
             id tag_id run_id  implementation_perc
        0  id_1  tag_1  run_1                  0.5
        1  id_2  tag_1  run_2                 -0.5
        2  id_3  tag_1  run_3                  1.5
        >>> progress_implementation_status(imp_data, clip=True)
             id tag_id run_id  implementation_perc
        0  id_1  tag_1  run_1                  0.5
        1  id_2  tag_1  run_2                  0.0
        2  id_3  tag_1  run_3                  0.5

    Args:
        implementation_data: Data ready for implementation status calculations.
        clip: If True, each ratio is first clipped between 0 and 2. Then, each value
            above 1 is considered to have implementation status of 2 - ratio.

    Returns:
        Dataframe with one row per recommendation and run id with the deviation
        implementation status percentage
    """
    implementation_data = implementation_data.copy()
    actual_change = (
        implementation_data['current_value'] - implementation_data['before_recs_value']
    )
    rec_change = (
        implementation_data['recommended_value']
        - implementation_data['before_recs_value']
    )
    implementation_perc = actual_change / rec_change
    if clip:
        implementation_perc = implementation_perc.clip(0, 2)
        implementation_perc = np.where(
            implementation_perc > 1,
            2 - implementation_perc,
            implementation_perc,
        )
    implementation_data['implementation_perc'] = implementation_perc

    return implementation_data[["id", "tag_id", "run_id", "implementation_perc"]]
