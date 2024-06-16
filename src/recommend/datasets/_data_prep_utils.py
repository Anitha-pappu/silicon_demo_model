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

from optimizer import Penalty, penalty
from recommend.solution import Solutions

SEED = 42
_COLUMNS_TO_MASK_RANDOMLY = (
    "air_flow01",
    "air_flow02",
    "air_flow03",
    "air_flow04",
    "air_flow05",
    "air_flow06",
    "air_flow07",
    "column_level01",
    "column_level02",
    "column_level03",
    "column_level04",
    "column_level05",
    "column_level06",
    "column_level07",
    "ore_pulp_flow",
    "ore_pulp_ph",
    "ore_pulp_density",
)


def add_mock_penalties(solutions: Solutions) -> None:
    rand = np.random.default_rng(SEED)
    for solution in solutions.values():
        solution.problem.penalties.extend(
            [_create_first_penalty(rand), create_second_penalty(rand)],
        )


def create_second_penalty(rand: np.random.Generator) -> Penalty:
    return penalty(
        lambda x: rand.normal(50, 200, size=len(x)),  # noqa: WPS111, WPS432
        ">=",
        100,
        name="second_random",
    )


def _create_first_penalty(rand: np.random.Generator) -> Penalty:
    return penalty(
        lambda x: rand.normal(100, 30, size=len(x)),  # noqa: WPS111, WPS432
        "<=",
        120,  # noqa: WPS432
        name="first_random",
    )


def mask_values_randomly(data: pd.DataFrame, n_readings_to_mask: int) -> None:
    random_coordinates_to_mask = list(set(zip(
        np.random.choice(data.index, n_readings_to_mask, replace=True),
        np.random.choice(_COLUMNS_TO_MASK_RANDOMLY, n_readings_to_mask, replace=True),
    )))
    for coord in random_coordinates_to_mask:
        data.loc[coord[0], coord[1]] = np.nan
