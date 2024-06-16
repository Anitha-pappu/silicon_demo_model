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
from functools import partial

import pandas as pd

from optimizer import Penalty, Repair
from optimizer import penalty as create_penalty
from optimizer import repair as create_repair
from recommend import ObjectiveFunction, ProblemFactoryBase
from recommend.types import TProblem, TRowToOptimize

_FIRST_REPAIR_CONSTRAINT_VALUE = 9.705
_SECOND_REPAIR_CONSTRAINT_VALUE = 9.6
_FIRST_PENALTY_CONSTRAINT_VALUE = 4000
_SECOND_PENALTY_CONSTRAINT_VALUE = 9.8
_FIRST_PENALTY_MULTIPLIER = 0.0125
_SECOND_PENALTY_MULTIPLIER = 5


class ProblemFactory(ProblemFactoryBase[TProblem]):
    def _create_objective(self, row_to_optimize: TRowToOptimize) -> ObjectiveFunction:
        return self._model_registry["silica_conc_predictor"].predict  # type: ignore

    def _create_penalties(self, row_to_optimize: TRowToOptimize) -> tp.List[Penalty]:
        flow_penalty = create_penalty(  # noqa: WPS317
            _calculate_total_flow, ">=", _FIRST_PENALTY_CONSTRAINT_VALUE,
            name="starch_and_amina_flow",
            penalty_multiplier=_FIRST_PENALTY_MULTIPLIER,
        )
        ore_pulp_penatly = create_penalty(  # noqa: WPS317
            "ore_pulp_ph", "<=", _SECOND_PENALTY_CONSTRAINT_VALUE,
            penalty_multiplier=_SECOND_PENALTY_MULTIPLIER,
        )
        return [flow_penalty, ore_pulp_penatly]

    def _create_repairs(self, row_to_optimize: TRowToOptimize) -> tp.List[Repair]:
        first_ore_pulp_ph_repair = create_repair(
            "ore_pulp_ph",
            "<=",
            _FIRST_REPAIR_CONSTRAINT_VALUE,
            repair_function=partial(
                _set_value_to_a_column,
                column="ore_pulp_ph",
                value_to_set=_FIRST_REPAIR_CONSTRAINT_VALUE,
            ),
            name=f"ore_pulp_ph upper bound ({_FIRST_REPAIR_CONSTRAINT_VALUE:0.2f})",
        )
        second_ore_pulp_ph_repair = create_repair(  # noqa: WPS317
            "ore_pulp_ph", ">=", _SECOND_REPAIR_CONSTRAINT_VALUE,
            repair_function=partial(
                _set_value_to_a_column,
                column="ore_pulp_ph",
                value_to_set=_SECOND_REPAIR_CONSTRAINT_VALUE,
            ),
            name=f"ore_pulp_ph lower bound ({_SECOND_REPAIR_CONSTRAINT_VALUE:0.2f})",
        )
        amina_flow_allowed_values = [450, 500, 550, 600, 650, 700]
        amina_flow_set_repair = create_repair(  # noqa: WPS317
            "amina_flow", "in", amina_flow_allowed_values,
            name="amina_flow in set of allowed values",
        )
        return [
            first_ore_pulp_ph_repair,
            second_ore_pulp_ph_repair,
            amina_flow_set_repair,
        ]


def _calculate_total_flow(data: pd.DataFrame) -> pd.Series:
    return data["starch_flow"] + data["amina_flow"]


def _set_value_to_a_column(
    data: pd.DataFrame,
    column: str,
    value_to_set: float,
) -> pd.DataFrame:
    data = data.copy()
    data[column] = value_to_set
    return data
