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

import logging
import typing as tp

import pandas as pd

from modeling import ModelBase
from optimizer import Penalty, Repair
from optimizer import penalty as create_penalty
from recommend import ObjectiveFunction, ProblemFactoryBase
from recommend.types import TRowToOptimize

logger = logging.getLogger(__name__)


class SilicaProblemFactory(ProblemFactoryBase):
    """
    This implementation defines how problems for this use case will be created.

    Users can define their own objective/penalties/repairs and the way they are created
    by implementing abstract ``ProblemFactoryBase`` as we do here.

    After abstract methods are defined, any input data row can be turned into a problem
    by calling ``ProblemFactoryBase.create(...)``
    """

    def _create_objective(self, row_to_optimize: TRowToOptimize) -> ObjectiveFunction:
        return _Objective(self._model_registry["silica_conc_model"]).calculate

    def _create_penalties(self, row_to_optimize: TRowToOptimize) -> tp.List[Penalty]:
        """
        Returns penalties:
            * total flow must be less or equal 1000
        """
        return [get_total_flow_penalty(1000)]

    def _create_repairs(self, row_to_optimize: TRowToOptimize) -> tp.List[Repair]:
        """Returns no repairs since there are none in this use case"""
        return []


class _Objective(object):
    """
    Users can make a custom objective function suited for their use cases.
    Objective function can contain a single model, multiple models, or an equation.

    We define objective as a class to avoid using too many ``functools.partial``
    wrappers.

    Args:
        silica_concentrate_predictor: trained model; note that ``ModelBase`` does
            feature selection internally
    """
    def __init__(self, silica_concentrate_predictor: ModelBase) -> None:
        self.silica_conc_predictor = silica_concentrate_predictor

    def calculate(self, parameters: pd.DataFrame) -> pd.Series:  # noqa: WPS110
        """
        Predicts silica concentrate. Follows ``recommend.types.Predictor`` protocol.

        Args:
            parameters: matrix of plant states to predict concentrate for

        Returns: Objective values.
        """

        return self.silica_conc_predictor.predict(parameters)


def get_total_flow_penalty(upper_limit: float) -> Penalty:
    return create_penalty(_get_total_flow, "<=", upper_limit, name="amina+pulp")


def _get_total_flow(data: pd.DataFrame) -> pd.DataFrame:
    return data["amina_flow"] + data["ore_pulp_flow"]
