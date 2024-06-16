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

from optimizer.constraint.handler import BaseHandler
from optimizer.types import (
    MapsMatrixToMatrix,
    Matrix,
    ReducesMatrixToSeries,
    Vector,
)
from optimizer.utils.validation import (
    map_with_dim_check,
    reduce_with_dim_check,
)

_BASE_HANDLER_PROPERTIES = (
    "constraint",
    "constraint_values",
    "distances",
    "constraint_func",
)


class UserDefinedPenalty:
    def __init__(self, penalty: ReducesMatrixToSeries) -> None:
        self._penalty = penalty
        self.calculated_penalty: tp.Optional[Vector] = None
        _add_not_implemented_base_handler_properties(self)

    def __call__(self, parameters: Matrix, **kwargs: tp.Any) -> Vector:
        self.calculated_penalty = reduce_with_dim_check(parameters, self._penalty)
        if isinstance(parameters, pd.DataFrame):
            self.calculated_penalty = pd.Series(
                self.calculated_penalty, index=parameters.index, name=self.name,
            )
        return self.calculated_penalty

    @property
    def penalty_multiplier(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return _get_name(self._penalty, self.__class__)


class UserDefinedRepair:
    def __init__(self, repair: MapsMatrixToMatrix) -> None:
        self._repair = repair
        _add_not_implemented_base_handler_properties(self)

    def __call__(self, parameters: Matrix, **kwargs: tp.Any) -> Matrix:
        return map_with_dim_check(parameters, self._repair)

    @property
    def name(self) -> str:
        return _get_name(self._repair, self.__class__)


def _add_not_implemented_base_handler_properties(obj_reference: tp.Any) -> None:
    for base_handler_property in _BASE_HANDLER_PROPERTIES:
        setattr(
            obj_reference,
            base_handler_property,
            partial(_raise_not_implemented, obj_reference.__class__),
        )


def _raise_not_implemented(udf_wrapper_type: tp.Type[tp.Any]) -> None:
    class_name = udf_wrapper_type.__name__
    raise NotImplementedError(f"Isn't defined for {class_name}")


def _get_name(callable_obj: tp.Any, udf_wrapper_type: tp.Type[tp.Any]) -> str:
    class_name = udf_wrapper_type.__name__
    callable_name = callable_obj.__name__ if hasattr(callable_obj, "__name__") else ""
    return f"{class_name}({callable_name})"
