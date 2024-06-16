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
from pydantic import BaseModel, Field, field_validator, model_validator


class ControlledParameter(BaseModel):
    """
    Stores attributes for the single-optimizable parameter.

    The parameter value is meant to be within a given interval,
    independently of the values of the other variables in the optimization problem.

    To ensure that we use the following attributes of this class
    when defining a solution's domain for the optimization procedure.

    Args:
        name: control name used to refer control's values in the input dataframe
        op_min: min value of the control that can be set during the optimization
        op_max: max value of the control that can be set during the optimization
        max_delta: the biggest step change that can be done
            from current control's value; all provided nan values are cast to None
        step_size: step size of a change, used by DiscreteGridDomainGenerator
            to make solution space discrete; as a result only multiples of that
            value can be used to make a change; all provided nan values are cast to None
        direction_bound: takes one of three values
            * None – no constraint to control is applied
            * "decrease" – control can only take values lower than its current value
            * "increase" – control can only take values higher than its current value


    Raises:
        ValueError: if ``op_min`` or ``op_max`` is missing/nan
        ValueError: if ``op_min`` is greater than ``op_max``
    """

    name: str
    op_min: float = Field(allow_inf_nan=False)
    op_max: float = Field(allow_inf_nan=False)
    max_delta: tp.Optional[float] = Field(None, allow_inf_nan=False)
    step_size: tp.Optional[float] = Field(None, allow_inf_nan=False)
    direction_bound: tp.Optional[tp.Literal["increase", "decrease"]] = None

    @field_validator("max_delta", "step_size", "direction_bound", mode="before")
    def cast_nan_to_none(
        cls, value: tp.Optional[float],  # noqa: N805, WPS110
    ) -> tp.Optional[float]:
        if value is None or pd.isna(value):
            return None
        return value

    @model_validator(mode="before")
    def check_op_min_is_less_than_op_max(self) -> tp.Any:
        op_max = self.get('op_max')  # type: ignore
        op_min = self.get('op_min')  # type: ignore
        if op_min is None or op_max is None:
            raise ValueError("Both `op_min` and `op_max` are required")
        if op_min > op_max:
            raise ValueError("`op_min` must be less or equal than `op_max`")
        return self
