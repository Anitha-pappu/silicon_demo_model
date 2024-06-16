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
"""Schemas required by pydantic to validate the format of the configuration"""

import typing as tp

from pydantic import BaseModel, conlist, field_validator

from .utils import load_obj


class DerivedFeaturesRecipe(BaseModel):
    """Holder of necessary info for creating an eng feature"""

    dependencies: conlist(str, min_length=1)  # type: ignore
    function: tp.Callable[..., tp.Any]
    args: tp.Iterable[tp.Any] = []
    kwargs: tp.Mapping[str, tp.Any] = {}

    @field_validator("function", mode="before")
    def load_func(
        cls,  # noqa: N805
        func: tp.Union[tp.Callable[..., tp.Any], str],
    ) -> tp.Callable[..., tp.Any]:
        """if `func` is a path to an object, load it."""
        if isinstance(func, str):
            loaded_obj: tp.Callable[..., tp.Any] = load_obj(func)
            return loaded_obj  # noqa: WPS331
        return func

    class Config(object):
        arbitrary_types_allowed = True


class DerivedFeaturesCookBook(BaseModel):
    """Holder of info to create all eng features"""

    cookbook: tp.Dict[str, DerivedFeaturesRecipe]
