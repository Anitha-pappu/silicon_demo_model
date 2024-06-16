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

from typing_extensions import Self

from ..tag_dict import SupportsTagDict
from ._estimator import Estimator  # noqa: WPS436
from ._model import SupportsModel  # noqa: WPS436


@tp.runtime_checkable
class SupportsModelFactory(tp.Protocol):
    """
    Any object inherited from ``modeling.ModelFactoryBase`` satisfies this protocol.

    Factory's main method is ``create()`` that produces a model
    i.e., an object that ``SupportsModel``.
    """
    def __init__(
        self,
        model_init_config: tp.Dict[str, tp.Any],
        features_in: tp.Iterable[str],
        target: str,
    ) -> None:
        """
        Defines the structure of `model_init_config`.

        Args:
            model_init_config: Model initialization config follows structure
             of ModelFactoryBase inheritor.
            features_in: Features column names to be used for
             model training and prediction
            target: Name of the column used in model training
        """

    @classmethod
    def from_tag_dict(
        cls: tp.Type[Self],
        model_init_config: tp.Dict[str, tp.Any],
        tag_dict: SupportsTagDict,
        tag_dict_features_column: str,
        target: str,
    ) -> Self:
        """
        Initializes ModelBuilder from the ``tag_dict``.

        Method fetches model features and other information which potentially
        can be used for model initialization from the ``tag_dict``.

        Args:
            model_init_config: Model initialization config follows structure of
             ModelFactoryBase inheritor.
            tag_dict: Instance of TagDict with tag-level information
            tag_dict_features_column: Column name from TagDict to be used for
             identifying model features
            target: Column name to be used as model target

        Returns:
            ModelFactoryBase initialized instance
        """

    @property
    def model_init_config(self) -> tp.Dict[str, tp.Any]:
        """Returns model's init config"""

    @property
    def features_in(self) -> tp.List[str]:
        """Returns model's ``festures_in``"""

    @property
    def target(self) -> str:
        """Returns model's ``target``"""

    @staticmethod
    def create_model_instance(*args: tp.Any, **kwargs: tp.Any) -> Estimator:
        """
        Creates model instance to be wrapped with ModelBase
        """

    def create(self) -> SupportsModel:
        """
        Create `ModelBase` instance from a model produced by `create_model_instance`,
        and features and target taken from TagDict.
        """
