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

from ._factory import SupportsModelFactory  # noqa: WPS436
from ._model import SupportsModel  # noqa: WPS436


@tp.runtime_checkable
class SupportsModelTuner(tp.Protocol):
    """
    Any object inherited from ``modeling.ModelTunerBase`` satisfies this protocol.

    Tuner's main method is ``tune()`` that produces a tuned model
    i.e., an object that ``SupportsModel``.
    """

    def __init__(
        self,
        model_factory: SupportsModelFactory,
        model_tuner_config: tp.Dict[str, tp.Any],
    ) -> None:
        """Inits tuner instance with the factory and tuning config"""

    @property
    def model_tuner_config(self) -> tp.Dict[str, tp.Any]:
        """Returns tuner's config"""

    @property
    def hyperparameters_config(self) -> tp.Optional[tp.Dict[str, tp.Any]]:
        """Returns hyper-parameters' config"""

    def tune(
        self,
        data: pd.DataFrame,
        hyperparameters_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
        **tuner_fit_kwargs: tp.Any,
    ) -> SupportsModel:
        """
        Tune hyperparameters and return
        `ModelBase` instance with tuned hyperparameters.
        """
