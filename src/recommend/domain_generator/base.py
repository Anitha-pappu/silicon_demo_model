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
from abc import ABC, abstractmethod

from recommend.types import TRowToOptimize

from ..controlled_parameters import (
    ControlledParameter,
    ControlledParametersConfig,
)
from ..utils import pformat, validate_is_a_single_row_dataframe

TDomain = tp.Any


class BaseDomainGenerator(ABC):
    """Abstract class for generating domain space."""

    def __init__(self, controlled_parameters: tp.Any, **kwargs: tp.Any) -> None:
        if not isinstance(controlled_parameters, ControlledParametersConfig):
            controls_type = type(controlled_parameters).__class__.__name__
            raise TypeError(
                f"Wrong type provided for controlled_parameters: {controls_type}. "
                f"Expected `ControlledParametersConfig`",
            )
        self._controlled_parameters = controlled_parameters

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        controlled_parameters_repr = pformat(self._controlled_parameters, indent=4)
        return (
            f"{class_name}(\n"
            f"    controlled_parameters={controlled_parameters_repr},\n"
            f")"
        )

    def generate(
        self,
        row_to_optimize: TRowToOptimize,
        active_controlled_parameters: tp.List[str],
    ) -> tp.List[TDomain]:
        """
        Generates domain space for currently active-controlled columns
        ``active_controlled_parameters`` of the input row ``row_to_optimize``.

        IMPORTANT:
        The order of ``active_controlled_parameters`` must match the order of
        controlled variables defined in a problem
        (i.e. ``StatefulOptimizationProblem.optimizable_columns``).

        Args:
            row_to_optimize: optimized row for controlled columns of which
                domain is constructed; used for discovering initial value for each of
                active controlled parameters
            active_controlled_parameters: parameters to create domain for
        """
        validate_is_a_single_row_dataframe(row_to_optimize)
        self._validate_active_controls(active_controlled_parameters)

        domain_space = []
        for parameter in active_controlled_parameters:
            single_parameter_domain: TDomain = self.generate_for_single_parameter(
                row_to_optimize, self._controlled_parameters[parameter],
            )
            domain_space.append(single_parameter_domain)
        return domain_space

    @staticmethod
    @abstractmethod
    def generate_for_single_parameter(
        row_to_optimize: TRowToOptimize,
        control_config: ControlledParameter,
    ) -> TDomain:
        """
        Generates domain for single parameter based its config
        ``controlled_parameter_config`` and ``row_to_optimize``.
        """

    def _validate_active_controls(
        self, active_controls: tp.List[str],
    ) -> None:
        """
        Checks if provided active controls are defined in ``self.controlled_parameters``
        """
        missing_controls_configs = (
            set(active_controls).difference(self._controlled_parameters)
        )
        if missing_controls_configs:
            raise ValueError(
                f"Found some unknown active controls: {missing_controls_configs}",
            )
