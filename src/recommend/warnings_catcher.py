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

from __future__ import annotations

import logging
import typing as tp
import warnings
from abc import abstractmethod
from functools import cached_property

logger = logging.getLogger(__name__)

_TWarningsList = tp.List[warnings.WarningMessage]

TWarningsLevelExtended = tp.Literal["none", "row_aggregated", "aggregated", "detailed"]
TWarningsLevelBasic = tp.Literal["none", "aggregated", "detailed"]


def log_warnings(caught_warnings: _TWarningsList) -> None:
    for warning in caught_warnings:
        logger.warning(warning.message)


def show_warnings(
    warnings_to_show: tp.Iterable[warnings.WarningMessage],
) -> None:
    for warning_message in warnings_to_show:
        warnings.showwarning(
            warning_message.message,
            warning_message.category,
            warning_message.filename,
            warning_message.lineno,
            warning_message.file,
            warning_message.line,
        )


class WarningCatcherBase(tp.ContextManager[None]):
    """
    The base class that implements a wrapper pattern to wrap
    ``recommend.warnings_catcher.WarningsCatcher``.

    Catches only ``recommend.domain_generator.OutOfDomainWarning`` then aggregates them
    as abstract ``_process_caught_warnings`` specifies and handles as
    ``_handle_processed_warnings`` specifies.
    """
    def __enter__(self) -> None:
        self._warnings_catcher.__enter__()

    def __exit__(self, *exec_info: tp.Any) -> None:
        self._warnings_catcher.__exit__(*exec_info)

    @cached_property
    def _warnings_catcher(self) -> _WarningsCatcher:
        return _WarningsCatcher(
            self._warning_type,
            self._process_warnings,
            self._handle_processed_warnings,
        )

    @property
    @abstractmethod
    def _warning_type(self) -> tp.Type[Warning]:
        """Implement this method to define warning category to catch"""

    @abstractmethod
    def _process_warnings(
        self, caught_warnings: _TWarningsList,
    ) -> _TWarningsList:
        """
        Implement this method to define how caught ``_warning_type``
        warnings are processed
        """

    @abstractmethod
    def _handle_processed_warnings(self, caught_warnings: _TWarningsList) -> None:
        """
        Implement this method to define how processed ``_warning_type``
        warnings are handled
        """


class _WarningsCatcher(tp.ContextManager[None]):
    def __init__(
        self,
        warning_type: tp.Type[Warning],
        warnings_processor: tp.Callable[[_TWarningsList], _TWarningsList],
        processed_warnings_handler: tp.Callable[[_TWarningsList], None],
    ) -> None:
        """
        Context manager
        that catches ``warning_type`` messages and applies
        ``warnings_processor`` to them.
        The rest of the warnings are raised as usual.

        Args:
            warning_type: all warnings of this type will be caught;
                the rest of the types are just passed through
            warnings_processor: applies transformation to caught warnings
            processed_warnings_handler: is used to handle processed warnings
        """
        self._warning_type = warning_type
        self._warnings_processor = warnings_processor
        self._processed_warnings_handler = processed_warnings_handler

    def __enter__(self) -> None:
        self._warning_catcher = warnings.catch_warnings(record=True)
        self._caught_warnings = self._warning_catcher.__enter__()

    def __exit__(self, *exec_info: tp.Any) -> None:
        self._warning_catcher.__exit__(None, None, None)
        matching_warnings = [
            warning
            for warning in self._caught_warnings
            if warning.category is self._warning_type
        ]
        pass_through_warnings = [  # list comp since we want to preserve order
            warning
            for warning in self._caught_warnings
            if warning not in set(matching_warnings)
        ]
        show_warnings(pass_through_warnings)
        aggregated_warnings = self._warnings_processor(matching_warnings)
        self._processed_warnings_handler(aggregated_warnings)
