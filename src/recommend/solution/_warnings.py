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
import warnings

from recommend.warnings_catcher import (
    TWarningsLevelBasic,
    WarningCatcherBase,
    log_warnings,
)

_TWarningsList = tp.List[warnings.WarningMessage]


class NotImprovedSolutionWarning(UserWarning):
    """Is thrown when a proposed solution doesn't improve the objectve"""


class NotImprovedSolutionCatcher(WarningCatcherBase):
    def __init__(
        self,
        details_level: TWarningsLevelBasic,
        total_rows: tp.Optional[int] = None,
    ) -> None:
        """
        Aggregates ``NotImprovedSolutionWarning``
        warnings happening during the ``Solutions.to_frame()``

        Args:
            details_level: detail level of warnings;
                * none – hides ``NotImprovedSolutionWarning`` warnings
                * detailed – shows all ``NotImprovedSolutionWarning``
                    warnings without aggregation
                * aggregated – aggregates ``NotImprovedSolutionWarning`` warnings
                    into one single one: number of rows
                    with ``NotImprovedSolutionWarning`` warning
        """
        self._details_level = details_level
        self._total_rows = str(total_rows) if total_rows is not None else "??"

    @property
    def _warning_type(self) -> tp.Type[Warning]:
        return NotImprovedSolutionWarning

    def _process_warnings(
        self, caught_warnings: _TWarningsList,
    ) -> _TWarningsList:
        if self._details_level == "none":
            return []
        if self._details_level == "detailed":
            return caught_warnings
        elif self._details_level == "aggregated":
            return _aggregate_warnings(caught_warnings, self._total_rows)
        raise ValueError(f"Unknown {self._details_level = }")

    def _handle_processed_warnings(self, caught_warnings: _TWarningsList) -> None:
        """
        Logs warnings instead of re-throwing them since this they're intended
        final user-facing output that one should be just aware of instead of fixing
        """
        log_warnings(caught_warnings)


def _aggregate_warnings(
    not_improved_warnings: _TWarningsList,
    total_rows: str,
) -> _TWarningsList:
    n_affected_rows = len(not_improved_warnings)
    if not n_affected_rows:
        return []
    aggregated_warning = warnings.WarningMessage(
        f"Found [{n_affected_rows}/{total_rows}] rows with "
        f"objective not improved (same or worse). "
        f"Initial controls are returned.",
        category=NotImprovedSolutionWarning,
        filename=__file__,
        lineno=0,
    )
    return [aggregated_warning]
