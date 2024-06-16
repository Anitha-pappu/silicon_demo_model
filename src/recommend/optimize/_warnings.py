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

from recommend.domain_generator import OutOfDomainWarning
from recommend.warnings_catcher import (
    TWarningsLevelBasic,
    TWarningsLevelExtended,
    WarningCatcherBase,
    log_warnings,
)

_TWarningsList = tp.List[warnings.WarningMessage]


class OutOfDomainCatcher(WarningCatcherBase):
    def __init__(
        self,
        details_level: TWarningsLevelExtended,
        total_rows: tp.Optional[int] = None,
    ) -> None:
        """
        Aggregates warnings happening during the optimization

        Args:
            details_level: detail level of warnings;
                * none – hides ``OutOfDomainWarning`` warnings
                * detailed – shows all ``OutOfDomainWarning``
                    warnings without aggregation
                * row_aggregated – aggregates ``OutOfDomainWarning``
                    warnings per each row: number of tags
                    with ``OutOfDomainWarning`` warning
                * aggregated – aggregates ``OutOfDomainWarning`` warnings
                    into one single one: number of rows
                    with ``OutOfDomainWarning`` warning
            total_rows: number of rows in optimization;
                is used when ``details_level=="aggregated"`` to show
                the total number of rows
        """
        self._details_level = details_level
        self._total_rows = str(total_rows) if total_rows is not None else "??"

    @property
    def _warning_type(self) -> tp.Type[Warning]:
        return OutOfDomainWarning

    def _process_warnings(
        self, caught_warnings: _TWarningsList,
    ) -> _TWarningsList:
        if self._details_level == "none":
            return []
        if self._details_level in {"detailed", "row_aggregated"}:
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
    out_of_domain_warnings: _TWarningsList,
    total_rows: str,
) -> _TWarningsList:
    """
    Assumes warnings are aggregated per row
    """
    n_out_of_domain = len(out_of_domain_warnings)
    if not n_out_of_domain:
        return []
    aggregated_warning = warnings.WarningMessage(
        f"Found [{n_out_of_domain}/{total_rows}] rows with controls "
        f"that were out of bounds. "
        f"Proceeded as if the current value was equal "
        f"to closest min/max operating limit.",
        category=OutOfDomainWarning,
        filename=__file__,
        lineno=0,
    )
    return [aggregated_warning]


def get_per_artefact_warning_level(
    details_level: TWarningsLevelExtended,
) -> TWarningsLevelBasic:
    return tp.cast(
        TWarningsLevelBasic,
        "aggregated"
        if details_level in {"row_aggregated", "aggregated"}
        else details_level,
    )
