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
from recommend.types import TIndexDType
from recommend.warnings_catcher import (
    TWarningsLevelBasic,
    WarningCatcherBase,
    show_warnings,
)

_TWarningsList = tp.List[warnings.WarningMessage]


class OutOfDomainCatcher(WarningCatcherBase):
    def __init__(
        self,
        details_level: TWarningsLevelBasic,
        row_index: TIndexDType,
        n_controls: int,
    ) -> None:
        """
        Aggregates warnings happening during the solver creation

        Args:
            details_level: detail level of warnings;
                * none – hides ``OutOfDomainWarning`` warnings
                * detailed – shows all ``OutOfDomainWarning``
                    warnings without aggregation
                * aggregated – aggregates ``OutOfDomainWarning``
                    warnings into a single message: number of tags
                    with ``OutOfDomainWarning`` warning
            row_index: used to enrich warning message: which row was affected;
                used when ``details_level`` is either ``aggregated`` or ``detailed``
        """
        self._details_level = details_level
        self._row_index = row_index
        self._n_controls = n_controls

    @property
    def _warning_type(self) -> tp.Type[Warning]:
        return OutOfDomainWarning

    def _process_warnings(
        self, caught_warnings: _TWarningsList,
    ) -> _TWarningsList:
        if self._details_level == "none":
            return []
        if self._details_level == "detailed":
            return _decorate_warnings_messages(
                caught_warnings, prefix=f"Row index {self._row_index}. ",
            )
        elif self._details_level == "aggregated":
            return _aggregate_warnings(
                caught_warnings, self._row_index, self._n_controls,
            )
        raise ValueError(f"Unknown {self._details_level = }.")

    def _handle_processed_warnings(self, caught_warnings: _TWarningsList) -> None:
        """
        Throws warnings instead of logging them since.
        It is expected that higher-level code
        will catch, process and then output them in a nice way.
        """
        show_warnings(caught_warnings)


def _aggregate_warnings(
    out_of_domain_warnings: _TWarningsList,
    row_index: TIndexDType,
    n_controls: int,
) -> _TWarningsList:
    n_ood_warnings = len(out_of_domain_warnings)
    if not n_ood_warnings:
        return []
    warning = warnings.WarningMessage(
        f"Row index {row_index}. "
        f"Found [{n_ood_warnings}/{n_controls}] controls out of operating range. "
        f"Proceeded as if their current values were equal "
        f"to closest min/max operating limit.",
        category=OutOfDomainWarning,
        filename=__file__,
        lineno=0,
    )
    return [warning]


def _decorate_warnings_messages(
    out_of_domain_warnings: _TWarningsList, prefix: str,
) -> _TWarningsList:
    return [
        warnings.WarningMessage(
            f"{prefix}{warning.message}",
            warning.category,
            warning.filename,
            warning.lineno,
            warning.file,
            warning.line,
            warning.source,
        ) for warning in out_of_domain_warnings
    ]
