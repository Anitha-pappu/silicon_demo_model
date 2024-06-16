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

from recommend.types import TIndexDType
from recommend.utils import pformat
from recommend.warnings_catcher import TWarningsLevelBasic

from ._solution import COLUMNS as EXPORT_COLUMNS  # noqa: WPS436
from ._solution import Solution  # noqa: WPS436
from ._warnings import NotImprovedSolutionCatcher  # noqa: WPS436


class Solutions(tp.Mapping[TIndexDType, Solution]):
    """
    Maps row to optimize index into its solution

    Attributes:
        export_columns: contains columns that are used in export
            produced by ``to_frame`` method
    """

    export_columns = EXPORT_COLUMNS

    def __init__(self, solutions: tp.Iterable[Solution]) -> None:
        self._solutions_map = _build_map_from_row_index_to_solution(solutions)

    def __getitem__(self, key: TIndexDType) -> Solution:
        return self._solutions_map[key]

    def __len__(self) -> int:
        return len(self._solutions_map)

    def __iter__(self) -> tp.Iterator[TIndexDType]:
        return iter(self._solutions_map)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        solution_keys = list(self._solutions_map.keys())
        solution_indices = pformat(solution_keys, indent=8)
        return (
            f"{class_name}(\n"
            f"    keys={solution_indices},\n"
            f"    values=(...),\n"
            f")"
        )

    def to_frame(
        self,
        warnings_details_level: TWarningsLevelBasic = "aggregated",
    ) -> pd.DataFrame:
        """
        Extracts comparison of initial and optimized
        optimizable parameters for each solver and problem
        using ``Solution.extract_optimization_result_from_solver``.

        Args:
            warnings_details_level: TWarningsLevelExtended = "aggregated",
        """
        n_solutions = len(self)
        with NotImprovedSolutionCatcher(warnings_details_level, n_solutions):
            solutions_in_series_form = {
                index: solution.to_series()
                for index, solution in self._solutions_map.items()
            }
        # this is better than pd.DataFrame.from records since it preserves dtypes
        return pd.concat(solutions_in_series_form, axis=1).T

    @property
    def controls(self) -> tp.List[str]:
        controls: tp.Set[str] = set()
        for solution in self.values():
            controls |= solution.control_parameters_before.keys()
        return list(controls)

    @property
    def states(self) -> tp.Iterable[str]:
        return list(next(iter(self.values())).row_to_optimize.columns)


def _build_map_from_row_index_to_solution(
    solutions: tp.Iterable[Solution],
) -> tp.Dict[TIndexDType, Solution]:
    solutions_map = {}
    duplicated_index = []
    for solution in solutions:
        solution_index = solution.row_to_optimize_index
        if solution_index in solutions_map:
            duplicated_index.append(solution_index)
        solutions_map[solution_index] = solution
    if duplicated_index:
        raise ValueError(
            f"Found duplicated indices for solutions: {duplicated_index}. "
            f"Please use unique index for rows when optimizing.",
        )
    return solutions_map
