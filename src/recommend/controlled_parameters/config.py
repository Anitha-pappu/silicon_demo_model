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

import typing as tp

import pandas as pd
from pydantic import TypeAdapter

from ..types import TagDictLike, TRowToOptimize
from ..utils import (
    parse_records_from_dataframe_by_schema,
    pformat,
    validate_is_a_single_row_dataframe,
)
from .controlled_parameter import ControlledParameter

_TRecognizedControlledParameters = tp.Union[
    tp.Mapping[str, tp.Any], ControlledParameter,
]
_TIterableOfControls = tp.Iterable[_TRecognizedControlledParameters]
_ElementIdAttrPair = tp.Tuple[tp.Union[str, int], tp.Union[str, int]]
_TElementIdAttrPairToMessage = tp.Dict[_ElementIdAttrPair, tp.List[str]]


class ControlledParametersConfig(tp.Mapping[str, ControlledParameter]):
    def __init__(self, config: _TIterableOfControls) -> None:
        """
        Maps data_to_optimize index values into solutions

        Raises:
            pydantic.ValidationError: if there are any issues
                with parsing input ``config``
        """
        self._config = TypeAdapter(tp.List[ControlledParameter]).validate_python(config)
        self._name_to_config = {control.name: control for control in self._config}

    def __getitem__(self, parameter_name: tp.Any) -> ControlledParameter:
        if not isinstance(parameter_name, str):
            parameter_name_type = type(parameter_name).__name__
            raise TypeError(f"Unknown key type: {parameter_name_type}")
        return self._name_to_config[parameter_name]

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self._name_to_config)

    def __len__(self) -> int:
        return len(self._name_to_config)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        keys = pformat(set(self), indent=8)
        return (
            f"{class_name}(\n"
            f"    keys={keys},\n"
            f"    values=(...),\n"
            f")"
        )

    def copy(self) -> ControlledParametersConfig:
        return ControlledParametersConfig(control.copy() for control in self._config)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        columns_to_fields_mapping: tp.Optional[tp.Mapping[str, str]] = None,
    ) -> ControlledParametersConfig:
        """
        Creates ``ControlledParametersConfig`` instance
        that contains rows with controls definitions
        from dataframe

        Args:
            df: input data frame with rows containing contorlled parameters definitions
            columns_to_fields_mapping: is used to rename df columns
                to match fields of ``TagMetaData``

        Notes:
            * ``df`` must contain the required columns
              (i.e. ``ControlledParametersConfig`` required fields);
            * optional columns (i.e. ``ControlledParametersConfig`` optional fields)
              are used if provided

        Raises:
            ValueError: if ``ControlledParameter`` required columns
                are not present in the data

        Examples:
            ``df`` is a pd.DataFrame with columns "A", "B", "C".
            Since those columns don't match fields names, we have to provide a renaming
            map from config's columns into ``ControlledParametersConfig``s fields::

            >>> df = pd.DataFrame({
            >>>     "A": ["control_1", "state_1"],
            >>>     "B": [0, np.nan],
            >>>     "C": [1, np.nan],
            >>> })
            >>> rename_map = {"A": "name", "B": "op_min", "C": "op_max"}
            >>> ControlledParametersConfig.from_dataframe(df, rename_map)

        """
        raw_config = parse_records_from_dataframe_by_schema(
            df, ControlledParameter.schema(), columns_to_fields_mapping,
        )
        return ControlledParametersConfig(raw_config)

    @classmethod
    def from_tag_dict(
        cls,
        tag_dict: TagDictLike,
        columns_to_fields_mapping: tp.Optional[tp.Mapping[str, str]] = None,
    ) -> ControlledParametersConfig:
        """
        Creates ``ControlledParametersConfig`` instance
        that contains rows with controls definitions
        from TagDict-like object. As tag dict is a wrapper around data frame,
        we are parsing it in a same to df manner but with additional tags selection.

        Args:
            tag_dict: input tag dictionary; a wrapper around data frame which rows
                contain controlled parameters definitions
            columns_to_fields_mapping: used to rename df columns
                to match fields of ``TagMetaData``

        Notes:
            * ``tag_dict`` may contain other than ``control`` type of tags;
              we select only those that have ``control`` value in ``tag_type`` column
            * ``tag_dict`` must contain the required columns
              (i.e. ``ControlledParametersConfig`` required fields);
            * optional columns (i.e. ``ControlledParametersConfig`` optional fields)
              are used if provided

        Raises:
            ValueError: if ``ControlledParameter`` required columns
                are not present in the data

        Examples:
            ``df`` is a tag dictionary around a pd.DataFrame with columns "A", "B", "C".
            Since those columns don't match fields names, we have to provide a renaming
            map from config's columns into ``ControlledParametersConfig``s fields::

            >>> td = TagDict(pd.DataFrame({
            >>>     "A": ["control_1", "state_1"],
            >>>     "B": [0, np.nan],
            >>>     "C": [1, np.nan],
            >>> }))
            >>> rename_map = {"A": "name", "B": "op_min", "C": "op_max"}
            >>> ControlledParametersConfig.from_tag_dict(td, rename_map)
        """
        df = tag_dict.to_frame()
        df = df[df["tag_type"] == "control"]
        return ControlledParametersConfig.from_dataframe(df, columns_to_fields_mapping)


def get_notnull_controls(
    row_to_optimize: TRowToOptimize,
    controlled_columns: tp.List[str],
) -> tp.List[str]:
    """
    Returns columns from ``controlled_columns``
    that have notnull values in ``row_to_optimize``
    """
    validate_is_a_single_row_dataframe(row_to_optimize)
    row_index = row_to_optimize.index[0]
    is_notnull_column = (
        row_to_optimize.loc[row_index, controlled_columns].notnull()
    )
    return list(is_notnull_column.index[is_notnull_column])
