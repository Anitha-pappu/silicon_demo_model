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

"""Class for meta information validation"""

from __future__ import annotations

import typing as tp

import pandas as pd
from pydantic import BaseModel, TypeAdapter, field_validator

from ..types import TagDictLike
from ..utils import parse_records_from_dataframe_by_schema

_OptionalRenameMap = tp.Optional[tp.Mapping[str, str]]
_MetaData = tp.TypeVar("_MetaData", bound=BaseModel)


class TagMetaData(BaseModel):
    """
    Tags metadata schema.
    """
    id: str
    tag: str
    clear_name: str
    unit: str
    area: tp.Optional[str] = None
    precision: tp.Optional[int] = 2
    priority: tp.Optional[int] = 0
    tolerance: float

    @field_validator('area', mode="before")
    def change_nan_to_none(cls, str_value: tp.Any) -> tp.Any:  # noqa: N805
        """Converts NaN to None for pandas compatibility."""
        if isinstance(str_value, float) and pd.isna(str_value):
            return None
        return str(str_value)


class TargetMetaData(BaseModel):
    """
    Targets metadata schema.
    """
    id: str
    tag: str
    name: str
    unit: str
    aggregation: str
    objective: str
    precision: tp.Optional[int] = 2


class PlantStatusData(BaseModel):
    """
    Plant status metadata schema.
    """
    id: str
    tag: str
    clear_name: str
    unit: str
    area: tp.Optional[str] = None
    precision: tp.Optional[int] = 2
    column_name: str = ""
    section: str

    @field_validator('area', mode="before")
    def change_nan_to_none(cls, str_value: tp.Any) -> tp.Any:  # noqa: N805
        """Converts NaN to None for pandas compatibility."""
        if isinstance(str_value, float) and pd.isna(str_value):
            return None
        return str(str_value)


class MetaDataConfig(tp.Iterable[_MetaData]):
    def __init__(
        self,
        meta_data: tp.Iterable[_MetaData | tp.Mapping[str, tp.Any]],
        # todo: follow https://peps.python.org/pep-0696/ and resolve in python 3.12
        schema: tp.Type[_MetaData],
    ) -> None:
        """
        Implements ``typing.Iterable[_MetaData]`` interface that stores
        tag meta information of type ``_MetaData`` defined by a ``schema`` argument.

        If you want to change the expected tag metadata format, create your own
        pydantic class from BaseModel,
        define all the executed fields and use it as an argument
        when creating the instance of this class::

            Example with custom class defined
            >>> class CustomTagMeta(BaseModel):
            ...     raw_tag: str
            ...     tag_clear: str
            ...     human_readable_tag_name: str
            >>> tag_meta = MetaDataConfig(
            ...     [{
            ...         "raw_tag": "raw tag id",
            ...         "tag_clear": "...",
            ...         "human_readable_tag_name": "...",
            ...     }],
            ...     schema=CustomTagMeta,
            ... )
            >>> list(tag_meta)[0].raw_tag
            "raw tag id"

        Args:
            meta_data: iterable of objects to parse using ``schema``
            schema: schema (pydantic class) used to parse objects
                from ``tags_meta_data``
        """
        self._meta_data = TypeAdapter(
            # mypy doesn't recognize `schema` as a correct type
            tp.List[schema],  # type: ignore
        ).validate_python(meta_data)

    def __iter__(self) -> tp.Iterator[_MetaData]:
        return iter(self._meta_data)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        # todo: follow https://peps.python.org/pep-0696/ and resolve in python 3.12
        schema: tp.Type[_MetaData],
        columns_to_fields_mapping: _OptionalRenameMap = None,
    ) -> MetaDataConfig[_MetaData]:
        """
        Creates instance of ``MetaDataConfig``
        from ``df`` argument

        Args:
            df: dataframe specifying tag meta-data;
            schema: defines ``_MetaData`` which is used for df parsing
            columns_to_fields_mapping: maps columns of ``df``
              into fields of ``_MetaData``

        Examples:
            ``raw_config`` is a pd.DataFrame with columns ["A", "B", "C"].
            Since those columns don't match fields names, we have to provide a renaming
            map from config's columns into ``_MetaData``s fields::
                >>> df = pd.DataFrame({
                ...     "A": ["control_1", "state_1"],
                ...     "B": [0, np.nan],
                ...     "C": [1, np.nan],
                ... })
                >>> rename_map = {"A": "name", "B": "op_min", "C": "op_max"}
                >>> MetaDataConfig.from_dataframe(
                ... df, columns_to_fields_mapping=rename_map
                ... )

        Returns: parsed config
        """
        raw_config = parse_records_from_dataframe_by_schema(
            df, schema.schema(), columns_to_fields_mapping,
        )
        return MetaDataConfig(raw_config, schema)

    @classmethod
    def from_tag_dict(
        cls,
        tag_dict: TagDictLike,
        # todo: follow https://peps.python.org/pep-0696/ and resolve in python 3.12
        schema: tp.Type[_MetaData],
        columns_to_fields_mapping: _OptionalRenameMap = None,
    ) -> MetaDataConfig[_MetaData]:
        """
        Creates instance of ``MetaDataConfig``
        from ``tag_dict`` argument

        Args:
            tag_dict: tag dict wrapper around the dataset specifying tag meta-data
            columns_to_fields_mapping: maps columns of ``tag_dict`` underlying dataset
                into fields of ``_MetaData``
            schema: defines ``_MetaData`` which is used for df parsing

        Examples:
            ``tag_dict`` is a pd.DataFrame with columns ["A", "B", "C"].
            Since those columns don't match fields names, we have to provide a renaming
            map from config's columns into ``_MetaData``s fields::
                >>> td = TagDict(pd.DataFrame({
                >>>     "A": ["control_1", "state_1"],
                >>>     "B": [0, np.nan],
                >>>     "C": [1, np.nan],
                >>> }))
                >>> rename_map = {"A": "name", "B": "op_min", "C": "op_max"}
                >>> MetaDataConfig.from_tag_dict(
                ... td, columns_to_fields_mapping=rename_map
                ... )

        Returns: parsed config
        """
        return MetaDataConfig.from_dataframe(
            tag_dict.to_frame(), schema, columns_to_fields_mapping,
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(...)"
        )
