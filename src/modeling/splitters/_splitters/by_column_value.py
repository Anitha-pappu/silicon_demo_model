# mypy: disable-error-code="valid-type"
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
import logging
import typing as tp

import pandas as pd

from .base_splitter import SplitterBase

logger = logging.getLogger(__name__)

_WARNING_MESSAGE = (
    "None of the specified test values: {values_for_test}"
    ' are present in column "{column_name}".'
    "The split is not possible."
)


class ByColumnValueSplitter(SplitterBase):
    """
    Splits data based on single column values: sends rows belonging to a specified
    collection of values to the test piece, and the rest to the train piece.

    This splitter is useful when working with panel data, e.g.:
        - In oil wells data, send a pre-defined set of wells to test dataset.
        - In retail stores data, send a pre-defined set of stores to test dataset.

    Constructor args:
        column_name: To split by.
        values_for_test: Which labels from this column to send to a test set.
    """
    def __init__(self, column_name: str, values_for_test: tp.List[str]) -> None:
        self.column_name = column_name
        self.values_for_test = values_for_test
        self._validate_init_parameters()

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}("  # noqa: WPS237
            f"column_name={repr(self.column_name)}, "
            f"values_for_test={repr(self.values_for_test)}, "
            ")"
        )

    def _validate_init_parameters(self) -> None:
        """
        Validates parameters assigned to self during initialization and raises
        errors if they are not as expected.

        Raises:
            TypeError: If ``column_name`` is not string.
            ValueError: If ``column_name`` is an empty string, or if ``values_for_test``
                is an empty collection.
        """
        if not isinstance(self.column_name, str):
            raise TypeError(
                "Column name should be a string."
                f" Got {type(self.column_name)} instead.",
            )
        elif not self.column_name:
            raise ValueError("Column name is an empty string.")
        elif not self.values_for_test:
            raise ValueError("No values for test set provided.")

    def _split(self, data: pd.DataFrame) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        test_mask = self._resolve_test_mask(data)
        train_data = data[~test_mask]
        test_data = data[test_mask]
        return train_data, test_data

    def _resolve_test_mask(self, data: pd.DataFrame) -> pd.Series:
        """
        Get a boolean mask that dictates which rows of a dataset to set to a test set.

        Args:
            data: To split.

        Returns:
            A boolean mask that dictates which rows of a dataset to set to a test set.

        Raises:
            ValueError: If mask has number of unique values different from 2.
        """
        test_mask = data[self.column_name].isin(self.values_for_test)
        if test_mask.nunique() != 2:
            message = _WARNING_MESSAGE.format(
                values_for_test=self.values_for_test,
                column_name=self.column_name,
            )
            raise ValueError(message)
        return test_mask
