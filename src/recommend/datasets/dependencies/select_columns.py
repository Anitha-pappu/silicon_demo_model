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

import re
import typing as tp

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.exceptions import NotFittedError

from .transformer import Transformer


class SelectColumns(Transformer):
    """
    Selects columns from the dataset, with most of the
    functionality provided by ``pandas.DataFrame.filter``.

    Args:
        items: a list of columns to select, see
            ``pandas.DataFrame.filter`` for more details
        regex: a regex used to select columns, see
            ``pandas.DataFrame.filter`` for more details
    """

    def __init__(  # noqa: WPS234
        self,
        items: tp.Optional[tp.Union[tp.List[str], str]] = None,  # noqa: WPS110
        regex: tp.Optional[str] = None,
    ) -> None:
        # items and regex are mutually exclusive as enforced by pandas
        if items is None and regex is None:
            raise ValueError("Must state either items or regex.")
        if items is not None and regex is not None:
            raise ValueError("Must state either items or regex but not both.")

        if isinstance(items, str):
            items = [items]  # noqa: WPS110

        self.regex = re.compile(regex) if regex else None
        self.items = items  # noqa: WPS110
        self.selected_columns: tp.Optional[tp.List[str]] = None

    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: tp.Union[npt.NDArray[np.generic], pd.Series] = None,  # noqa: WPS111
        **fit_params: tp.Any,
    ) -> Transformer:
        """
        Stores a list of selected columns.

        Args:
            x: training data
            y: training y (no effect)

        Returns:
            self
        """
        self.check_x(x)
        self.selected_columns = x.filter(
            items=self.items, regex=self.regex,
        ).columns.tolist()
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:  # noqa: WPS111
        """
        Reduces x to the columns learned in the .fit step.

        Args:
            x: dataframe
        """
        self.check_x(x)
        if self.selected_columns is None:
            raise NotFittedError(".transform called before .fit.")
        x = x.filter(items=self.selected_columns)  # noqa: WPS111
        return x.reindex(columns=self.selected_columns)
