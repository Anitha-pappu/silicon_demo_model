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


class TransformerProtocol(tp.Protocol):
    """A protocol for transformer classes that implement a `fit()`
    and `transform()` method.

    Requires the following methods:

        - `fit(x: pd.DataFrame, y: Optional[pd.DataFrame] = None,
          **fit_params: Dict) -> T`
          Trains the transformer on training data and returns the trained transformer.
          Returns an instance of the same class.

        - `transform(x: pd.DataFrame) -> pd.DataFrame`
          Applies the learned transformation to the input data and returns
          the transformed data.

    Args:
        TTransformer: A type variable that represents the class
        implementing the protocol.

    Returns:
        An instance of the protocol.

    Raises:
        TypeError: If the class does not implement the required methods.
    """
    def fit(
        self,
        x: pd.DataFrame,  # noqa: WPS111
        y: tp.Optional[pd.DataFrame] = None,  # noqa: WPS111
        **fit_params: tp.Any,
    ) -> "TransformerProtocol":
        """Trains the transformer on `x` and `y`.

        Args:
            x: The training input data.
            y: An optional DataFrame of training target values.
            **fit_params: Optional additional keyword arguments to pass to
                          the transformer's `fit()` method.

        Returns:
            The trained transformer.

        Raises:
            TypeError: If `x` is not a pandas DataFrame.
                      If `y` is not `None` and is not a pandas DataFrame.
        """

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:  # noqa: WPS111
        """Applies the learned transformation to `x`.

        Args:
            x: The input data to transform.

        Returns:
            A new DataFrame containing the transformed data.

        Raises:
            TypeError: If `x` is not a pandas DataFrame.
        """

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:  # noqa: WPS111
        """Predict using the model.

        Args:
            x: The input data to predict.

        Returns:
            A new DataFrame containing predicted values.

        Raises:
            TypeError: If `x` is not a pandas DataFrame.
        """
