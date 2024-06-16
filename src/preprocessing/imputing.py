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

import numpy as np
import pandas as pd
from sklearn.impute._base import _BaseImputer  # NOQA
from sklearn.utils.validation import check_is_fitted

from preprocessing.core import TransformerProtocol
from preprocessing.tags_config import TagImputationParameters, TagsConfig
from preprocessing.utils import load_obj

logger = logging.getLogger(__name__)

TModelKwargs = tp.Dict[str, tp.Union[str, int, float]]


# TODO: Revisit imputer when refactoring usecases
class ModelBasedImputer(_BaseImputer):
    def __init__(
        self,
        model: tp.Union[TransformerProtocol, str],
        target_name: str,
        model_kwargs: tp.Optional[TModelKwargs] = None,
        copy: bool = True,
    ):
        """
        Used to predict ommited data in dataframes.

        Args:
            model: Here you should add name of your imputer class
            target_name: The name of your target function
            model_kwargs: Model arguments, for example random_state=0
            copy: Data copy
        """

        super().__init__(missing_values=np.NaN, add_indicator=False)

        self._model: tp.Optional[TransformerProtocol] = None

        if isinstance(model, str):
            self._model_type: str = model
        else:
            self._model = model

        self._model_kwargs = model_kwargs if model_kwargs else {}
        self.copy = copy
        self.features = None
        self.target_name = target_name

    def fit(self, X: pd.DataFrame) -> "ModelBasedImputer":  # noqa: WPS111,N803
        """This is "fit" method for imputer. It is to follow syntax from sklearn.

        Args:
            X: data

        Returns:
            self
        """
        x_validated = self._validate_input(X)
        self.features = x_validated.drop(self.target_name, axis=1).columns
        x_train = x_validated.dropna().reset_index(drop=True)
        y_train = x_train[self.target_name]
        if self._model is None:
            self._model = load_obj(self._model_type)(**self._model_kwargs)
        self._model.fit(x_train[self.features], y_train)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: WPS111,N803
        """This is "transform" method for imputer. It is to follow syntax from sklearn.

        Args:
            X: data

        Returns:
            self
        """
        check_is_fitted(self._model)
        x_validated = self._validate_input(X)
        target_nan_mask = x_validated[self.target_name].isnull()
        x_test = x_validated[target_nan_mask].drop(self.target_name, axis=1)
        if self._model is None:
            self._model = load_obj(self._model_type)(**self._model_kwargs)
        x_validated[self.target_name][target_nan_mask] = self._model.predict(x_test)
        return x_validated

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: WPS111,N803
        """Validates input

        Args:
            X: data

        Raises:
            ValueError: if data contains non_numeric columns
            ValueError: if data contains null values
            ValueError: if target variable is missing from data

        Returns:
            data
        """
        x_numeric = X.select_dtypes(include=np.number)
        x_non_numeric = X.drop(x_numeric, axis=1)
        if x_non_numeric.shape[1] != 0:
            raise ValueError("X must be numeric")
        if self.target_name in X.columns:
            if X.drop(self.target_name, axis=1).isnull().sum().sum():
                raise ValueError("X mustn't have NaNs")
            if len(X[self.target_name].dropna()) == len(X):
                logger.warning("No missing values to impute")
        else:
            raise ValueError("Target variable must be in X")
        if self.copy:
            return X.copy()
        return X


def transform_numeric_imputer(
    data: pd.DataFrame,
    transformer: TransformerProtocol,
    cols_list: tp.List[str],
) -> pd.DataFrame:
    """
    Used to impute missing numerical data in a given dataset.

    We assume that the transformer has
    already been fit, and thus, using the passed transformer, the data
    is transformed.

    Note:
        Only numerical columns are imputed

    Args:
        data: input data
        transformer: A Sklearn compatible transformer
        cols_list: list of numerical columns to inpute

    Returns:
        dataframe without missing numeric vals
    """
    impute_data: pd.DataFrame = data.copy()

    if transformer is not None and getattr(transformer, "fit_transform", None) is None:
        raise TypeError("Passed Transformer does not have a fit_transform method")

    impute_data.loc[:, cols_list] = transformer.transform(impute_data[cols_list])
    return impute_data


def interpolate_cols(
    data: pd.DataFrame,
    impute_config: TagsConfig[TagImputationParameters],
    **kwargs: tp.Any,
) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Used to interpolate data on columns specified in method_df DataFrame.

    Args:
        data: input data
        impute_config: DataFrame with columns 'tag_name' and 'imputation_rule'
                   which specifies the interpolation method for each tag (column name)
        **kwargs: kwargs used for pandas.Series.interpolate

    Returns:
        dataframe with interpolated data and a dataframe with interpolation info
    """

    interpolated_data = data.copy()
    missing_data_info = []

    for tag_name in impute_config.keys():
        imputation_rule = impute_config[tag_name].imputation_rule

        if tag_name not in interpolated_data.columns:
            raise ValueError(f"Tag '{tag_name}' not found in dataframe columns.")

        # Count missing data before interpolation
        missing_count = interpolated_data[tag_name].isna().sum()
        missing_percentage = (missing_count / len(interpolated_data[tag_name])) * 100

        missing_data_info.append({
            'tag_name': tag_name,
            'imputation_rule': imputation_rule,
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
        })

        # Perform the interpolation based on the specified method for this tag
        interpolated_data[tag_name] = interpolated_data[tag_name].interpolate(
            method=imputation_rule,
            **kwargs,
        )
        logger.info(
            f"Interpolating '{tag_name}' column using "
            f"'{imputation_rule}' method.",
        )

    return interpolated_data, pd.DataFrame(missing_data_info)
