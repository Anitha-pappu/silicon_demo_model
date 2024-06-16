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

import numpy.typing as npt
import pandas as pd

_TFeatureImportanceDict = tp.Dict[str, float]


@tp.runtime_checkable
class ShapExplanation(tp.Protocol):
    """
    Type stub protocol for `shap.Explanation`
    """

    @property
    def values(self) -> npt.NDArray[tp.Any]:  # noqa: WPS110
        """
        `np.array` of SHAP values
        """

    @property
    def data(self) -> npt.NDArray[tp.Any]:
        """
        `np.array` of original data called to be explained
        """

    @property
    def base_values(self) -> npt.NDArray[tp.Any]:
        """
        `np.array` of SHAP base values – E(F(X)).
        """

    @property
    def feature_names(self) -> tp.List[str]:
        """
        `np.array` of column names from data
        """


@tp.runtime_checkable
class SupportsShapFeatureImportance(tp.Protocol):
    """
    Can be easily implemented by inheriting from
    ``modeling.ProducesShapFeatureImportance``.
    """
    def produce_shap_explanation(
        self, data: pd.DataFrame, **kwargs: tp.Any,
    ) -> ShapExplanation:
        """
        Produce an instance of shap.Explanation based on provided data.

        Args:
            data: data to calculate SHAP values
            **kwargs: additional keyword arguments that
             are required for method implementation

        Notes:
            Columns in the resulting matrix should match
             original order of columns in provided data

        Returns:
            ``shap.Explanation`` containing prediction base values and SHAP values
        """

    def get_shap_feature_importance(
        self, data: pd.DataFrame, **kwargs: tp.Any,
    ) -> _TFeatureImportanceDict:
        """
        Calculate feature importance from SHAP values.

        Notes:
            This method consecutively calls two methods:
            first `produce_shap_explanation` and then
            `get_shap_feature_importance_from_explanation`

            So if you are producing shap explanations by
            `produce_shap_explanation` (lets say, to build a shap summary plot),
            you can reduce one extra shap explanation calculation, using
            `get_shap_feature_importance_from_explanation` with a pre-calculated
            explanation instead of calling this method.

        Args:
            data: data to calculate SHAP values
            **kwargs: keyword arguments to
             pass into ``produce_shap_explanation``

        Notes:
            Columns in the resulting dict should match
             original order of columns in provided data.

        Returns:
            Mapping from feature name into numeric feature importance
        """

    @staticmethod
    def get_shap_feature_importance_from_explanation(
        explanation: ShapExplanation,
    ) -> _TFeatureImportanceDict:
        """
        Calculate feature importance from the provided SHAP explanation.
        Importance is calculated as a mean absolute shap value.

        Args:
            explanation: shap-explanation object

        Returns:
            Mapping from feature name into numeric feature importance
        """
