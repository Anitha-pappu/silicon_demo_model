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
from warnings import warn

import numpy as np
from pygam import LinearGAM
from pygam.terms import Term
from pygam.utils import TablePrinter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from typing_extensions import Self

from modeling.types import Matrix, Vector


class SklearnCompatibleLinearGAM(BaseEstimator, RegressorMixin):
    """A scikit-learn compatible wrapper for pyGAM's LinearGAM class.

    This class allows you to use LinearGAM with scikit-learn's tools such as
    GridSearchCV, cross_val_score, etc.

    Attributes:
        terms: The terms to use in the model.
         Defaults to 'auto'.
        max_iter: The maximum number of iterations for the solver.
            Defaults to 100.
        tol: The tolerance for the solver. Defaults to 0.0001.

    """
    def __init__(
        self,
        terms: Term = 'auto',
        max_iter: int = 100,
        tol: float = 1e-4,
    ):
        """Initializes the SklearnCompatibleLinearGAM.

        Args:
            terms: The terms to use in the model.
             Defaults to 'auto'.
            max_iter: The maximum number of iterations for the solver.
                Defaults to 100.
            tol: The tolerance for the solver. Defaults to 0.0001.
        """
        self.terms = terms
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: Matrix,  # noqa: WPS111, N803
        y: Vector,  # noqa: WPS111
    ) -> Self:
        """Fits the model to the data.

        Args:
            X: The feature matrix. The name is to match with the Sklearn requirement
            y: The target variable. The name is to match with the Sklearn requirement

        Returns:
            SklearnCompatibleLinearGAM: The fitted model.
        """
        data, target_vector = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            force_all_finite='allow-nan',
        )

        target_vector = _check_shape_and_ravel_if_needed(target_vector)
        self.model_ = LinearGAM(
            terms=self.terms,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        self.model_.fit(data, target_vector)
        # sklearn estimator requires to have attributes "n_iter_"
        self.n_iter_ = self.max_iter
        self.n_features_in_ = data.shape[1]

        return self

    def predict(self, X: Matrix) -> Vector:  # noqa: WPS111, N803
        """Makes predictions using the fitted model.

        Args:
            X: The feature matrix.  # noqa: WPS111

        Returns:
            np.ndarray: The predicted values.
        """
        check_is_fitted(self)
        data = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            force_all_finite='allow-nan',
        )
        return self.model_.predict(data)

    def partial_dependence(
        self,
        data: Matrix,
        feature_idx: int,
        **kwargs: tp.Dict[tp.Any, tp.Any],
    ) -> Vector:
        """Computes the partial dependence for a feature.

        Args:
            data: The feature matrix.
            feature_idx: The index of the feature.

        Returns:
            np.ndarray: The partial dependence for the feature.
        """
        check_is_fitted(self)
        data = check_array(
            data,
            accept_sparse=False,
            dtype=np.float64,
            force_all_finite='allow-nan',
        )
        return self.model_.partial_dependence(
            term=feature_idx,
            X=data,
            **kwargs,
        )

    def summary(self) -> TablePrinter:
        """
        Generates summary of model results via native
        ``TablePrinter`` machinery.
        """
        check_is_fitted(self)
        return self.model_.summary()


def _check_shape_and_ravel_if_needed(
    target_vector: Vector,
) -> Vector:
    """Check if target vector is in 1d array.

    Args:
        target_vector: expected to be a 1d array

    Returns:
        modified target vector if not a 1d array.
    """

    if target_vector.ndim == 2 and target_vector.shape[1] == 1:
        warn(
            "A column-vector y was passed when a 1d array was expected. \
            Please change the shape of y to (n_samples, ), \
                for example using ravel().", DataConversionWarning,
        )
        target_vector = np.ravel(target_vector)

    return target_vector
