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
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from modeling import SklearnCompatibleLinearGAM

# Create a simple regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)


# Initialize the model
@pytest.fixture
def gam_model() -> SklearnCompatibleLinearGAM:
    return SklearnCompatibleLinearGAM()


def test_fit(gam_model) -> None:

    gam_model.fit(X, y)

    check_is_fitted(gam_model)


def test_sklearn_compatible_estimator(gam_model) -> None:
    check_estimator(gam_model)


def test_predict(gam_model) -> None:

    gam_model.fit(X, y)

    y_pred = gam_model.predict(X)

    assert len(y_pred) == len(y)
    assert isinstance(y_pred, np.ndarray)
    assert np.std(y_pred) > 0


def test_partial_dependence(gam_model) -> None:

    gam_model.fit(X, y)

    pd = gam_model.partial_dependence(X, feature_idx=0)

    assert len(pd) == len(X)
    assert isinstance(pd, np.ndarray)
    assert np.std(pd) > 0


def test_fit_with_null_values(gam_model) -> None:

    X_with_nulls = X.copy()
    X_with_nulls[0, 0] = np.nan

    with pytest.raises(ValueError):
        gam_model.fit(X_with_nulls, y)


def test_predict_with_null_values(gam_model) -> None:

    gam_model.fit(X, y)

    X_with_nulls = X.copy()
    X_with_nulls[0, 0] = np.nan
    with pytest.raises(ValueError):
        gam_model.predict(X_with_nulls)
