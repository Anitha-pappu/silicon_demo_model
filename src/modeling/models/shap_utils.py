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
import shap
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from .. import api

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = type(None)  # noqa: WPS440. To avoid ``catboost`` in reqs

ESTIMATORS_REQUIRING_PERMUTATION = (
    SVR,
    KNeighborsRegressor,
    KernelRidge,
    AdaBoostRegressor,
    BaggingRegressor,
    MLPRegressor,
    GaussianProcessRegressor,
    RANSACRegressor,
)

ESTIMATORS_REQUIRING_TREE_EXPLAINER = (CatBoostRegressor,)

logger = logging.getLogger(__name__)

TExplainEstimator = tp.Union[api.Estimator, tp.Callable[..., tp.Any]]


def explain_sklearn_estimator_robustly(
    data: pd.DataFrame,
    estimator: api.Estimator,
    features: tp.List[str],
    algorithm: tp.Optional[str],
    **kwargs: tp.Any,
) -> shap.Explanation:
    """ Prevents errors with the explainers.

    Notes:
        Useful for default methods, when it is desirable to produce an output in
        a robust way.
        This models behaves according to the following in order to avoid (most) errors:

        - disable the additivity check on Tree models (caused errors on
          ``RandomForestRegressor``).
          More info `in this StackOverflow post <https://stackoverflow.com/questions/
          68233466/shap-exception-additivity-check-failed-in-treeexplainer>`_.
        - for the models that are otherwise known not work with the "normal" way of
          using shap:

          - uses the ``estimator.predict`` method
          - raises a warning in case the algorithm is not ``"permutation"``
          - these models are those in the module variable
            ``KNOWN_PROBLEMATIC_ESTIMATORS``


        If ``algorithm`` is ``None``, chooses a robust one:

        - Uses ``"permutation"`` for problematic estimators that require using it
        - otherwise chooses ``"auto"``

        Intended to be called with ``self._features_in`` within ``SklearnModel``
    """
    # TODO: Think about making the .predict take a np.ndarray as input as long as it
    #  has the right size (assuming the columns are the features_in)
    data = data[features]
    if _check_requires_permutation(estimator):
        estimator_to_explain: TExplainEstimator = estimator.predict
        if algorithm is None:
            algorithm = "permutation"
        if algorithm != "permutation":
            logger.warning(
                "Using algorithm different from \"permutation\" with known problematic "
                "estimator.\nConsider using 'algorithm=\"permutation\"' if there are "
                "issues.",
            )
        explainer_class = shap.Explainer
    elif _check_requires_tree_explainer(estimator):
        estimator_to_explain = estimator
        explainer_class = shap.TreeExplainer
    else:
        estimator_to_explain = estimator
        if algorithm is None:
            algorithm = "auto"
        explainer_class = shap.Explainer

    explainer = explainer_class(
        model=estimator_to_explain,
        masker=data,
        algorithm=algorithm,
        **kwargs,
    )
    logger.info(f"Using `{explainer.__class__}` to extract SHAP values...")
    if isinstance(explainer, shap.explainers.Tree):
        logger.info("Using shap.explainers.Tree with disabled `check_additivity`")
        return explainer(data, check_additivity=False)
    return explainer(data)


def _check_requires_permutation(estimator: api.Estimator) -> bool:
    """Check if an estimator requires ``permutation`` as shap algorithm."""
    return isinstance(estimator, ESTIMATORS_REQUIRING_PERMUTATION)


def _check_requires_tree_explainer(estimator: api.Estimator) -> bool:
    """Check if an estimator requires shap explainer to be built
    via ``TreeExplainer`` constructor."""
    return isinstance(estimator, ESTIMATORS_REQUIRING_TREE_EXPLAINER)
