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

"""
Contains modeling specific charts that are used
to compose modeling performance overview by calling `get_modeling_overview`.
"""

from . import benchmark_models
from ._actual_vs_feature import (
    plot_actual_vs_predicted,
    plot_actual_vs_residuals,
)
from ._compare import (
    plot_distplot_comparison_for_numeric_features,
    plot_feature_comparison_for_train_test,
)
from ._metrics_table import create_table_train_test_metrics
from ._partial_dependence import plot_partial_dependency_for_sklearn
from ._pdp import create_pdp_grid_from_data, plot_partial_dependency_plots
from ._shap import plot_shap_dependency, plot_shap_summary
from ._validation_approach import (
    get_split_details,
    plot_consecutive_validation_periods,
    plot_validation_representation,
)