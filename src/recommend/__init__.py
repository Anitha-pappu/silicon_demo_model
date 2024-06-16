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

from . import common_constraints, cra_export, datasets, functional, report
from .controlled_parameters import (
    ControlledParameter,
    ControlledParametersConfig,
    get_notnull_controls,
)
from .implementation_tracker import (
    calculate_implementation_status,
    collect_recs_vs_actual_implementation,
)
from .optimization_explainer import (
    create_grid_for_optimizable_parameter,
    create_optimization_explainer_plot,
)
from .optimize import optimize
from .problem_factory import ObjectiveFunction, ProblemFactoryBase
from .solution import Solution, Solutions
from .solver_factory import SolverFactory
from .uplift import (
    BaselineUplifts,
    check_uplift_model_error_stat_significance,
    check_uplift_stat_significance,
    get_impact_estimation,
    get_value_after_recs_counterfactual,
    get_value_after_recs_impact,
)

__version__ = "0.40.0"
