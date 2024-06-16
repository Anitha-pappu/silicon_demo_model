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

import warnings

from kedro.pipeline import Pipeline, node

from optimus_core.utils import partial_wrapper
from recommend import ControlledParametersConfig, optimize
from recommend.functional import create_problem_factory, create_solver_factory
from recommend.report import get_solutions_overview

from ..problem_factory import SilicaProblemFactory

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_recommend_steps() -> Pipeline:
    return Pipeline(
        [
            node(
                ControlledParametersConfig,
                inputs="params:recommend.controlled_parameters",
                outputs="controlled_parameters_config",
                name="load_create_problem_factory",
            ),
            node(
                partial_wrapper(
                    create_problem_factory,
                    factory_spec={"type": SilicaProblemFactory},
                ),
                inputs={
                    "controlled_parameters_config": "controlled_parameters_config",
                    "problem_spec": "params:recommend.problem",
                    "silica_conc_model": "trained_model",
                },
                outputs="problem_factory",
                name="create_problem_factory",
            ),
            node(
                create_solver_factory,
                inputs={
                    "controlled_parameters_config": "controlled_parameters_config",
                    "solver_spec": "params:recommend.solver",
                    "stopper_spec": "params:recommend.stopper",
                },
                outputs="solver_factory",
                name="create_solver_factory",
            ),
            node(
                optimize,
                inputs={
                    "problem_factory": "problem_factory",
                    "solver_factory": "solver_factory",
                    "data_to_optimize": "test_data",
                    "n_jobs": "params:recommend.optimize.n_jobs",
                },
                outputs="solutions",
                name="optimize",
            ),
            node(
                lambda solutions: solutions.to_frame(),
                inputs={"solutions": "solutions"},
                outputs="recommend_results",
                name="convert_solutions_to_frame",
            ),
        ],
    ).tag("recommend")


def get_recommend_report_steps() -> Pipeline:
    """
    Generate historical counterfactual recommendation report
    """
    return Pipeline(
        [
            node(
                get_solutions_overview,
                inputs={
                    "solutions": "solutions",
                    "actual_target_column": "params:recommend.actual_target_column",
                    "objective_units": "params:recommend.objective_units",
                    "controls_config": "params:recommend.controlled_parameters",
                    "reference_data": "train_data",
                },
                outputs="solutions_overview",
                name="get_solutions_overview",
            ),
        ],
    ).tag("recommend_report")
