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
This is a SCRIPT for updating some datasets' artefacts.
Use it when the structure of the module changes and pickles stop working.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from optimizer import StatefulOptimizationProblem
from optimizer.loggers import BasicLogger, BestTrajectoryLogger, PenaltyLogger
from optimizer.solvers import DifferentialEvolutionSolver
from optimizer.stoppers import NoImprovementStopper
from recommend import (
    ControlledParametersConfig,
    Solutions,
    SolverFactory,
    cra_export,
    datasets,
    optimize,
)
from recommend.datasets._io_utils import (  # noqa: WPS436
    dump_csv_data,
    dump_json_data,
    dump_pickle_data,
)
from recommend.datasets.dependencies import ProblemFactory, SelectColumns

_DEFAULT_RANDOM_SEED = 123
_DEPTH = 15
_MIN_IMP_TRACKING_INCREASE_PC = -0.2
_MAX_IMP_TRACKING_INCREASE_PC = 1.2
_TEST_SIZE = 0.4
_BASELINE_TEST_SIZE = 0.4
_MIN_UPLIFT = -2.2
_MAX_UPLIFT = 0.5
_BASELINE_MAX_ITER = 1000
_BASELINE_ALPHA = 0.1
_BASELINE_L1_RATIO = 0.01
_REC_PERFORMANCE_PCT_RIGHT = 0.2
_IMP_STATUS_MEAN = 1.0
_IMP_STATUS_STD = 0.5
_IMP_STATUS_MIN = 0.8
_IMP_STATUS_MAX = 1.2


def main() -> None:  # noqa: WPS210
    data = datasets.get_sample_recommend_input_data()
    # create data clustering based on target; required to showcase reporting
    data["silica_conc_cluster"] = pd.cut(data["silica_conc"], 6)

    train_data, test_data = train_test_split(
        data,
        random_state=_DEFAULT_RANDOM_SEED,
        shuffle=False,
        test_size=_TEST_SIZE,
    )
    silica_model = _create_model(train_data)
    _create_solutions(silica_model, test_data)
    solutions = datasets.get_sample_solutions()
    _create_sample_runs_cra(solutions)
    controlled_params_config = datasets.get_sample_controlled_parameters_config()
    _create_sample_recs_cra(solutions, controlled_params_config)
    _create_sample_implementation_tracking_data(
        solutions,
        controlled_params_config,
    )
    _create_sample_states_cra(solutions)
    baseline_train_data, baseline_test_data = train_test_split(
        train_data,
        random_state=_DEFAULT_RANDOM_SEED,
        shuffle=False,
        test_size=_BASELINE_TEST_SIZE,
    )
    baseline_model = _create_sample_baseline_model(baseline_train_data)
    _create_sample_baseline_model_errors(baseline_test_data, baseline_model)
    _create_sample_actual_values_after_recs(test_data, baseline_model)
    _create_recs_performance(test_data, solutions)


def _create_model(train_data: pd.DataFrame) -> Pipeline:
    """Creates a sample trained model"""
    features = [
        "iron_feed",
        "silica_feed",
        "starch_flow",
        "amina_flow",
        "ore_pulp_flow",
        "ore_pulp_ph",
        "ore_pulp_density",
        "total_air_flow",
        "total_column_level",
        "feed_diff_divide_silica",
    ]
    model = make_pipeline(
        SelectColumns(items=features),
        KNNImputer(),
        RandomForestRegressor(max_depth=_DEPTH, random_state=_DEFAULT_RANDOM_SEED),
    )
    model.fit(train_data[features], train_data["silica_conc"])
    dump_pickle_data(model, file_name="sample_trained_model")
    return model


def _create_solutions(silica_model: Pipeline, data_to_optimize: pd.DataFrame) -> None:
    """Creates a sample Solutions object with optimization details"""
    sense = "minimize"
    controlled_parameters_config = datasets.get_sample_controlled_parameters_config()
    problem_factory = ProblemFactory(
        controlled_parameters_config,
        StatefulOptimizationProblem,
        {"sense": sense},
        model_registry={"silica_conc_predictor": silica_model},
    )
    solver_factory = SolverFactory(
        controlled_parameters_config,
        DifferentialEvolutionSolver,
        {
            "sense": sense,
            "seed": 0,
            "maxiter": 100,
            "mutation": [0.5, 1.0],
            "recombination": 0.7,
            "strategy": "best1bin",
        },
        NoImprovementStopper,
        {'patience': 100, 'min_delta': 0.1},
    )
    solutions = optimize(
        data_to_optimize,
        problem_factory,
        solver_factory,
        loggers=[BestTrajectoryLogger(), PenaltyLogger(), BasicLogger()],
        n_jobs=-1,
    )
    dump_pickle_data(solutions, "sample_solutions")


def _create_sample_runs_cra(solutions: Solutions) -> None:
    """
    Creates sample run information to export to CRA. Most recent run is removed as this
    dataset is used in the implementation tracking pipeline and no data would be
    available to assess its implementation status
    """
    sample_runs_cra = cra_export.prepare_runs(solutions)
    last_run_id = pd.DataFrame(sample_runs_cra).sort_values("timestamp").iloc[-1]["id"]
    sample_runs_cra = [run for run in sample_runs_cra if run["id"] != last_run_id]
    dump_json_data(sample_runs_cra, "sample_runs_cra")


def _create_sample_recs_cra(
    solutions: Solutions,
    controlled_parameters_config: ControlledParametersConfig,
) -> None:
    """
    Creates sample recommendations to export to CRA. Most recent run is removed as this
    dataset is used in the implementation tracking pipeline and no data would be
    available to assess its implementation status
    """
    tag_meta = datasets.get_sample_tags_meta()
    target_meta = datasets.get_sample_targets_meta()
    runs_cra = cra_export.prepare_runs(solutions)
    last_run_id = pd.DataFrame(runs_cra).sort_values("timestamp").iloc[-1]["id"]
    sample_recs_cra = cra_export.prepare_recommendations(
        solutions,
        controlled_parameters_config,
        tag_meta,
        target_meta,
        target_name="silica_conc",
    )
    sample_recs_cra = [run for run in sample_recs_cra if run["run_id"] != last_run_id]
    dump_json_data(sample_recs_cra, "sample_recommendations_cra")


def _create_sample_states_cra(solutions: Solutions) -> None:
    """
    Creates sample states to export to CRA. Most recent run is removed as this
    dataset is used in the implementation tracking pipeline and no data would be
    available to assess its implementation status
    """
    tag_meta = datasets.get_sample_tags_meta()
    runs_cra = cra_export.prepare_runs(solutions)
    last_run_id = pd.DataFrame(runs_cra).sort_values("timestamp").iloc[-1]["id"]
    sample_states_cra = cra_export.prepare_states(solutions, tag_meta)
    sample_states_cra = [
        run for run in sample_states_cra if run["run_id"] != last_run_id
    ]
    dump_json_data(sample_states_cra, "sample_states_cra")


def _create_sample_implementation_tracking_data(
    solutions: Solutions,
    controlled_parameters_config: ControlledParametersConfig,
) -> None:
    """
    Creates sample data with values for control parameters to assess implementation
    status.

    The value for a control in the next timestamp available is calculated by adding a
    percentage between  ``_MIN_IMP_TRACKING_INCREASE_PC`` and
    ``_MAX_IMP_TRACKING_INCREASE_PC`` of the recommended sensor change to the current
    sensor value.
    """
    tag_meta = datasets.get_sample_tags_meta()

    runs_cra = pd.DataFrame(cra_export.prepare_runs(solutions))
    recs_cra = (
        pd.DataFrame(
            cra_export.prepare_recommendations(
                solutions,
                controlled_parameters_config,
                tag_meta,
                datasets.get_sample_targets_meta(),
                target_name="silica_conc",
            ),
        )
        .rename({"value": "recommended_value"}, axis=1)
    )
    states_cra = (
        pd.DataFrame(cra_export.prepare_states(solutions, tag_meta))
        .rename({"value": "current_value"}, axis=1)
    )
    imp_status_input = (
        recs_cra
        .merge(states_cra, on=["run_id", "tag_id"])
        .merge(runs_cra, left_on="run_id", right_on="id")
    )

    recommended_minus_current = (
        imp_status_input["recommended_value"] - imp_status_input["current_value"]
    )
    rand_gen = np.random.default_rng(_DEFAULT_RANDOM_SEED)
    random_increase = (
        rand_gen.uniform(
            _MIN_IMP_TRACKING_INCREASE_PC,
            _MAX_IMP_TRACKING_INCREASE_PC,
            len(imp_status_input),
        ) * recommended_minus_current
    )
    imp_status_input["value"] = (
        imp_status_input["current_value"] + random_increase
    )
    imp_status_input = imp_status_input.sort_values("timestamp")
    imp_status_input["value"] = imp_status_input.groupby(
        "tag_id",
    )["value"].shift(1)
    imp_status_input["value"] = imp_status_input["value"].fillna(
        imp_status_input["current_value"],
    )
    imp_status_input["tag_id"] = imp_status_input["tag_id"].map(
        {conf.id: conf.tag for conf in tag_meta},
    )

    imp_status_input = imp_status_input.pivot_table(
        index="timestamp", values="value", columns="tag_id",
    ).reset_index()
    imp_status_input.columns.name = None
    dump_csv_data(imp_status_input, "sample_implementation_status_input_data")


def _create_sample_baseline_model(train_data: pd.DataFrame) -> Pipeline:
    """Creates a sample trained baseline model"""
    features = [
        "iron_feed",
        "silica_feed",
        "feed_diff_divide_silica",
    ]
    model = make_pipeline(
        SelectColumns(items=features),
        StandardScaler(),
        ElasticNet(
            max_iter=_BASELINE_MAX_ITER,
            random_state=_DEFAULT_RANDOM_SEED,
            alpha=_BASELINE_ALPHA,
            l1_ratio=_BASELINE_L1_RATIO,
        ),
    )
    model.fit(train_data[features], train_data["silica_conc"])
    dump_pickle_data(model, file_name="sample_trained_baseline_model")

    return model


def _create_sample_baseline_model_errors(
    test_data: pd.DataFrame,
    model: Pipeline,
) -> None:
    """
    Creates a sample dataset with model errors and modifies the target column for it to
    be a good prediction
    """
    test_data = test_data.copy()
    test_data["baseline"] = model.predict(test_data)
    rand_gen = np.random.default_rng(_DEFAULT_RANDOM_SEED)
    test_data["silica_conc"] = test_data["baseline"] + rand_gen.normal(
        0,
        0.1,
        len(test_data),
    )
    test_data["error"] = test_data["baseline"] - test_data["silica_conc"]
    dump_csv_data(test_data, "sample_baseline_model_errors_data")


def _create_sample_actual_values_after_recs(
    baseline_test_data: pd.DataFrame,
    baseline_model: Pipeline,
) -> None:
    """
    Creates simulated data after OAI recommendations. It has the baseline value
    increased by a uniform distribution between ``_MIN_UPLIFT`` and ``_MAX_UPLIFT`` to
    simulate the impact of recommendations. In a real scenario, controls would have also
    been modified by recommendations. However, as they are not used for baselining, it
    is not needed to perform this change.
    """
    baseline_test_data = baseline_test_data.copy()
    rand_gen = np.random.default_rng(_DEFAULT_RANDOM_SEED)
    random_increase = (
        rand_gen.uniform(
            _MIN_UPLIFT,
            _MAX_UPLIFT,
            len(baseline_test_data),
        )
    )
    baseline_test_data["silica_conc"] = (
        baseline_model.predict(baseline_test_data) + random_increase
    )
    dump_csv_data(baseline_test_data, file_name="sample_actual_value_after_recs_data")


def _create_recs_performance(test_data: pd.DataFrame, solutions: Solutions) -> None:
    """Creates a sample dataset with recommendations performance"""

    tags = pd.DataFrame(solutions.controls, columns=["tag"])
    timestamps = pd.DataFrame(test_data["timestamp"].unique(), columns=["timestamp"])
    rand_gen = np.random.default_rng(_DEFAULT_RANDOM_SEED)
    recs_performance = tags.merge(timestamps, how="cross")
    recs_performance_n = len(recs_performance)
    recs_performance["area"] = np.where(
        recs_performance["tag"].isin(solutions.controls[:3]),
        "area_1",
        "area_2",
    )
    recs_performance["uptime"] = (
        rand_gen.uniform(0, 1, recs_performance_n) > _REC_PERFORMANCE_PCT_RIGHT
    )
    recs_performance["reviewed"] = (
        rand_gen.uniform(0, 1, recs_performance_n) > _REC_PERFORMANCE_PCT_RIGHT
    )
    recs_performance["approved"] = (
        rand_gen.uniform(0, 1, recs_performance_n) > _REC_PERFORMANCE_PCT_RIGHT
    )
    recs_performance["implemented"] = (
        rand_gen.uniform(0, 1, recs_performance_n) > _REC_PERFORMANCE_PCT_RIGHT
    )

    recs_performance["reviewed"] = (
        recs_performance["reviewed"] & recs_performance["uptime"]
    )
    recs_performance["approved"] = (
        recs_performance["reviewed"] & recs_performance["approved"]
    )
    recs_performance["implemented"] = (
        recs_performance["approved"] & recs_performance["implemented"]
    )
    recs_performance["implementation_status"] = rand_gen.normal(
        _IMP_STATUS_MEAN, _IMP_STATUS_STD, recs_performance_n,
    )
    recs_performance["implementation_status"] = np.where(
        np.logical_and(
            recs_performance["implemented"],
            ~recs_performance["implementation_status"].between(
                _IMP_STATUS_MIN, _IMP_STATUS_MAX,
            ),
        ),
        rand_gen.uniform(_IMP_STATUS_MIN, _IMP_STATUS_MAX, recs_performance_n),
        recs_performance["implementation_status"],
    )

    dump_csv_data(recs_performance, file_name="sample_recommendations_performance_data")


if __name__ == "__main__":
    main()
