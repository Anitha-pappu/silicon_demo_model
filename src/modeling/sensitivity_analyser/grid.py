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

from modeling.report.charts import create_pdp_grid_from_data

from .configs import GridCalculationConfig, TParameters


def verify_provided_grid(
    grid_parameters: TParameters,
    data: pd.DataFrame,
    grid_calculation_config: GridCalculationConfig,
    initial_slider_position: tp.Mapping[str, float],
) -> tp.Dict[str, tp.List[float]]:
    parameters_grid = {}
    if isinstance(grid_parameters, list):
        parameters_grid = {
            parameter: create_pdp_grid_from_data(
                parameter,
                data,
                grid_calculation_config.points_count,
                grid_calculation_config.calculation_strategy,
                grid_calculation_config.start_point_quantile,
                grid_calculation_config.end_point_quantile,
            )
            for parameter in grid_parameters
        }
    if isinstance(grid_parameters, dict):
        for parameter_name, parameter_grid in grid_parameters.items():
            if parameter_grid is None:
                parameters_grid[parameter_name] = create_pdp_grid_from_data(
                    parameter_name,
                    data,
                    grid_calculation_config.points_count,
                    grid_calculation_config.calculation_strategy,
                    grid_calculation_config.start_point_quantile,
                    grid_calculation_config.end_point_quantile,
                )
            else:
                parameters_grid[parameter_name] = parameter_grid
    parameters_with_sorted_grid = {}
    for name, grid in parameters_grid.items():
        if name in initial_slider_position:
            parameters_with_sorted_grid[name] = sorted(
                grid + [initial_slider_position[name]],
            )
        else:
            parameters_with_sorted_grid[name] = sorted(grid)
    return parameters_with_sorted_grid
