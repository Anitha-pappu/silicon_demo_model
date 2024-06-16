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

import numpy as np
import pandas as pd

TGridOptions = tp.Literal["quantiles", "uniform", "quantiles+uniform"]


def create_pdp_grid_from_data(
    parameter_name: str,
    data: pd.DataFrame,
    points_count: int,
    calculation_strategy: TGridOptions,
    start_point_quantile: float = 0.05,
    end_point_quantile: float = 0.95,
) -> tp.List[float]:
    """
    Creates a grid for partial dependency plot
    using the statistical data from the dataset.
    Args:
        parameter_name: name of the parameter for which the grid is created
        data: data to calculate statistics from
        points_count: number of points in the grid
        calculation_strategy: calculation strategy for the grid. Supported strategies:
            - 'quantiles' - grid points are calculated as quantiles
            - 'uniform' - grid points are calculated as uniform distribution
            - 'quantiles+uniform' - grid points are
             calculated using quantiles and uniform distribution
        start_point_quantile: minimum quantile used as a starting point in the grid
        end_point_quantile: maximum quantile used as an ending point in the grid,
         endpoint is included

    Returns:
        List of grid points
    """
    if calculation_strategy == "quantiles":
        quantiles = np.linspace(
            start_point_quantile,
            end_point_quantile,
            endpoint=True,
            num=points_count,
        )
        return list(data[parameter_name].quantile(q=quantiles))
    if calculation_strategy == "uniform":
        return list(np.linspace(  # type: ignore
            *data[parameter_name].quantile(
                q=[start_point_quantile, end_point_quantile],
            ),
            endpoint=True,
            num=points_count,
        ))
    if calculation_strategy == "quantiles+uniform":
        n_quantiles = points_count // 2
        n_uniform = points_count - n_quantiles
        quantiles = np.linspace(
            start_point_quantile,
            end_point_quantile,
            endpoint=True,
            num=n_quantiles,
        )
        quantiles_grid: tp.List[float] = (
            list(data[parameter_name].quantile(q=quantiles))
        )
        uniform_grid: tp.List[float] = list(np.linspace(  # type: ignore
            *data[parameter_name].quantile(
                q=[start_point_quantile, end_point_quantile],
            ),
            endpoint=True,
            num=n_uniform,
        ))
        return quantiles_grid + uniform_grid
    raise ValueError(
        f"Unknown grid calculation strategy: {calculation_strategy}. "
        "Please use one of the following: "
        "'quantiles', 'uniform', 'quantiles+uniform'",
    )
