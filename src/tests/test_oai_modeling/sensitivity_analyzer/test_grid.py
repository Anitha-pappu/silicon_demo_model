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

import pandas as pd
import pytest

from modeling.sensitivity_analyser.configs import (
    GridCalculationConfig,
    TParameters,
)
from modeling.sensitivity_analyser.grid import verify_provided_grid


class TestVerifyProvidedGrid(object):
    """
    Test `verify_provided_grid` checks user input for grid is correct
    or creates a grid out from the GridCalculationConfig config.
    """
    @pytest.mark.parametrize(
        "grid_parameters",
        [
            (["% Iron Feed", "% Silica Feed"]),
            (["% Iron Feed"]),
            ({"% Iron Feed": [1, 2, 3]}),
            ({"% Iron Feed": [3, 2, 1]}),
            ({"% Iron Feed": [1, 2, 3], "% Silica Feed": [4, 5, 6]}),
            ({"% Iron Feed": [1, 2, 3], "% Silica Feed": None}),
            ({"% Iron Feed": None, "% Silica Feed": [4, 5, 6]}),
        ],
    )
    def test_verify_provided_grid_returns_correct_keys(
        self,
        simple_data: pd.DataFrame,
        grid_parameters: TParameters,
    ) -> None:
        grid_config = GridCalculationConfig(
            points_count=10,
            calculation_strategy="uniform",
        )
        verified_grid = verify_provided_grid(
            grid_parameters,
            simple_data,
            grid_config,
            initial_slider_position={},
        )
        assert set(grid_parameters) == set(verified_grid.keys())

    @pytest.mark.parametrize(
        "grid_parameters",
        [
            (["% Iron Feed", "% Silica Feed"]),
            (["% Iron Feed"]),
            ({"% Iron Feed": [1, 2, 3]}),
            ({"% Iron Feed": [3, 2, 1]}),
            ({"% Iron Feed": [1, 2, 3], "% Silica Feed": [4, 5, 6]}),
            ({"% Iron Feed": [1, 2, 3], "% Silica Feed": None}),
            ({"% Iron Feed": None, "% Silica Feed": [4, 5, 6]}),
        ],
    )
    def test_verify_provided_grid_returns_sorted_grid(
        self,
        simple_data: pd.DataFrame,
        grid_parameters: TParameters,
    ) -> None:
        grid_config = GridCalculationConfig(
            points_count=10,
            calculation_strategy="uniform",
        )
        verified_grid = verify_provided_grid(
            grid_parameters,
            simple_data,
            grid_config,
            initial_slider_position={},
        )
        for grid_value in verified_grid.values():
            assert grid_value == sorted(grid_value)

    @pytest.mark.parametrize(
        "grid_parameters",
        [
            (["% Iron Feed", "% Silica Feed"]),
            (["% Iron Feed"]),
            ({"% Iron Feed": None, "% Silica Feed": None}),
        ],
    )
    @pytest.mark.parametrize(
        "points_count",
        [10, 20, 50, 100],
    )
    def test_verify_provided_grid_return_correct_number_of_points(
        self,
        simple_data: pd.DataFrame,
        grid_parameters: TParameters,
        points_count: int,
    ) -> None:
        grid_config = GridCalculationConfig(
            points_count=points_count,
            calculation_strategy="uniform",
        )
        verified_grid = verify_provided_grid(
            grid_parameters,
            simple_data,
            grid_config,
            initial_slider_position={},
        )
        for grid_value in verified_grid.values():
            assert len(grid_value) == points_count

    @pytest.mark.parametrize(
        "grid_parameters",
        [
            (["% Iron Feed", "% Silica Feed"]),
            (["% Iron Feed"]),
            ({"% Iron Feed": None, "% Silica Feed": None}),
        ],
    )
    @pytest.mark.parametrize(
        "points_count",
        [10, 20, 50, 100],
    )
    def test_initial_slider_position_is_included_in_grid(
        self,
        simple_data: pd.DataFrame,
        grid_parameters: TParameters,
        points_count: int,
    ) -> None:
        grid_config = GridCalculationConfig(
            points_count=points_count,
            calculation_strategy="uniform",
        )
        initial_slider_positions = {
            "% Iron Feed": 5,
            "% Silica Feed": 10,
        }
        verified_grid = verify_provided_grid(
            grid_parameters,
            simple_data,
            grid_config,
            initial_slider_position=initial_slider_positions,
        )
        for feature_name, grid_value in verified_grid.items():
            assert len(grid_value) == points_count + 1
            assert initial_slider_positions[feature_name] in grid_value
