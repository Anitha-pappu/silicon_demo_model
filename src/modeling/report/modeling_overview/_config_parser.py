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
from dataclasses import dataclass

TRawConfig = tp.Optional[tp.Dict[str, tp.Any]]
TConfig = tp.TypeVar("TConfig")

TGridOptions = tp.Literal["quantiles", "uniform", "quantiles+uniform"]
TAxisRangeOptions = tp.Literal["average", "all"]


@dataclass
class ModelPerformanceConfig(object):
    add_default_baselines: bool = True
    performance_table_sort_by: str = "mae"
    performance_table_sort_order: str = "asc"


@dataclass
class ValidationApproachConfig(object):
    sort_feature_comparison_by_shap: bool = True


@dataclass
class PDPSectionConfig(object):
    max_features_to_display: int = 20
    n_point_in_grid: int = 20
    grid_calculation_strategy: TGridOptions = "quantiles+uniform"
    y_axis_range_mode: TAxisRangeOptions = "all"
    y_axis_tick_values_precision: str = ".2f"
    n_sample_to_calculate_predictions: int = 100
    random_state: int = 42


def parse_config(raw_config: TRawConfig, config_class: tp.Type[TConfig]) -> TConfig:
    return (
        config_class(**raw_config)
        if raw_config
        else config_class()
    )
