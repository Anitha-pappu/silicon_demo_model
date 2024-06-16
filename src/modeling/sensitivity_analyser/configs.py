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

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

TRawConfig = tp.Optional[tp.Mapping[str, tp.Any]]

TParametersWithRanges = tp.Mapping[str, tp.Optional[tp.List[float]]]
TParametersWithoutRanges = tp.List[str]
TParameters = tp.Union[TParametersWithRanges, TParametersWithoutRanges]

TGridOptions = tp.Literal["quantiles", "uniform", "quantiles+uniform"]


class GridCalculationConfig(PydanticBaseModel):
    """
    Dataclass to parse grid calculations configs passed into
    ``modeling.sensitivity_analyzer.get_sensitivity_analyzer``

    Args:
        points_count: total number of points to be generated in the grid;
        calculation_strategy: strategy for grid composition:
            * "quantiles": grid will be composed based on the quantiles of the data;
            * "uniform": first, `start_point_quantile` and `end_point_quantile`;
             will be calculated, then grid will be
             composed uniformly between these points;
            * "quantiles+uniform": combined grid from the methods above;
        start_point_quantile: quantile extracted from data used as a start of the grid
        end_point_quantile: quantile extracted from data used as the end of the grid


    """
    points_count: int
    calculation_strategy: TGridOptions
    start_point_quantile: float = 0.05
    end_point_quantile: float = 0.95


class FeaturesToPlotConfig(GridCalculationConfig):
    """
    Dataclass to parse ``features_to_plot_grid_config`` parameter
    """
    points_count: int = 20
    calculation_strategy: TGridOptions = "quantiles+uniform"


class FeaturesToManipulateConfig(GridCalculationConfig):
    """
    Dataclass to parse ``manageable_conditions_grid_config`` parameter
    """
    points_count: int = 9
    calculation_strategy: TGridOptions = "uniform"


class LayoutConfig(PydanticBaseModel):
    """
    Dataclass to parse ``layout_config`` parameter

    Args:
        show_histogram: enable or disable
         histogram calculation and visualization
        show_legend: enable or disable
         legend rendering for sensitivity charts
        share_y_axis: enable or disable the shared Y-axis between the subplots
        max_plots_in_row: number of subplots in the row
        sidebar_width: width of the sidebar layout of the Dash application;
         Can be either fixed ("XXpx") when sidebar remains stable when
         size of the window is changed or take proportion of the screen ("XX%")
         when sidebar will always take the chosen proportion of the width of the window
        visualization_mapping: mapping from names used by the model
         to human-readably names used for all types of visualizations
        slider_marks_round: mapping from model feature into number used to round
         floating number used for sliders' marks
        slider_marks_show_every: mapping from model feature name into number;
         sets the distance between marks on the slider that are visualized.
         Use it when names of the values of the features are
         take too much space of the screen
         or when there are too many slider values
        slider_annotation_round: mapping from model feature into number used to round
         floating number used for sliders' annotations
        save_slider_position: enable or disable button that saves
         current slider position that is shown on the plots;
        show_informational_tooltips: enable or disable informational tooltips
    """
    show_histogram: bool = True
    show_legend: bool = True
    share_y_axis: bool = True
    max_plots_in_row: int = 3
    sidebar_width: str = "500px"
    save_slider_position: bool = True
    slider_marks_show_every: tp.Mapping[str, int] = Field(default_factory=dict)
    slider_marks_round: tp.Mapping[str, int] = Field(default_factory=dict)
    slider_annotation_round: tp.Mapping[str, int] = Field(default_factory=dict)
    visualization_mapping: tp.Mapping[str, str] = Field(default_factory=dict)
    show_informational_tooltips: bool = True
