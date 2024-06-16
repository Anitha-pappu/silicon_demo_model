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

from ...configs import LayoutConfig

DEFAULT_SLIDER_MARKS_ROUND = 2
DEFAULT_SLIDER_SHOW_EVERY = 1


def create_marks_for_slider(
    slider_values: tp.List[float],
    initial_slider_position: tp.Mapping[str, float],
    layout_config: LayoutConfig,
    condition: str,
    store_data: tp.Optional[tp.Dict[str, int]] = None,
) -> tp.Dict[int, tp.Any]:
    show_every = (
        layout_config.slider_marks_show_every[condition]
        if condition in layout_config.slider_marks_show_every
        else DEFAULT_SLIDER_SHOW_EVERY
    )
    if store_data is None:
        bold_mark_index = get_initial_value_for_slider(
            slider_values,
            condition,
            initial_slider_position,
        )
    else:
        bold_mark_index = store_data[condition]
    residual_to_show = bold_mark_index % show_every
    marks = {}
    for slider_value_index, slider_value in enumerate(slider_values):
        slider_tick_round = (
            layout_config.slider_marks_round[condition]
            if condition in layout_config.slider_marks_round
            else DEFAULT_SLIDER_MARKS_ROUND
        )
        marks[slider_value_index] = {
            "label": (
                str(round(slider_value, slider_tick_round))
                if slider_value_index % show_every == residual_to_show else ""
            ),
            "style": {
                "font-weight": (
                    "bold"
                    if slider_value_index == bold_mark_index
                    else "normal"
                ),
                "color": "black",
            },
        }
    return marks


def get_initial_value_for_slider(
    condition_values: tp.List[float],
    condition: str,
    initial_slider_position: tp.Mapping[str, float],
) -> int:
    return (
        condition_values.index(initial_slider_position[condition])
        if condition in initial_slider_position else 0
    )
