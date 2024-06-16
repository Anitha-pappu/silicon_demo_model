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
from types import MappingProxyType

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash_iconify import DashIconify

from ...callbacks_registry import (
    CALLBACK_ID_BUTTON,
    CALLBACK_ID_CONDITION_ANNOTATION,
    CALLBACK_ID_CONDITION_SLIDER,
    CALLBACK_ID_INFO_ICON,
    CALLBACK_ID_RESET_BUTTON,
)
from ...configs import LayoutConfig
from ..text_components import (
    INFO_ICON_TEXT,
    MAIN_BUTTON_TEXT,
    MAIN_BUTTON_TOOLTIP_TEXT,
    RESET_BUTTON_TEXT,
    RESET_BUTTON_TOOLTIP_TEXT,
    SIDEBAR_TEXT_TITLE,
)
from .slider_utils import create_marks_for_slider, get_initial_value_for_slider

SIDEBAR_STYLE = MappingProxyType({
    "position": "fixed",
    "overflow-y": "scroll",
    "top": 0,
    "max-height": "100%",
    "left": 0,
    "bottom": 0,
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
})

DISPLAY_NONE = MappingProxyType({"display": "none"})


def create_sidebar(
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
    initial_slider_values: tp.Mapping[str, float],
    layout_config: LayoutConfig,
) -> html.Div:
    show_button = layout_config.save_slider_position
    div_style = dict(SIDEBAR_STYLE, width=layout_config.sidebar_width)
    return html.Div(
        [
            html.H1(
                children=[
                    SIDEBAR_TEXT_TITLE,
                    html.Span(
                        DashIconify(
                            icon="material-symbols:info-outline-rounded",
                            id=CALLBACK_ID_INFO_ICON,
                        ),
                        style=(
                            {"float": "right"}
                            if layout_config.show_informational_tooltips
                            #  readability is better this way
                            else DISPLAY_NONE
                        ),
                    ),
                    dbc.Tooltip(
                        INFO_ICON_TEXT,
                        target=CALLBACK_ID_INFO_ICON,
                        placement="bottom",
                        className="text-left",
                        style=(
                            {"text-transform": "none"}
                            if layout_config.show_informational_tooltips
                            else DISPLAY_NONE
                        ),
                    ),
                ],
                className="display-7",
            ),
            html.Hr(),
            *_create_conditions_layout(
                features_to_manipulate,
                initial_slider_values,
                layout_config,
            ),
            html.Div(
                style={
                    "display": "flex",
                    "justify-content": "space-between",
                    "width": "100%",
                },
                children=[
                    html.Button(
                        MAIN_BUTTON_TEXT,
                        id=CALLBACK_ID_BUTTON,
                        n_clicks=0,
                        className="btn btn-outline-primary",
                        style=(
                            {"width": "70%", "margin-right": "5%"}
                            if show_button else DISPLAY_NONE
                        ),
                    ),
                    dbc.Tooltip(
                        MAIN_BUTTON_TOOLTIP_TEXT,
                        target=CALLBACK_ID_BUTTON,
                        placement="bottom",
                        delay={"show": 1000, "hide": 50},
                        style=(
                            {"text-transform": "none"}
                            if layout_config.show_informational_tooltips
                            else DISPLAY_NONE
                        ),
                    ),
                    html.Button(
                        RESET_BUTTON_TEXT,
                        id=CALLBACK_ID_RESET_BUTTON,
                        style={"width": "25%"} if show_button else DISPLAY_NONE,
                        className="btn btn-outline-primary",
                    ),
                    dbc.Tooltip(
                        RESET_BUTTON_TOOLTIP_TEXT,
                        target=CALLBACK_ID_RESET_BUTTON,
                        delay={"show": 1000, "hide": 50},
                        placement="bottom",
                        style=(
                            {"text-transform": "none"}
                            if layout_config.show_informational_tooltips
                            else DISPLAY_NONE
                        ),
                    ),
                ],
            ),
        ],
        style=div_style,
    )


def _create_conditions_layout(
    features_to_manipulate: tp.Mapping[str, tp.List[float]],
    initial_slider_values: tp.Mapping[str, float],
    layout_config: LayoutConfig,
) -> tp.List[html.Div]:
    components = []
    for condition, condition_grid in features_to_manipulate.items():
        components.append(
            html.Div(
                [
                    html.P(
                        id=CALLBACK_ID_CONDITION_ANNOTATION.format(condition=condition),
                    ),
                    dcc.Slider(
                        id=CALLBACK_ID_CONDITION_SLIDER.format(condition=condition),
                        step=None,
                        value=get_initial_value_for_slider(
                            condition_grid,
                            condition,
                            initial_slider_values,
                        ),
                        marks=create_marks_for_slider(
                            condition_grid,
                            initial_slider_values,
                            layout_config,
                            condition,
                        ),
                        updatemode="drag",
                    ),
                ],
                style={"padding": "1rem 0rem"},
            ),
        )
    return components
