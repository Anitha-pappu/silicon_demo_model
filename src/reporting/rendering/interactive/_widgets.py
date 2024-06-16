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

import ipywidgets
import pandas as pd
from plotly import offline

from reporting.rendering.types import ReprHtmlCompatible
from reporting.rendering.types.identifiers import (
    remove_section_description_from_structure,
)

from ._figures_manipulation import flatten_dict  # noqa: WPS436
from .protocols import (
    TDictWithFloatsOrStr,
    TInteractiveRenderableDict,
    TInteractiveRenderableFlatDict,
)


def create_plot_demonstration_widget(
    plot_set: TInteractiveRenderableDict,
    sort_by_meta_data: tp.Optional[TDictWithFloatsOrStr] = None,
    nested_names_separator: str = ".",
    ascending: bool = True,
) -> None:
    """
    Creates widget from plot set

    Args:
        plot_set: figures to show, might be nested dict with figures
        nested_names_separator: if nested dict is provided, how to concat figure's
            prefix with name
        sort_by_meta_data: mapping from figure's name to figure's value to sort by
            in widget selector
        ascending: sorting order
    """
    offline.init_notebook_mode()
    # todo: move to notebook mode
    plot_set = remove_section_description_from_structure(plot_set)
    flattened_report_structure = flatten_dict(plot_set, nested_names_separator)
    interactive_renderable_report_structure = _convert_objects_into_interactive(
        flattened_report_structure=flattened_report_structure,
    )
    order_of_keys_in_selector = list(
        pd.Series(sort_by_meta_data)
        .sort_values(ascending=ascending)
        .index,
    ) if sort_by_meta_data is not None else list(
        interactive_renderable_report_structure,
    )

    @ipywidgets.interact(plot_name=order_of_keys_in_selector)
    def show_figure(plot_name: str) -> ReprHtmlCompatible:  # noqa: WPS430
        return flattened_report_structure[plot_name]


# Add other supported types in the future
# At the moment this conversion function is just a placeholder for when more generic
#  python objects will be supported
def _convert_objects_into_interactive(
    flattened_report_structure: TInteractiveRenderableFlatDict,
) -> TInteractiveRenderableFlatDict:
    """
    Ensure objects are interactive-renderable
    Notes:
        Assumes the input is a flattened dict, like the one used for creating the
        interactive widget
    """
    return {
        key: _convert_object_into_interactive(content_element=content_element)
        for key, content_element in flattened_report_structure.items()
    }


# Add other supported types in the future
# At the moment this conversion function is just a placeholder for when more generic
#  python objects will be supported
def _convert_object_into_interactive(content_element):
    """ Convert a supported object into an object with a _repr_html_ method
    """
    if isinstance(content_element, ReprHtmlCompatible):
        return content_element
    object_type = type(content_element)
    return NotImplementedError(
        f"Object of type {object_type} not supported for interactive reports",
    )
