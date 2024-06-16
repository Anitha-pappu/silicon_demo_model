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
Nodes of the On off Logic pipeline.
"""

import logging

import pandas as pd

from preprocessing.tags_config import (
    TagMetaParameters,
    TagOnOffDependencyParameters,
    TagsConfig,
)

logger = logging.getLogger(__name__)


# TODO: simply the input by removing meta_config
def set_off_equipment_to_zero(
    data: pd.DataFrame,
    meta_config: TagsConfig[TagMetaParameters],
    on_off_dep_config: TagsConfig[TagOnOffDependencyParameters],
) -> pd.DataFrame:
    """
    Mark sensor tags to zero based on the on/off tag dependencies defined
    in the Tag Dictionary.

    Args:
        data: input data
        meta_config: TagsConfig object with TagMetaParameters
        on_off_dep_config: TagsConfig object with TagOnOffDependencyParameters
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data should be a Pandas dataframe, got {type(data)}")

    tags = set(data.columns)

    # Extract on_off tags from tags_meta_config
    on_off_tags = {
        tag_param.tag_name
        for tag_param in meta_config.values()
        if tag_param.tag_type == "on_off" and tag_param.tag_name in tags
    }
    if not on_off_tags:
        logger.warning(
            "There are no on/off tags defined in Tag Dictionary "
            "which match any of the columns in the supplied dataframe",
        )
        return data.copy()

    # Build a graph of tag dependencies
    tags_dependents = build_dependents_graph(on_off_dep_config)

    tag_to_dependents = {
        on_off_tag: set(_bfs(on_off_tag, tags_dependents)) & set(tags)
        for on_off_tag in on_off_tags
    }

    data = data.copy()

    # in cases where on-off tags have missing values, we impute with
    # the last known value. To change this behavior, consider implementing
    # custom logic here or earlier in the pipeline.
    data[list(on_off_tags)] = data[list(on_off_tags)].fillna(method="ffill")

    for on_off_tag, dependents in tag_to_dependents.items():
        if not dependents:
            continue

        # set tags to 0 when on/off tag is off. Change rule here as required.
        # For example could set to None or np.NaN instead.
        logger.info(
            f"Setting '{dependents}' to zero when '{on_off_tag}' is off.",
        )
        data.loc[data[on_off_tag] == 0, list(dependents)] = 0

    return data


def build_dependents_graph(
    on_off_dep_config: TagsConfig[TagOnOffDependencyParameters],
) -> dict[str, set[str]]:
    dependents: dict[str, set[str]] = {}

    for tag_dep_param in on_off_dep_config.values():
        for dep in tag_dep_param.on_off_dependencies:
            dependents.setdefault(dep, set()).add(tag_dep_param.tag_name)

    return dependents


def _bfs(key: str, edges: dict[str, set[str]]) -> set[str]:
    """ breadth first search through a dict of edges """
    if key not in edges:
        return set()

    collected = set()
    queue = [key]

    while queue:
        key_to_collect = queue.pop(0)
        collected.add(key_to_collect)
        queue.extend(edges.get(key_to_collect, set()) - collected)

    collected.remove(key)
    return collected
