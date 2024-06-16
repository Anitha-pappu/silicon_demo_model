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

from recommend.cra_export import MetaDataConfig, TagMetaData

ValidTypes = tp.Union[str, float, tp.Sequence['ValidTypes'], None]
TSingleDict = tp.Dict[str, ValidTypes]
TJson = tp.List[TSingleDict]


def collect_recs_vs_actual_implementation(
    cra_recommendations: TJson,
    cra_states: TJson,
    cra_runs: TJson,
    imp_data: pd.DataFrame,
    tag_meta: MetaDataConfig[TagMetaData],
    offset: str,
) -> pd.DataFrame:
    """
    Generates data to assess implementation status of recommendations.
    Output data has a row for each recommendation in a run to evaluate. It contains tag
    initial, recommended and current value.

    Example::

        >>> cra_run = [{
        ...     "id": "run_id_1",
        ...     "timestamp": "2023-01-01 02:00:00",
        ...}]
        >>> cra_recs = [{
        ...     "id": "id_1",
        ...     "value": 100,
        ...     "tolerance": 10,
        ...     "run_id": "run_id_1",
        ...     "tag_id": "tag_1_id",
        ...     "target_id": "target_id",
        ...     "is_flagged": False,
        ...     "status": "Pending",
        ...}]
        >>> cra_state = [{
        ...     "id": "id_2",
        ...     "value": 150,
        ...     "run_id": "run_id_1",
        ...     "tag_id": "tag_1_id",
        ...}]
        >>> data = pd.DataFrame(
        ...     [
        ...         [pd.Timestamp("2023-01-01 02:00:00"), 110],
        ...         [pd.Timestamp("2023-01-01 03:00:00"), 120],
        ...         [pd.Timestamp("2023-01-01 04:00:00"), 130],
        ...     ],
        ...     columns=["timestamp", "tag_1"]
        ... )
        >>> tags_meta = MetaDataConfig(
        ...     [{
        ...         "id": "tag_1_id",
        ...         "tag": "tag_1",
        ...         "clear_name": "Tag 1 Name",
        ...         "unit": "Tag 1 Unit",
        ...         "tolerance": 10,
        ...     }],
        ...     schema=TagMetaData,
        ... )
        >>> collect_recs_vs_actual_implementation(
        ...     cra_recs,
        ...     cra_state,
        ...     cra_run,
        ...     data,
        ...     tags_meta,
        ...     "1H",
        ... ).loc[0]
        tag_id                          tag_1_id
        run_id                          run_id_1
        recommended_value                    100
        start_value                          150
        id                                  id_1
        run_timestamp        2023-01-01 02:00:00
        timestamp            2023-01-01 03:00:00
        current_value                        120
        Name: 0, dtype: object

    Args:
        cra_recommendations: Previous inputs to 'recommendations' endpoint of cra_api.
        cra_states: Previous inputs to 'states' endpoint of cra_api.
        cra_runs: Previous inputs to 'runs' endpoint of cra_api.
        imp_data: Input data.
        tag_meta: Tag metadata.
        offset: Time period between recommendation timestamp and its tracking in a
            format compatible with pd.Timedelta. For example, if recommendations are
            evaluated 3 hours from its creation, this argument should be "3H".

    Returns:
        Dataframe with data ready for implementation status calculations.
    """
    cra_recs_pd = (
        pd.DataFrame(cra_recommendations)
        .rename({"value": "recommended_value"}, axis=1)
    )
    cra_states_pd = (
        pd.DataFrame(cra_states)
        .rename({"value": "before_recs_value"}, axis=1)
        .drop("id", axis=1)
    )
    cra_recs_states_pd = cra_recs_pd.merge(cra_states_pd, on=["tag_id", "run_id"])
    cra_recs_states_pd = cra_recs_states_pd[
        ["tag_id", "run_id", "recommended_value", "before_recs_value", "id"]
    ]
    cra_runs_pd = (
        pd.DataFrame(cra_runs)
        .rename({"id": "run_id", "timestamp": "run_timestamp"}, axis=1)
    )[["run_id", "run_timestamp"]]
    cra_runs_pd["run_timestamp"] = pd.to_datetime(cra_runs_pd["run_timestamp"])
    cra_data = cra_recs_states_pd.merge(cra_runs_pd, on="run_id")
    cra_data["start_imp_tracking"] = (
        cra_data["run_timestamp"] + pd.Timedelta(offset)
    )
    cra_data = cra_data.sort_values("run_timestamp")

    imp_data = imp_data.copy()
    imp_data.columns = (
        pd.Series(imp_data.columns)
        .replace({conf.tag: conf.id for conf in tag_meta})
        .values
    )
    imp_data["timestamp"] = pd.to_datetime(imp_data["timestamp"])
    imp_data_melt = imp_data.melt(
        id_vars="timestamp",
        var_name="tag_id",
        value_name="current_value",
    )

    recs_vs_actual = cra_data.merge(
        imp_data_melt,
        left_on=["tag_id", "start_imp_tracking"],
        right_on=["tag_id", "timestamp"],
    )

    return recs_vs_actual.drop(["start_imp_tracking"], axis=1)
