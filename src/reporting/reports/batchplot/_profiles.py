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

import logging
import typing as tp
from itertools import chain

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm.auto import tqdm

from ...charts.batchplot import (
    TMappingToRange,
    TRange,
    add_time_step_column,
    plot_batch_profile,
)
from ...charts.primitives import add_vertical_lines

logger = logging.getLogger(__name__)

_TSensorsGroups = tp.Dict[str, tp.List[str]]
_TSensorsToPlot = tp.Union[_TSensorsGroups, tp.List[str]]

_TIME_STEP_PER_BATCH_COLUMN = "__time_step_per_batch"
_BATCH_PROFILE_TITLE = "Profile of Batch #{batch_id}, started on {start_time}"


def plot_batches_profiles(
    sensor_data: pd.DataFrame,
    sensors_to_plot: _TSensorsToPlot,
    batch_id_column: str,
    datetime_column: str,
    time_step_range: tp.Optional[TRange] = None,
    align_time_between_batches: bool = True,
    batch_meta: tp.Optional[pd.DataFrame] = None,
    times_to_mark: tp.Optional[tp.List[str]] = None,
    verbose: bool = False,
) -> tp.Dict[int, go.Figure]:
    """Create a profile plot for all the batches fixing the axis limits

    Args:
        sensor_data: sensors' time series; each timestamp is assigned to a batch
        sensors_to_plot: sensors' names to plot
        batch_id_column: column in `sensor_data` standing for integer ID of the batch
        datetime_column: column in `sensor_data` standing for timestamp of the sensor
         data
        time_step_range: plotting range for relative time inside batch
        align_time_between_batches: flag to disable time alignment
        (using same ranges for x-axis for all batch profiles) between batches
        batch_meta: batches' metadata with batch level information used for plotting
         `times_to_mark`
        times_to_mark: column names from `batch_meta` to plot as vertical lines for
         each batch profile
        verbose: shows progress bar if set to true

    Notes:
        sensor_data:
            Columns:
                Name: `datetime_column`, dtype: datetime
                Name: `batch_id_column`, dtype: int
                Name: `sensors_to_plot[i]`, dtype: float
            Example::
                datetime_column = 'datetime'
                batch_id_column = 'batch_id'
                sensors_to_plot = ['filter_infeed', 'reactor_P']

                |    |   batch_id | datetime            |   filter_infeed |   reactor_P |
                |---:|-----------:|:--------------------|----------------:|------------:|
                |  0 |          0 | 2017-03-27 00:02:00 |     -0.150427   |    0        |
                |  1 |          0 | 2017-03-27 00:03:00 |     -1.51221    |    0.544865 |
                |  2 |          0 | 2017-03-27 00:04:00 |      1.68297    |    0.970921 |

        batch_meta:
            Index:
                Int64Index, Name: batch_id, dtype: int
            Columns:
                Name: batch_id, dtype: datetime
                Name: `times_to_mark[i]`, dtype: datetime
            Example::
                times_to_mark = ['reactor_start_time','reactor_end_time','filter_start_time','filter_end_time']

                |   batch_id | reactor_start_time   | reactor_end_time    | filter_start_time   | filter_end_time     |
                |-----------:|:---------------------|:--------------------|:--------------------|:--------------------|
                |          0 | 2017-03-27 00:02:00  | 2017-03-27 00:40:00 | 2017-03-27 00:52:00 | 2017-03-27 01:05:00 |
                |          1 | 2017-03-27 00:41:00  | 2017-03-27 01:58:00 | 2017-03-27 02:12:00 | 2017-03-27 02:24:00 |
                |          2 | 2017-03-27 01:59:00  | 2017-03-27 02:09:00 | 2017-03-27 02:44:00 | 2017-03-27 02:45:00 |

    Returns:
        Dict mapping from batch_id to batch profile chart
    """  # noqa: E501

    if times_to_mark is not None and batch_meta is None:
        logger.warning(
            "Please provide `batch_meta` when providing `times_to_mark`. "
            "Otherwise `times_to_mark` won't be shown on the batch profile.",
        )

    sensors_to_plot_by_phases = _validate_sensors_to_plot(sensor_data, sensors_to_plot)

    sensor_data = sensor_data.copy()
    sensor_data = add_time_step_column(
        sensor_data,
        timestamp=datetime_column,
        batch_id_col=batch_id_column,
        time_step_column=_TIME_STEP_PER_BATCH_COLUMN,
    )
    sensors_ranges = _get_features_plotting_limits(
        sensor_data,
        sensors_to_plot_by_phases,
        _TIME_STEP_PER_BATCH_COLUMN,
        time_step_range,
    )
    time_step_ranges = _get_time_step_ranges(
        sensor_data,
        batch_id_column=batch_id_column,
        time_step_column=_TIME_STEP_PER_BATCH_COLUMN,
        time_step_range=time_step_range,
        align_time_between_batches=align_time_between_batches,
    )
    all_batch_profiles_fig = {}
    grouped_sensor_data = (
        tqdm(sensor_data.groupby(batch_id_column)) if verbose
        else sensor_data.groupby(batch_id_column)
    )
    batch_id: int
    for batch_id, sensors_per_batch in grouped_sensor_data:
        _add_single_batch_profile_fig(
            all_batch_profiles_fig,
            sensors_per_batch,
            time_step_ranges,
            batch_id,
            sensors_to_plot_by_phases,
            datetime_column,
            sensors_ranges,
            batch_meta,
            times_to_mark,
        )
    return all_batch_profiles_fig


def _add_single_batch_profile_fig(
    all_batch_profiles_fig,
    sensors_per_batch,
    time_step_ranges,
    batch_id,
    sensors_to_plot_by_phases,
    datetime_column,
    sensors_ranges,
    batch_meta,
    times_to_mark,
):
    """ Adds one entry to the ``all_batch_profiles_fig`` dict.

    The key is the ``batch_id``, the value is the figure ``fig`` for that batch
    Modifies ``all_batch_profiles_fig`` in place.
    """
    fig = plot_batch_profile(
        one_batch_sensor_data=(
            sensors_per_batch.set_index(_TIME_STEP_PER_BATCH_COLUMN)
            .sort_index()
            .loc[slice(*time_step_ranges[batch_id])]
        ),
        sensors_to_plot_by_phases=sensors_to_plot_by_phases,
        title=_BATCH_PROFILE_TITLE.format(
            batch_id=batch_id, start_time=sensors_per_batch[datetime_column].min(),
        ),
        sensors_plot_range=sensors_ranges,
        time_step_range=time_step_ranges[batch_id],
    )
    events_timings = _extract_events_timings(
        sensors_per_batch[datetime_column], batch_id, batch_meta, times_to_mark,
    )
    add_vertical_lines(fig, events_timings)
    all_batch_profiles_fig[batch_id] = fig


def _validate_sensors_to_plot(
    sensor_data: pd.DataFrame, sensors_to_plot: _TSensorsToPlot,
) -> tp.Dict[str, tp.List[str]]:
    """
    Checks:
        * input format is either a list or a mapping into lists
        * all sensors exist in `sensor_data` and are of numeric dtype

    Returns: validated sensors in a mapping format
    """
    sensors_by_groups = (
        dict(unknown=sensors_to_plot)
        if isinstance(sensors_to_plot, list)
        else sensors_to_plot
    )

    missing_sensors = {
        sensor
        for sensor in chain.from_iterable(sensors_by_groups.values())
        if sensor not in sensor_data.columns
    }
    if missing_sensors:
        raise ValueError(f"Couldn't find following sensors: {missing_sensors}")

    non_numeric_sensors = {
        sensor
        for sensor in chain.from_iterable(sensors_by_groups.values())
        if not np.issubdtype(sensor_data[sensor].dtype, np.number)
    }
    if non_numeric_sensors:
        logger.warning(
            f"Please cast your data before plotting profiles. "
            f"Following non-numeric data columns were provided: {non_numeric_sensors}.",
        )
    return sensors_by_groups


def _get_features_plotting_limits(
    sensor_data: pd.DataFrame,
    sensors_to_plot_by_phases: tp.Dict[str, tp.List[str]],
    time_step_column: str,
    time_step_range: tp.Optional[TRange],
) -> TMappingToRange:
    """
    Returns: mapping {`column` -> `min/max limit`}
        for `sensors_to_plot_by_phases.values()` columns of `sensor_data`.
        It'll be useful to have the same limits across all batches' profiles.

    Notes:
        * Limits are calculated based on sensors' distributions.
        * Limits are calculated within provided `time_step_range`
        * Non-numeric columns are assigned `(None, None)` limit
    """
    sensors = list(chain.from_iterable(sensors_to_plot_by_phases.values()))
    return {
        feature: _get_series_limit(series, index_scope=time_step_range)
        for feature, series in sensor_data.set_index(time_step_column)[sensors].items()
    }


def _get_time_step_ranges(
    sensor_data: pd.DataFrame,
    batch_id_column: str,
    time_step_column: str,
    time_step_range: tp.Optional[TRange],
    align_time_between_batches: bool,
) -> tp.Dict[int, tp.Tuple[float, float]]:
    if time_step_range is not None:
        return {
            batch_id: time_step_range
            for batch_id in sensor_data[batch_id_column].unique()
        }

    time_step_ranges = {
        batch_id: (
            batch_sensors[time_step_column].min(),
            batch_sensors[time_step_column].max(),
        )
        for batch_id, batch_sensors in sensor_data.groupby(batch_id_column)
    }
    if align_time_between_batches:
        aligned_x_range_start = min(start for start, _ in time_step_ranges.values())
        aligned_x_range_end = max(end for _, end in time_step_ranges.values())
        time_step_ranges = {
            batch_id: (aligned_x_range_start, aligned_x_range_end)
            for batch_id in sensor_data[batch_id_column].unique()
        }
    return time_step_ranges


def _get_series_limit(
    single_sensor_data: pd.Series,
    index_scope: tp.Optional[TRange],
    quantile_to_drop_from_both_tails: float = 0.05,
    extra_margin_multiplier: float = 0.1,
) -> TRange:
    """
    Computes the line plot limits of a given batch time series.
    If series are not numeric, returns (None, None).

    Args:
        single_sensor_data: single sensor time series for all batches
        quantile_to_drop_from_both_tails: percentage of data to drop
            from both sides of distribution when computing limits
        index_scope: start & end index to consider when computing the limit
        extra_margin_multiplier: multiplication used for extending margin between
            limits, used as: `new_limit = limit - extra_room * range_between_limits`

    Returns:
        The minimum and the maximum limit for the variable
    """

    if not np.issubdtype(single_sensor_data.dtype, np.number):
        return None, None

    if index_scope is not None:
        # in order for scope to work, we will need a sorted series
        single_sensor_data = single_sensor_data.sort_index().loc[slice(*index_scope)]

    # computing the limits using quantiles
    quantile_to_drop = quantile_to_drop_from_both_tails / 2
    lim_min = single_sensor_data.quantile(q=quantile_to_drop)
    lim_max = single_sensor_data.quantile(q=1 - quantile_to_drop)

    # calculating margin
    lim_range = lim_max - lim_min
    margin = lim_range * extra_margin_multiplier

    return lim_min - margin, lim_max + margin


def _extract_events_timings(
    one_batch_timestamps: pd.Series,
    batch_id: int,
    batch_meta: tp.Optional[pd.DataFrame],
    times_to_mark: tp.Optional[tp.List[str]],
) -> tp.Optional[tp.Dict[str, int]]:
    if times_to_mark is None or batch_meta is None:
        return None

    if batch_id not in batch_meta.index:
        logger.warning(
            f"Couldn't find meta information about batch_id #{batch_id}. "
            f"This batch profile will be plotted without event lines on it.",
        )
        return None

    ordered_timestamps = one_batch_timestamps.sort_values()
    return {
        event_name: ordered_timestamps.searchsorted(event_time, side="left")
        for event_name, event_time in batch_meta.loc[batch_id, times_to_mark].items()
    }
