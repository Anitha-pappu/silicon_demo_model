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

import datetime
import math
import typing as tp

import numpy as np
import pandas as pd
from numpy import typing as npt
from pandas.core.dtypes.common import is_datetime64_any_dtype
from plotly import graph_objects as go

TRange = tp.Tuple[float, float]
_P = tp.TypeVar("_P", bound=npt.NBitBase)  # noqa: WPS111
_TNumericNDArray = npt.NDArray[np.number[_P]]
TVector = tp.Union[_TNumericNDArray[_P], pd.Series]
HISTOGRAM_Y_RANGE_MULTIPLIER = 20
BIN_NUMBER_UPPER_BOUND_RELATIVE_TO_DATA_SIZE = 0.9
BIN_NUMBER_UPPER_BOUND_MIN = 150.0
BEGINNING_OF_UNIX_TIME = "1970-01-01"
TTimeIntervalRelatedScalar = tp.Union[pd.Timedelta, datetime.timedelta]
TTimeIntervalRelatedSearchBound = tp.Tuple[
    TTimeIntervalRelatedScalar, TTimeIntervalRelatedScalar,
]


def _get_optimal_bins_number(data: _TNumericNDArray[_P]) -> int:
    """ Returns the number of bins needed in order to have bins of optimal width.

    It calculates the optimal bin width with ``calculate_optimal_bin_width``, then just
     divides the data range by that to get the ideal number of bins (then rounds it up).
    """
    optimal_bin_width = calculate_optimal_bin_width(data)
    values_range = data.max() - data.min()
    # print('bin number',int(math.ceil(values_range / optimal_bin_width)), values_range, optimal_bin_width)
    return int(math.ceil(values_range / optimal_bin_width))


def get_histogram_data(
    # `values`` seems an appropriate name in this case
    values: _TNumericNDArray[_P],  # noqa: WPS110
) -> tp.Tuple[_TNumericNDArray[_P], _TNumericNDArray[_P]]:
    """ Gets positions, counts and height of the histogram bars for the ``values``.

    When generates the bars, only includes these with strictly more than zero points
     (if the bar would have zero height, then there is no bar).
    It returns
    - bar "centers" (called like this, but are the left side at the moment, see TODO
     below)
    - bar counts: the number of values in each bin
    - bar heights: a modified version of the bar coints, that accounts for the fact that
     the bars are only supposed to fill part of the graph
    """
    values_dropna = values[~np.isnan(values)]
    bar_counts, bin_edges = np.histogram(
        values_dropna, bins=_get_optimal_bins_number(values_dropna),
    )
    # TODO: Check if a /2 is missing after the parenthesis in the next line
    #  I think this is part of two bugs that compensate
    #  - this is the left side of the bar, not the center
    #  - by default plotly aligns vertical bars on the left, horizontal bar at center
    #  The solution is to
    #  - make this function return the actual centers
    #  - use the x-anchor property set to center when drawing the go.Bar
    #  N.B.: Need to veryfiy if/that this works first
    bar_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])
    bar_centers = bar_centers[bar_counts > 0]
    bar_counts = bar_counts[bar_counts > 0]
    return bar_centers, bar_counts


def calculate_optimal_bin_width(
    data: TVector[_P],
    search_bounds: tp.Optional[tp.Tuple[float, float]] = None,
    search_precision: int = 1000,
) -> float:
    """
    Calculates optimal histogram bin width using cross-validation estimated squared
    error

    ``search_bounds`` are the min and max size of the bin. Default is None, if not
        provided then they are estimated
    ``search_precision`` is the number of candidate bin widths to be generated and
        examined in order to find the ideal number

    Notes:
        This function searches for the optimal bin width within the search bounds.
        If ``search_bounds`` are not provided, they are estimated by using
         ``_determine_bin_width_search_bounds``.
        Once the search bounds are defined, a number ``search_precision`` of candidate
         bin widths is generated, by taking a uniform sampling between min and max of
         the search bounds.
    """
    data_is_time_type = _is_time_data(data)
    if data_is_time_type:  # Makes sure to deal with numbers, not dates
        data, search_bounds = _convert_time_inputs_to_float(  # type: ignore
            data,
            search_bounds,
        )
    if search_bounds is None:
        search_bounds = _determine_bin_width_search_bounds(data=data)
    if min(search_bounds) == max(search_bounds):  # Only one value possible
        optimal_bin_size = min(search_bounds)
        if data_is_time_type:
            optimal_bin_size = pd.to_timedelta(optimal_bin_size, unit="S")
        return optimal_bin_size

    windows = np.linspace(*search_bounds, num=search_precision)
    errs = [
        _calculate_approximation_err_given_bin_width(data, window)
        for window in windows
    ]
    optimal_bin_size = windows[np.argmin(errs)]
    if data_is_time_type:
        optimal_bin_size = pd.to_timedelta(optimal_bin_size, unit="S")
    return optimal_bin_size


def _is_time_data(data: TVector[_P]) -> bool:
    """ Detect if data contains time-related information.

    Notes:
        Recognizes datetime types.
        Implementation tested for the following types:
            - pandas datetime64
            - numpy array containing
                - ``np.datetime64``
                - ``pd.Timestamp``
                - ``datetime.date``
        Note that ``datetime.datetime`` is a subclass of ``datetime.date,`` so the
         former is also accepted See [here](https://stackoverflow.com/questions/16991948/detect-if-a-variable-is-a-datetime-object)
         for more info
    """  # noqa: E501
    timestamp_types_in_numpy_array = [np.datetime64, pd.Timestamp, datetime.date]
    if isinstance(data, pd.Series):
        if is_datetime64_any_dtype(data):
            return True
    elif isinstance(data, np.ndarray):
        is_numpy_time_data = any(
            np.issubdtype(data.dtype, time_type)
            for time_type in timestamp_types_in_numpy_array
        )
        if is_numpy_time_data:
            return True
    return False


def _convert_time_inputs_to_float(
    data: TVector[_P],
    search_bounds: tp.Optional[TTimeIntervalRelatedSearchBound],
) -> tp.Tuple[float, tp.Tuple[float, ...] | None]:
    """ Converts time-related data into floats.

    Used for preparing data for the calculation of the ideal bin width
    ``data``is a series with timestamp data
    ``bounds`` for the data is either ``None`` or a timedelta
    The float values are to be interperted as seconds
    """

    total_seconds: float = pd.to_timedelta(
        pd.Series(data) - pd.Timestamp(BEGINNING_OF_UNIX_TIME),
    ).dt.total_seconds()
    if search_bounds is not None:
        total_seconds_for_each_bound: tuple[float, ...] = tuple(
            pd.to_timedelta(bound).total_seconds() for bound in search_bounds
        )
        return total_seconds, total_seconds_for_each_bound
    return total_seconds, search_bounds


def _determine_bin_width_search_bounds(data: TVector[_P]) -> TRange:
    """ Determines the lower and upper bounds for the optimal bin width to use in a
     histogram of ``data``.

    Notes:
        This function assumes the input ``data`` are of numeric type.
        Search bounds are estimated by taking:
        - the upper bound slightly wider than the data range (some buffer is added so
            the extrema of the datasets are both included).
            This is a good upper bound because it corresponds to (more or less) one
            single bin for all the data (aka "one bin to rule them all" :-D ).
        - a width that would cover the range with either 150 bins, or with
            almost as many bins as are datapoints (technically 0.9 bins per datapoint),
            whichever the highest.
            This is a good lower bound because it corresponds to (on average,
            approximately) one bin for each point

        Handles the case of constant data by returning a bin width of 1
    """
    data_range = data.max() - data.min()
    if data_range == 0:
        return 1.0, 1.0
    one_united_bin_width = data_range + 1  # small addition for max width
    large_number_of_bins = min(
        BIN_NUMBER_UPPER_BOUND_MIN,
        data.size * BIN_NUMBER_UPPER_BOUND_RELATIVE_TO_DATA_SIZE,
    )  # don't want it to be too large
    large_number_of_bins_width = data_range / large_number_of_bins
    search_bounds = (large_number_of_bins_width, one_united_bin_width)
    return search_bounds  # noqa: WPS331  # Naming makes meaning clearer


def _calculate_approximation_err_given_bin_width(
    data: TVector[_P], bin_width: float,
) -> float:
    """
    Implements the method of minimizing integrated mean squared error (with
    leave-one-out) cross validation to determine the ideal number of bin.

    More info
    - in the original paper https://digitalassets.lib.berkeley.edu/sdtr/ucb/text/34.pdf
    - on Wikipedia https://en.wikipedia.org/wiki/Histogram#Minimizing_cross-validation_estimated_squared_error
    """  # noqa: E501
    if bin_width < 0:
        raise ValueError("Please pick window >= 0")

    n_points = data.size
    first_term = 2 / ((n_points - 1) * bin_width)

    n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    bin_counts, _ = np.histogram(data, n_bins)
    squared_sums = (bin_counts ** 2).sum()

    second_term = (
        squared_sums * (n_points + 1) / (n_points ** 2 * (n_points - 1) * bin_width)
    )
    err: float = first_term - second_term
    return err  # noqa: WPS331  # Namings makes meaning clearer


def add_histogram_trace(
    fig: go.Figure,
    feature_values: _TNumericNDArray[_P],
    row: int,
    column: int,
    visible: bool = True,
) -> None:
    """
    Append a histogram trace to the figure into a given row and column.
    This histogram is appended to an invisible secondary y-axis.
    When users changes the primary y-axis, histogram is remained unchanged.
    """
    bar_centers: _TNumericNDArray[_P]
    bar_counts: _TNumericNDArray[_P]
    if visible:
        bar_centers, bar_counts = get_histogram_data(
            feature_values,
        )
    else:
        # Type check fails with empty
        # array for some reason
        bar_centers = np.empty(0)  # type: ignore
        bar_counts = np.empty(0)  # type: ignore
    fig.add_trace(
        go.Bar(
            x=bar_centers,
            y=bar_counts,
            customdata=bar_counts,
            name="feature hist",
            marker={"color": "lightgrey", "line_width": 0},
            opacity=1,
            hovertemplate="%{customdata}",
            legendgroup="histogram",
            showlegend=False,
            visible=visible,
        ),
        row=row,
        col=column,
        secondary_y=True,
    )
    fig.update_yaxes(
        showline=False,
        tickvals=[],
        fixedrange=True,
        linecolor="black",
        row=row,
        col=column,
        side="right",
        secondary_y=True,
        range=(0, bar_counts.mean() * HISTOGRAM_Y_RANGE_MULTIPLIER),
    )
