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

import numpy as np
import pandas as pd

_GROUP_TYPING = tp.Dict[
    str,
    tp.Dict[
        str,
        tp.Dict[
            tp.Literal["lower_value", "upper_value"], float,
        ],
    ],
]
_AGG_FUNCTION_TYPING = tp.Callable[[pd.Series], float] | str


class BaselineUplifts(object):
    def __init__(
        self,
        baseline_data: pd.DataFrame,
        after_implementation_data: pd.DataFrame,
        baseline_column: str = "model_prediction",
        after_implementation_column: str = "value_after_recs",
        datetime_column: str = "timestamp",
        group_characteristics: tp.Optional[_GROUP_TYPING] = None,
        default_group: str = "other",
        agg_granularity: tp.Optional[pd.Timedelta | str] = None,
        agg_granularity_function: _AGG_FUNCTION_TYPING = "sum",
        agg_granularity_method: tp.Literal["block", "moving"] = "block",
        original_granularity: tp.Optional[pd.Timedelta | str] = None,
    ):
        """
        Contains the information of uplift data for impact analysis.

        Args:
            baseline_data: Baseline data.
            after_implementation_data: Optimized data.
            baseline_column: Name of the prediction column in baseline data.
            after_implementation_column: Name of the prediction column in optimized
                data.
            datetime_column: Name of the datetime column.
            group_characteristics: Dictionary with keys of groups of uplift data and
                values a list of the columns that they define them. Each column is a
                dictionary with keys the column name and values its `upper_value` and
                `lower_value`.
            default_group: Name of the group of observations that do not belong to any
                other group.
            agg_granularity: Aggregation granularity if required, from pandas offset
                aliases
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            agg_granularity_function: Function to aggregate uplift data.
            agg_granularity_method: Method used to aggregate uplift data.

                - "block": Divides the data in blocks of size `agg_granularity` and
                    aggregates them.
                - "moving": Creates rolling sections in the data of size
                    `agg_granularity` and aggregates them.
            original_granularity: Original granularity of the uplift data, from pandas
                offset aliases
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        """
        self._datetime_column = datetime_column
        self._group_characteristics = group_characteristics
        self._default_group = default_group
        self._group_names = ["all_data"]
        self._has_groups = self._group_characteristics is not None
        self._data = _calculate_data(
            baseline_data,
            baseline_column,
            after_implementation_data,
            after_implementation_column,
            self._datetime_column,
        )
        self._original_data = self.data.copy()
        self._original_granularity = _get_original_granularity(
            pd.Timedelta(original_granularity),
            self._original_data,
        )
        self._agg_granularity = (
            self._original_granularity
            if pd.isna(pd.Timedelta(agg_granularity))
            else pd.Timedelta(agg_granularity)
        )
        self._agg_granularity_function = agg_granularity_function
        self._agg_granularity_method = agg_granularity_method
        self._recalculate_uplift_data()

    @property
    def data(self) -> pd.DataFrame:
        """
        Dataframe with uplift data.
        """
        return self._data

    @property
    def group_names(self) -> tp.List[str]:
        """
        Names of the groups of uplift data.
        """
        return self._group_names

    @property
    def has_groups(self) -> bool:
        """
        Whether uplift data has groups.
        """
        return self._has_groups

    @property
    def original_granularity(self) -> pd.Timedelta:
        """
        Original granularity of the uplift data.
        """
        return self._original_granularity

    @property
    def agg_granularity(self) -> pd.Timedelta:
        """
        Aggregation granularity of the uplift data.
        """
        return self._agg_granularity

    def update_groups(
        self,
        group_characteristics: _GROUP_TYPING,
        default_group: tp.Optional[str] = None,
    ) -> None:
        """
        Updates groups in uplift data.

        Args:
            group_characteristics: Dictionary with keys of groups of uplift data and
                values a list of the columns that they define them. Each column is a
                dictionary with keys the column name and values its `upper_value` and
                `lower_value`.
            default_group: Name of the group of observations that do not belong to any
                other. It is only updated if it is not None.

        """
        self._group_characteristics = group_characteristics
        if default_group is not None:
            self._default_group = default_group
        self._recalculate_uplift_data()

    def update_aggregation(
        self,
        agg_granularity: pd.Timedelta | str,
        agg_granularity_function: tp.Optional[_AGG_FUNCTION_TYPING] = None,
        agg_granularity_method: tp.Optional[tp.Literal["block", "moving"]] = None,
    ) -> None:
        """
        Updates aggregation in uplift data.

        Args:
            agg_granularity: Aggregation granularity if required, from pandas offset
                aliases
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            agg_granularity_function: Function to aggregate uplift data. It is only
                updated if it is not None.
            agg_granularity_method: Method used to aggregate uplift data. It is only
                updated if it is not None.

                - "block": Divides the data in blocks of size `agg_granularity` and
                    aggregates them.
                - "moving": Creates rolling sections in the data of size
                    `agg_granularity` and aggregates them.

        """
        self._agg_granularity = pd.Timedelta(agg_granularity)
        if agg_granularity_method is not None:
            self._agg_granularity_method = agg_granularity_method
        if agg_granularity_function is not None:
            self._agg_granularity_function = agg_granularity_function
        self._recalculate_uplift_data()

    def group_iterator(
        self, dropna: bool = True,
    ) -> tp.Iterator[tp.Tuple[str, pd.DataFrame]]:
        """
        Iterates over groups of uplift data.

        Args:
            dropna: Whether to drop na values in the data.

        """
        for group in self._group_names:
            if group == "all_data":
                data = self._data
            else:
                data = self._data[self._data["group"] == group]
            if dropna:
                data = data.dropna()
            yield group, data

    def _recalculate_uplift_data(self) -> None:
        self._data = self._original_data.copy()
        if self._agg_granularity != self._original_granularity:
            self._data = _aggregate_uplift_data_to_granularity(
                data=self._data,
                agg_granularity=self._agg_granularity,
                time_between_obs=self._original_granularity,
                datetime_column=self._datetime_column,
                agg_function=self._agg_granularity_function,
                method=self._agg_granularity_method,
            )
        if self._group_characteristics is not None:
            self._data = _assign_group_to_data(
                data=self._data,
                group_characteristics=self._group_characteristics,
                default_group=self._default_group,
                datetime_column=self._datetime_column,
            )
            self._has_groups = True
            self._group_names = ["all_data"] + self.data["group"].unique().tolist()
            if self._default_group in self._group_names:
                self._group_names.remove(self._default_group)
                self._group_names.append(self._default_group)
        else:
            self._data = self._data[[self._datetime_column, "uplift"]]


def _get_original_granularity(
    original_granularity: tp.Optional[pd.Timedelta],
    data: pd.DataFrame,
    datetime_column: str = "timestamp",
) -> pd.Timedelta:
    """
    Get the original granularity of the uplift data.

    Args:
        original_granularity: Original granularity of the uplift data.
        data: Dataframe with uplift data.
        datetime_column: Name of the datetime column.

    Returns:
        Original granularity of the uplift data.

    """
    if not pd.isna(original_granularity):
        return original_granularity
    data = data.sort_values(datetime_column)
    min_diff = np.diff(data[datetime_column]).min()
    return pd.Timedelta(min_diff)


def _calculate_data(
    baseline_data: pd.DataFrame,
    baseline_column: str,
    optimized_data: pd.DataFrame,
    optimized_column: str,
    datetime_column: str,
) -> pd.DataFrame:
    """
    Calculate uplift data from optimized and baseline data.

    Args:
        baseline_data: Baseline data.
        baseline_column: Name of the prediction column in baseline data.
        optimized_data: Optimized data.
        optimized_column: Name of the prediction column in optimized data.
        datetime_column: Name of the datetime column.

    Returns:
        Dataframe with uplift data for impact analysis.

    """
    data = optimized_data.merge(baseline_data, on=datetime_column)
    data["uplift"] = data[optimized_column] - data[baseline_column]

    return data


def _assign_group_to_data(
    data: pd.DataFrame,
    group_characteristics: _GROUP_TYPING,
    default_group: str = "other",
    datetime_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Assigns groups to uplift data.

    Args:
        data: Dataframe with uplift data.
        group_characteristics: Dictionary with keys of groups of uplift data and values
            a list of the columns that they define them. Each column is a dictionary
            with keys the column name and values its `upper_value` and `lower_value`.
        default_group: Name of the group of observations that do not belong to any other
            group.
        datetime_column: Name of the datetime column.

    Returns:
        Dataframe with uplift data and groups.

    """
    in_group = {
        group_name: pd.Series(True, index=data.index)  # noqa= WPS528
        for group_name in group_characteristics
    }
    for group_name, group_columns in group_characteristics.items():
        for column, limits in group_columns.items():
            col_filter = data[column].between(
                limits["lower_value"],
                limits["upper_value"],
                inclusive="left",
            )
            in_group[group_name] = in_group[group_name] & col_filter

    num_groups = pd.DataFrame(in_group).sum(axis=1)
    if num_groups.max() > 1:
        raise ValueError(
            "Uplifts are assigned to multiple groups. "
            "Please check the groups definition.",
        )
    data["group"] = pd.DataFrame(in_group).idxmax(axis=1)
    data.loc[num_groups == 0, "group"] = default_group

    return data[[datetime_column, "uplift", "group"]]


def _aggregate_uplift_data_to_granularity(
    data: pd.DataFrame,
    agg_granularity: pd.Timedelta,
    time_between_obs: pd.Timedelta,
    datetime_column: str = "timestamp",
    agg_function: tp.Callable[[pd.Series], float] | str = "sum",
    method: tp.Literal["block", "moving"] = "block",
) -> pd.DataFrame:
    """
    Aggregates uplift data to granularity.

    Args:
        data: Dataframe with uplift data.
        agg_granularity: Aggregation granularity if required, from pandas offset aliases
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        time_between_obs: Time between uplift data values.
        datetime_column: Name of the datetime column.
        agg_function: Function to aggregate uplift data.
        method: Method used to aggregate uplift data.

            - "block": Divides the data in blocks of size `agg_granularity` and
                aggregates them.
            - "moving": Creates rolling sections in the data of size `agg_granularity`
                and aggregates them.

    Returns:
        Dataframe with uplift data aggregated to granularity.

    """
    max_timestamp = (
        data[datetime_column].max()
        - pd.Timedelta(agg_granularity)
        + time_between_obs
    )
    if method == "block":
        data = _aggregate_uplift_data_block(
            data, agg_granularity, datetime_column, agg_function,
        )
    elif method == "moving":
        data = _aggregate_uplift_data_moving(
            data, agg_granularity, datetime_column, agg_function,
        )
    else:
        raise ValueError(
            f"Aggregation method {method} is not supported. "
            f"Please choose between 'block' and 'moving'.",
        )

    return (
        data[data[datetime_column] <= max_timestamp]
        .sort_values(datetime_column)
        .reset_index(drop=True)
    )


def _aggregate_uplift_data_block(
    data: pd.DataFrame,
    agg_granularity: pd.Timedelta,
    datetime_column: str,
    agg_function: tp.Callable[[pd.Series], float] | str,
) -> pd.DataFrame:
    """
    Aggregates uplift data to granularity using blocks of such granularity.

    Args:
        data: Dataframe with uplift data.
        agg_granularity: Aggregation granularity if required, from pandas offset aliases
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        datetime_column: Name of the datetime column.
        agg_function: Function to aggregate uplift data.

    Returns:
        Dataframe with uplift data aggregated to granularity.

    """
    dates = pd.DataFrame(
        pd.date_range(
            data[datetime_column].min(),
            data[datetime_column].max(),
            freq=agg_granularity,
        ),
        columns=["timestamp_block"],
    )
    data = pd.merge_asof(
        data.sort_values(datetime_column),
        dates,
        left_on=[datetime_column],
        right_on=["timestamp_block"],
    )
    data = data.drop(datetime_column, axis=1)
    data = data.rename(columns={"timestamp_block": datetime_column})

    return data.groupby(
        datetime_column, as_index=False,
    ).agg(agg_function, numeric_only=True)


def _aggregate_uplift_data_moving(
    data: pd.DataFrame,
    agg_granularity: pd.Timedelta,
    datetime_column: str,
    agg_function: tp.Callable[[pd.Series], float] | str,
) -> pd.DataFrame:
    """
    Aggregates uplift data to granularity using moving windows of such granularity.

    Args:
        data: Dataframe with uplift data.
        agg_granularity: Aggregation granularity if required, from pandas offset aliases
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        datetime_column: Name of the datetime column.
        agg_function: Function to aggregate uplift data.

    Returns:
        Dataframe with uplift data aggregated to granularity.

    """
    data[f"reverse_{datetime_column}"] = (
        data[datetime_column].max() - data[datetime_column]
    )
    data = data.set_index(datetime_column)
    numeric_cols = data.select_dtypes("number").columns.tolist()
    data = data[numeric_cols]
    data = data.sort_values(f"reverse_{datetime_column}")
    data = data.rolling(
        window=agg_granularity, on=f"reverse_{datetime_column}",
    ).agg(agg_function)

    return data.reset_index().drop(f"reverse_{datetime_column}", axis=1)
