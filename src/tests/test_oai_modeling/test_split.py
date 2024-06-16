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
Test splitting functionality.
"""
import typing as tp
from itertools import product
from math import ceil, floor

import numpy as np
import pandas as pd
import pytest

from modeling.splitters import (
    ByColumnValueSplitter,
    ByDateSplitter,
    ByFracSplitter,
    ByIntervalsSplitter,
    ByLastWindowSplitter,
    BySequentialSplitter,
)


class TestSplitByDate(object):
    """ Test split by date functionality. """

    def test_raises_on_date_out_of_bounds(self, simple_data):
        with pytest.raises(ValueError, match="lies outside"):
            splitter = ByDateSplitter("date", "3/10/2199 00:00")
            splitter.split(simple_data)

    def test_raises_when_timezones_misaligned(self, simple_data):
        tz = "US/Eastern"
        simple_data = simple_data.set_index("date").tz_localize(tz=tz).reset_index()
        date = "3/10/17 10:00"
        splitter = ByDateSplitter("date", date)
        with pytest.raises(ValueError, match="doesn't match timezone"):
            splitter.split(simple_data)

    def test_raises_on_missing_column(self, simple_data):
        with pytest.raises(ValueError, match="column is missing"):
            ByDateSplitter("missing_column", "3/10/2199 00:00").split(simple_data)

    @pytest.mark.parametrize("split_datetime,is_valid", [
        ("3/10/17 10:00", True),
        ("2017-03-10 10:00", True),
        ("8am on March 3rd of 2017", True),
        (42, True),
        ("Breakfast time", False),
        ("M@rch", False),
        ("<1-1-1>", False),
        ("42/42/42 42:42", False),
    ])
    def test_cannot_initialize_if_invalid_datetime(
        self,
        split_datetime: tp.Any,
        is_valid: bool,
    ) -> None:
        if is_valid:
            ByDateSplitter("date", split_datetime)
        else:
            with pytest.raises(ValueError):
                ByDateSplitter("date", split_datetime)

    def test_split_correct(self, simple_data):
        date = pd.to_datetime("3/10/17 10:00")
        splitter = ByDateSplitter("date", date)
        before, after = splitter.split(simple_data.sample(frac=1))
        assert all(before["date"] < date)
        assert all(after["date"] >= date)

    def test_string_representation(self):
        date = pd.to_datetime("3/10/17 10:00")
        splitter = ByDateSplitter("date", date)
        assert isinstance(repr(splitter), str)
        assert "ByDateSplitter" in repr(splitter)


class TestSplitByFrac(object):
    """ Test split by frac functionality. """

    def test_raises_sort_date(self, simple_data):
        with pytest.raises(ValueError, match="parameter `datetime_column` to specify"):
            before, after = ByFracSplitter(sort=True).split(simple_data)

    def test_raises_on_missing_column(self, simple_data):
        with pytest.raises(ValueError, match="column is missing"):
            ByFracSplitter(
                datetime_column="missing_column",
                sort=True,
                test_size=0.3,
            ).split(simple_data)

    def test_no_shuffle_on_sort(self, simple_data):
        simple_data = simple_data.sample(frac=1)

        test_size = 0.3

        train, test = ByFracSplitter(
            datetime_column="date",
            sort=True,
            test_size=test_size,
        ).split(simple_data)

        n = len(simple_data)

        assert len(test) == ceil(n * test_size)
        assert len(train) == floor(n * (1 - test_size))

        assert all(
            train_date < test_date
            for train_date, test_date in product(train["date"], test["date"])
        )

    def test_shuffle_on_no_sort(self, simple_data):
        simple_data = simple_data.sort_values(by="date")
        test_size = 0.3
        train, test = ByFracSplitter(
            datetime_column="date",
            test_size=test_size,
            shuffle=True,
            random_state=0,
        ).split(simple_data)

        n = len(simple_data)

        assert len(test) == ceil(n * test_size)
        assert len(train) == floor(n * (1 - test_size))

        assert any(
            train_date < test_date
            for train_date, test_date in product(train["date"], test["date"])
        )
        assert any(
            train_date > test_date
            for train_date, test_date in product(train["date"], test["date"])
        )

    def test_string_representation(self):
        splitter = ByFracSplitter(
            datetime_column="date",
            test_size=0.5,
            shuffle=True,
            random_state=0,
        )
        assert isinstance(repr(splitter), str)
        assert "ByFracSplitter" in repr(splitter)


class TestSplitByPeriod(object):
    """ Test split by period functionality. """

    def test_raises_no_periods(self, simple_data):
        with pytest.raises(ValueError, match="at least one of"):
            ByIntervalsSplitter("date").split(simple_data)

    @pytest.mark.parametrize(
        "train_periods", [
            [("2020-01-01 00:00", "2020-01-01 00:00")],
            [
                ("2020-01-01 00:00", "2020-01-01 00:00"),
                ("2020-01-01 00:00", "2020-01-01 00:00"),
            ],
        ],
    )
    def test_raises_on_missing_column(self, simple_data, train_periods):
        with pytest.raises(ValueError, match="column is missing"):
            ByIntervalsSplitter(
                "missing_column",
                train_intervals=train_periods,
            ).split(simple_data)

    @pytest.mark.parametrize(
        "train_periods", [
            [("2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 00:00")],
            None,
            [
                ("2020-01-01 00:00", "2020-01-01 00:00"),
                ("2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 00:00"),
            ],
        ],
    )
    @pytest.mark.parametrize(
        "test_periods",
        [
            [("2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 00:00")],
            None,
            [
                ("2020-01-01 00:00", "2020-01-01 00:00"),
                ("2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 00:00"),
            ],
        ],
    )
    def test_raises_period_not_two_values(
        self, simple_data, train_periods, test_periods,
    ):
        if train_periods is None and test_periods is None:
            return

        with pytest.raises(ValueError, match="can only have two values"):
            ByIntervalsSplitter(
                datetime_column="date",
                train_intervals=train_periods,
                test_intervals=test_periods,
            ).split(simple_data)

    @pytest.mark.parametrize(
        "train_periods,test_periods",
        [
            (
                [("2020-01-01 00:00", "2020-01-02 00:00")],
                [("2020-01-01 12:00", "2020-01-03 00:00")],
            ),
            (
                [("2020-01-02 00:00", "2020-01-03 00:00")],
                [("2020-01-01 12:00", "2020-01-02 12:00")],
            ),
            (
                [("2020-01-01 00:00", "2020-01-02 00:00")],
                [("2020-01-01 06:00", "2020-01-01 18:00")],
            ),
            (
                [("2020-01-01 06:00", "2020-01-01 18:00")],
                [("2020-01-01 00:00", "2020-01-02 00:00")],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "train_past_non_overlapping",
        [None, [("2019-01-01 00:00", "2019-05-01 00:00")]],
    )
    @pytest.mark.parametrize(
        "train_future_non_overlapping",
        [None, [("2021-01-01 00:00", "2021-05-01 00:00")]],
    )
    @pytest.mark.parametrize(
        "test_past_non_overlapping", [None, [("2019-06-01 00:00", "2019-07-01 00:00")]],
    )
    @pytest.mark.parametrize(
        "test_future_non_overlapping",
        [None, [("2021-06-01 00:00", "2021-06-01 00:00")]],
    )
    def test_raises_overlapping_periods(
        self,
        simple_data,
        train_periods,
        test_periods,
        train_past_non_overlapping,
        train_future_non_overlapping,
        test_past_non_overlapping,
        test_future_non_overlapping,
    ):
        def assert_raises(_train, _test):  # noqa: WPS430
            with pytest.raises(ValueError, match="overlapping"):
                ByIntervalsSplitter(
                    "date", train_intervals=_train, test_intervals=_test,
                ).split(simple_data)

        # TODO: Refactor multiline conditions
        if (  # noqa: WPS337
            train_past_non_overlapping is None
            and train_future_non_overlapping is None
            and test_past_non_overlapping is None
            and test_future_non_overlapping is None
        ):
            assert_raises(train_periods, test_periods)

        train_past_non_overlapping = train_past_non_overlapping or []
        train_future_non_overlapping = train_future_non_overlapping or []

        test_past_non_overlapping = test_past_non_overlapping or []
        test_future_non_overlapping = test_future_non_overlapping or []

        # Explicitly concatenate in the wrong order to test sorting.
        train_periods = (
            train_future_non_overlapping + train_periods + train_past_non_overlapping
        )
        test_periods = (
            test_future_non_overlapping + test_periods + test_past_non_overlapping
        )

        assert_raises(train_periods, test_periods)

    def test_single_value_split(self, simple_data):
        period = [("3/10/17 5:00", "3/10/17 14:00")]

        train, test = ByIntervalsSplitter(
            "date", train_intervals=period,
        ).split(simple_data)

        assert all(
            (train["date"] >= period[0][0]) & (train["date"] < period[0][1]),
        )
        assert all(
            (test["date"] < period[0][0]) | (test["date"] >= period[0][1]),
        )

        train, test = ByIntervalsSplitter(
            "date", test_intervals=period,
        ).split(simple_data)

        assert all(
            (test["date"] >= period[0][0]) & (test["date"] < period[0][1]),
        )
        assert all(
            (train["date"] < period[0][0]) | (train["date"] >= period[0][1]),
        )

    def test_multiple_train_periods(self, simple_data):
        train_periods = [
            ["3/10/17 3:00", "3/10/17 9:00"],
            ["3/10/17 14:00", "3/10/17 23:00"],
        ]

        train, test = ByIntervalsSplitter(
            "date", train_intervals=train_periods,
        ).split(simple_data)

        assert len(train) == 12
        assert len(test) == len(simple_data) - len(train)

    def test_multiple_test_periods(self, simple_data):
        test_periods = [
            ["3/10/17 00:00", "3/10/17 05:00"],
            ["3/10/17 03:00", "3/10/17 06:00"],
            ["3/10/17 12:00", "3/10/17 15:00"],
        ]

        train, test = ByIntervalsSplitter(
            "date", test_intervals=test_periods,
        ).split(simple_data)

        assert len(test) == 8
        assert len(train) == len(simple_data) - len(test)

    def test_multiple_train_test_periods(self, simple_data):
        train_periods = [
            ["3/10/18 00:00", "3/10/18 10:00"],  # No effect
            ["3/10/17 08:00", "3/10/17 10:00"],
            ["3/10/17 19:00", "3/10/17 20:00"],
        ]

        test_periods = [
            ["3/10/17 11:00", "3/10/17 15:00"],
            ["3/10/17 00:00", "3/10/17 05:00"],
        ]

        train, test = ByIntervalsSplitter(
            "date", test_intervals=test_periods, train_intervals=train_periods,
        ).split(simple_data)

        assert len(train) == 3
        assert len(test) == 8
        assert pd.to_datetime("3/10/17 6:00") not in simple_data["date"]

    def test_string_representation(self):
        train_periods = [
            ["3/10/18 00:00", "3/10/18 10:00"],  # No effect
            ["3/10/17 08:00", "3/10/17 10:00"],
            ["3/10/17 19:00", "3/10/17 20:00"],
        ]
        test_periods = [
            ["3/10/17 11:00", "3/10/17 15:00"],
            ["3/10/17 00:00", "3/10/17 05:00"],
        ]
        splitter = ByIntervalsSplitter(
            "date",
            test_intervals=test_periods,
            train_intervals=train_periods,
        )
        assert isinstance(repr(splitter), str)
        assert "ByIntervalsSplitter" in repr(splitter)


class TestSplitByLastWindow(object):
    @pytest.mark.parametrize("freq", ["S", "Min", "H", "D", "M", "Y"])
    def test_raises_on_missing_column(self, simple_data, freq):
        with pytest.raises(ValueError, match="column is missing"):
            ByLastWindowSplitter("missing_column", freq).split(simple_data)

    @pytest.mark.parametrize("freq,is_valid", [
        ("1H", True),
        ("1W", True),
        ("1xBet", False),
        ("<invalid>", False),
        ("...", False),
        ("1abc2W", False),
    ])
    def test_raises_if_invalid_frequency(
        self,
        freq: str,
        is_valid: bool,
    ):
        if is_valid:
            ByLastWindowSplitter("date", freq)
        else:
            with pytest.raises(ValueError, match="Invalid frequency"):
                ByLastWindowSplitter("date", freq)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("freq", ["S", "Min", "H", "D", "M", "Y"])
    def test_last_n_freq(self, simple_data, n, freq):
        if freq in {"S", "Min", "H"}:
            simple_data["date"] = [ts.replace(hour=0) for ts in simple_data["date"]]

        offsets = pd.Series([
            pd.tseries.frequencies.to_offset(f"{i}{freq}")
            for i in range(len(simple_data))
        ])
        simple_data["date"] += offsets
        freq = f"{n}{freq}"
        before, after = ByLastWindowSplitter("date", freq).split(simple_data)
        assert len(after) == n
        assert len(before) == len(simple_data) - n
        assert not set(before["date"]) & set(after["date"])

    def test_string_representation(self):
        splitter = ByLastWindowSplitter("date", "1H")
        assert isinstance(repr(splitter), str)
        assert "ByLastWindowSplitter" in repr(splitter)


class TestSplitBySequentialWindow(object):

    def test_raises_on_missing_column(self, simple_data):
        with pytest.raises(ValueError, match="column is missing"):
            BySequentialSplitter("missing_column", "5H", "3H").split(simple_data)

    @pytest.mark.parametrize(
        "block_freq,train_freq", [("1S", "1H"), ("1W", "2W"), ("10Y", "100Y")],
    )
    def test_raises_train_freq_larger(self, simple_data, block_freq, train_freq):
        with pytest.raises(ValueError, match="greater than"):
            BySequentialSplitter("date", block_freq, train_freq).split(simple_data)

    def test_same_unit(self, simple_data):
        train, test = BySequentialSplitter("date", "5H", "3H").split(simple_data)

        assert len(train) + len(test) == len(simple_data)
        assert not set(train["date"]) & set(test["date"])
        train_boundaries = [
            (pd.to_datetime(start_date), pd.to_datetime(end_date))
            for start_date, end_date in (
                ("3/10/17 1:00", "3/10/17 3:00"),
                ("3/10/17 6:00", "3/10/17 8:00"),
                ("3/10/17 11:00", "3/10/17 13:00"),
                ("3/10/17 16:00", "3/10/17 18:00"),
            )
        ]
        mask = np.zeros_like(train["date"]).astype(bool)
        for bound in train_boundaries:
            mask |= (
                (bound[0] <= train["date"]) & (train["date"] <= bound[1])
            )

        assert all(mask)

    def test_different_units(self, simple_data):
        offset = pd.Series([pd.to_timedelta(f"{i}M") for i in range(len(simple_data))])
        simple_data["date"] += offset
        simple_data["date"] = [ts.replace(hour=0) for ts in simple_data["date"]]

        train, test = BySequentialSplitter("date", "5Min", "180S").split(simple_data)

        assert len(train) + len(test) == len(simple_data)
        assert not set(train["date"]) & set(test["date"])

        train_boundaries = [
            (pd.to_datetime(start_date), pd.to_datetime(end_date))
            for start_date, end_date in (
                ("3/10/17 00:00", "3/10/17 00:03"),
                ("3/10/17 00:05", "3/10/17 00:08"),
                ("3/10/17 00:10", "3/10/17 00:13"),
                ("3/10/17 00:15", "3/10/17 00:18"),
            )
        ]
        mask = np.zeros_like(train["date"]).astype(bool)
        for bound in train_boundaries:
            mask |= (
                (bound[0] <= train["date"]) & (train["date"] < bound[1])
            )
        assert all(mask)

    def test_string_representation(self):
        splitter = BySequentialSplitter("date", "5Min", "180S")
        assert isinstance(repr(splitter), str)
        assert "BySequentialSplitter" in repr(splitter)


class TestSplitByColumnValue(object):
    """Tests for ``ByColumnValueSplitter`` class."""

    country_labels = ("USA", "CAN", "MEX")
    company_labels = ("McKinsey", "QB")

    @pytest.fixture(scope="class")
    def sample_df(self) -> pd.DataFrame:
        """Sample dataframe to be used in testing this splitter."""
        rng = np.random.default_rng(42)
        n_rows = 100
        data = pd.DataFrame({
            "country": rng.choice(self.country_labels, n_rows),
            "company": rng.choice(self.company_labels, n_rows),
            "value": rng.random(n_rows),
        })
        return data

    @pytest.mark.parametrize("values_for_test", [
        ("USA",),
        ("CAN",),
        ("USA", "MEX"),
        ("MEX", "USA"),
        ("CAN", "USA"),
        ["CAN", "USA"],
        {"CAN", "USA"},
    ])
    def test_standard_case(
        self,
        sample_df: pd.DataFrame,
        values_for_test: tp.Iterable[str],
    ) -> None:
        """
        Test the split in standard case: a valid subset of column values is provided.
        """

        # Initialize splitter
        splitter = ByColumnValueSplitter(
            column_name="country",
            values_for_test=values_for_test,
        )
        # Split data
        train_data, test_data = splitter.split(sample_df)

        # Assert that the target column in the test set has just that single value
        obtained_test_labels = set(test_data[splitter.column_name])
        expected_test_labels = set(values_for_test)
        assert obtained_test_labels == expected_test_labels

        # Assert that the target column in the train set has all values but that one
        obtained_train_labels = set(train_data[splitter.column_name])
        expected_train_labels = set(self.country_labels) - set(values_for_test)
        assert obtained_train_labels == expected_train_labels

    @pytest.mark.parametrize("column_name,should_raise", [
        ("country", False),
        ("non_existent_column", True),
    ])
    def test_split_raises_on_invalid_column_name(
        self,
        column_name: str,
        should_raise: bool,
        sample_df: pd.DataFrame,
    ) -> None:
        """
        Test that ``.split()`` raises an error if a column doesn't exist in the df.
        """
        splitter = ByColumnValueSplitter(
            column_name=column_name,
            values_for_test=("USA",),
        )
        if should_raise:
            with pytest.raises(KeyError):
                splitter.split(sample_df)
        else:
            splitter.split(sample_df)

    @pytest.mark.parametrize("column_name,should_raise", [
        ("country", False),
        ("non_existent_column", False),
        ("", True),
    ])
    def test_init_raises_on_empty_column_name(
        self,
        column_name: str,
        should_raise: bool,
    ) -> None:
        """Test that the splitter cannot be created if a column name is empty."""
        if should_raise:
            with pytest.raises(ValueError):
                ByColumnValueSplitter(
                    column_name=column_name,
                    values_for_test=("value",),
                )
        else:
            ByColumnValueSplitter(
                column_name=column_name,
                values_for_test=("value",),
            )

    @pytest.mark.parametrize("values_for_test,should_raise", [
        (("USA", "MEX"), False),
        (("USA",), False),
        (("Ok that I am a weird value",), False),
        ((1, 2), False),
        (None, True),
        (list(), True),
    ])
    def test_init_raises_on_invalid_test_values(
        self,
        values_for_test: tp.Any,
        should_raise: bool,
    ) -> None:
        """Test that the splitter cannot be created if test values are invalid."""
        if should_raise:
            with pytest.raises(ValueError):
                ByColumnValueSplitter(
                    column_name="valid",
                    values_for_test=values_for_test,
                )
        else:
            ByColumnValueSplitter(
                column_name="valid",
                values_for_test=values_for_test,
            )

    def test_string_representation(self):
        """Test that the string representation is as expected."""
        splitter = ByColumnValueSplitter(
            column_name="country",
            values_for_test=("USA", "CAN"),
        )
        representation = repr(splitter)
        assert isinstance(representation, str)
        expected_substrings = ("ByColumnValueSplitter", "country")
        assert all(
            substring in representation
            for substring in expected_substrings
        )

    def test_raises_when_no_data_assigned_to_test(
        self,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that the splitter raises value error if test dataset is empty."""
        splitter = ByColumnValueSplitter(
            column_name="country",
            values_for_test=("GER",),
        )
        with pytest.raises(ValueError):
            splitter.split(sample_df)

    def test_can_be_initialized_from_repr(self) -> None:
        """Test that string representation allows initializing an object from it."""
        existing_splitter = ByColumnValueSplitter(
            column_name="country",
            values_for_test=("GER", "USA"),
        )
        existing_repr = repr(existing_splitter)
        new_splitter = eval(existing_repr)  # noqa: S307
        assert isinstance(new_splitter, ByColumnValueSplitter)
