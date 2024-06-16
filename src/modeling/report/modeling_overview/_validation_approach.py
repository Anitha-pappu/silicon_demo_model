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

from modeling.report.charts import (
    get_split_details,
    plot_consecutive_validation_periods,
    plot_feature_comparison_for_train_test,
    plot_validation_representation,
)
from reporting.charts.primitives import plot_correlation
from reporting.rendering.identifiers import Code, SectionHeader, Text
from reporting.rendering.types import TRenderingAgnosticDict

from . import _section_description as section_descriptions  # noqa: WPS450

_DEFAULT_PLOT_TITLE_SIZE = 20


def plot_validation_approach(
    target: str,
    features: tp.List[str],
    timestamp_column: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> TRenderingAgnosticDict:
    """
    Returns: visual representation of validation which contains
        * Visual Representation
        * Consecutive Periods List
        * Correlation heatmaps for train and test datasets
        * Feature on splits comparison
    """
    consecutive_periods_timestamps = plot_consecutive_validation_periods(
        train_data, test_data, timestamp_column,
    )
    consecutive_validation_periods = {
        name: Code(code=periods, code_formatter=None, language="js")
        for name, periods in consecutive_periods_timestamps.items()
    }
    return {
        SectionHeader(
            header_text="Visual Representation Of Split",
            description=section_descriptions.VISUAL_REPRESENTATION_DESCRIPTION,
        ): [
            Text(
                text=get_split_details(train_data, test_data),
                title="Split Details",
                title_size=_DEFAULT_PLOT_TITLE_SIZE,
            ),
            plot_validation_representation(
                train_data, test_data, target, timestamp_column,
            ),
        ],
        SectionHeader(
            header_text="Consecutive Periods",
            description=section_descriptions.CONSECUTIVE_PERIODS_DESCRIPTION,
        ): consecutive_validation_periods,
        SectionHeader(
            header_text="Features correlation heatmap",
            description=section_descriptions.FEATURE_CORRELATION_DESCRIPTION,
        ): {
            "Train": plot_correlation(
                data=train_data,
                mask="lower",
            ),
            "Test": plot_correlation(
                data=test_data,
                mask="lower",
            ),
        },
        SectionHeader(
            header_text="Train vs. Test Comparisons",
            description=section_descriptions.TRAIN_TEST_COMPARISON_DESCRIPTION,
        ): {
            "Target": plot_feature_comparison_for_train_test(
                train_data, test_data, [target],
            ),
            **plot_feature_comparison_for_train_test(
                train_data, test_data, features,
            ),
        },
    }
