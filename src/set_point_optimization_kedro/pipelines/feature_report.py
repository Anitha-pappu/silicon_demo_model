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

import warnings

from kedro.pipeline import Pipeline, node

from reporting.reports import plot_feature_overviews

warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_feature_report() -> Pipeline:
    return Pipeline(
        [
            node(
                plot_feature_overviews,
                inputs={
                    "data": "model_input_data",
                    "features": "params:feature_report.features",
                    "timestamp": "params:feature_report.datetime_column",
                    "target": "params:feature_report.target_column",
                    "title": "params:feature_report.title",
                },
                outputs="feature_report",
                name="plot_features",
            ),
        ],
    ).tag("feature_report")
