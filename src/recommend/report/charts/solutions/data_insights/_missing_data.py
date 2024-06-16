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

import pandas as pd
import plotly.graph_objects as go

from reporting.types import TColumn

_TITLE_SIZE = 20
_CBAR_LEN = 0.2
_CBAR_THICKNESS = 15


def plot_missing_data_heatmap(
    data: pd.DataFrame,
    timestamp_column: TColumn = "timestamp",
) -> go.Figure | str:
    is_missing = data.isnull().T

    if not is_missing.any().any():
        return "No missing data found"

    columns = data.columns.copy()
    if columns.name is not None:
        columns.name = None

    fig = go.Figure()
    fig.add_heatmap(
        z=is_missing.astype(int),
        customdata=is_missing,
        x=data[timestamp_column],
        y=columns,
        hovertemplate="%{x}<br>missing: %{customdata} <extra></extra>",
        colorscale="Greys_r",
        colorbar=dict(
            tickvals=[0.1, 0.9],
            ticktext=["Not Missing", "Missing"],
            y=1,
            yanchor="top",
            yref="paper",
            len=_CBAR_LEN,
            thickness=_CBAR_THICKNESS,
        ),
        zmin=0,
        zmax=1,
    )
    return fig


def get_missing_data_summary(
    data: pd.DataFrame,
    reference_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Creates dataframe with missing data stats for the ``data`` and ``reference``

    Summary has two columns:
    ["nans in data to optimize, %", "nans in reference data, %"],
    second column is added only if ``reference_data`` is not None;

    Summary is indexed over union of columns ``data.columns``
    and ``reference_data.columns``

    Args:
        data: data to get missing summary for
        reference_data: if provided, one more column with reference data's missing
            values stats is returned

    Returns: summary dataframe

    Examples::
        >>> df = pd.DataFrame({'col_one': [0, np.nan], 'col_two': [1, 2]})
        >>> ref_df = pd.DataFrame({'col_one': [3, 4], 'col_two': [5, np.nan]})
        >>> summary = get_missing_data_summary(df, ref_df)
        |           |   nans in data to optimize, % |
        |:----------|------------------------------:|
        | col_one   |                             0 |
        | col_two   |                            50 |

    """
    is_missing = data.isnull().T

    is_missing_by_column = is_missing.sum(axis=1) / data.shape[0] * 100

    opt_data_column = "nans in data to optimize, %"
    is_missing_by_column_summary = pd.DataFrame({opt_data_column: is_missing_by_column})

    if reference_data is not None:
        reference_is_missing = reference_data.isnull()
        reference_is_missing_by_column = (
            reference_is_missing.sum(axis=0) / reference_data.shape[0] * 100
        )
        is_missing_by_column_summary["nans in reference data, %"] = (
            reference_is_missing_by_column
        )
    return is_missing_by_column_summary
