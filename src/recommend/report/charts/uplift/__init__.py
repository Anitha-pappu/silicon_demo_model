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

from ._data_filtering import (
    clean_data_filtering,
    plot_implementation_ratio,
    plot_performance_summary,
    plot_performance_summary_by_tag,
)
from ._impact_deep_dive import (
    plot_gap_to_optimal,
    plot_impact_timeline,
    plot_impact_timeline_cumulative,
    plot_impact_waterfall,
    plot_objective_values,
)
from ._impact_significance import (
    get_text_for_test_results,
    plot_baseline_bias_histogram,
    plot_uplift_histogram,
    plot_uplifts_and_baseline_error_histogram,
)
from ._overview import plot_impact_summary
