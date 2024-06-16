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
Split module.
"""
from ._splitters.base_splitter import SplitterBase  # noqa: WPS436
from ._splitters.by_column_value import ByColumnValueSplitter  # noqa: WPS436
from ._splitters.by_date_splitter import ByDateSplitter  # noqa: WPS436
from ._splitters.by_frac_splitter import ByFracSplitter  # noqa: WPS436
from ._splitters.by_intervals_splitter import (  # noqa: WPS436
    ByIntervalsSplitter,
)
from ._splitters.by_last_window import ByLastWindowSplitter  # noqa: WPS436
from ._splitters.by_sequential_splitter import (  # noqa: WPS436
    BySequentialSplitter,
)
from ._splitters.functional import create_splitter, split_data  # noqa: WPS436
