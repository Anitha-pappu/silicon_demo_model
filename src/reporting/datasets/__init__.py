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
Contains mock datasets used in tutorials/tests for showcasing package functionality
"""

from ._datasets import (
    get_batch_meta_with_features,
    get_mill_data,
    get_sensor_data_batched_phased,
    get_throughput_data,
)
from ._report_structures import (
    create_advanced_report_structure,
    create_basic_report_structure,
    create_multilevel_report_structure,
)
from .io_utils import DATA_DIR
