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
Protocols used to indicate content should be rendered in a specific way
"""

from ._code import Code
from ._section_header import (
    SectionHeader,
    remove_section_description_from_structure,
)
from ._table import Table
from ._text import Text

TIdentifier = Code | SectionHeader | Text | Table
SUPPORTED_IDENTIFIERS = (Code, Table, Text, SectionHeader)
