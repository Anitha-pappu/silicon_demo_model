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

from dataclasses import dataclass


# TODO: Move this object out from identifiers subpackage because
#  it creates a confusion between objects involves in
#  rendering process and rendarable objects
#  OAI-2717
@dataclass(frozen=True)
class SectionHeader(object):  # todo: add missing docstring
    header_text: str
    description: str | None = None

    def __hash__(self) -> int:
        # is required by pdf rendering module
        return hash(f"{self.header_text}{self.description}")
