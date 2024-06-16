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

import re
import textwrap
import typing as tp


def wrap_string(
    string: tp.Optional[str],
    max_length: int = 25,
    wrap_by: str = "<br>",
    replace_with_hyphens: tp.Iterable[str] = ("_",),
) -> tp.Optional[str]:
    """
    Wraps a string to a maximum length and replaces text patterns in
    ``replace_with_hyphens`` with a space.

    Args:
        string: string to wrap
        max_length: maximum length of the string
        wrap_by: string to wrap by
        replace_with_hyphens: patterns to replace with a space

    Returns:
        wrapped string
    """
    if string is None or len(string) <= max_length:
        return string

    # we will use `textwrap.wrap` to perform wrapping on mutated string
    # and then use that to collect wrapped from initial string
    initial_string = string
    for replace in replace_with_hyphens:
        string = string.replace(replace, " ")
    current_index = 0
    wrapping = []
    for wrapped_seq in textwrap.wrap(string, max_length):
        wrapping.append(
            initial_string[current_index: current_index + len(wrapped_seq)],
        )
        current_index += len(wrapped_seq) + 1
    return wrap_by.join(wrapping)


def get_num_lines_for_str(string: str) -> int:
    """
    Returns the number of lines in a string.

    Args:
        string: string to count lines in

    Returns:
        number of lines in the string
    """
    return len(re.findall(pattern="<br>", string=string))
