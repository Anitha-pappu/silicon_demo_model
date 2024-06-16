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

from .implementation_methods import (
    deviation_implementation_status,
    progress_implementation_status,
)

IMPLEMENTATION_STATUS_METHODS: tp.Dict[  # noqa: WPS407
    str, tp.Callable[..., pd.DataFrame],
] = {
    "deviation": deviation_implementation_status,
    "progress": progress_implementation_status,
}


class TMethod(tp.Protocol):
    def __call__(
        self,
        implementation_data: pd.DataFrame,
        **kwargs: tp.Any,
    ) -> pd.DataFrame:
        """
        Signature of functions that can be passed as method in
        ``calculate_implementation_status``, where ``kwargs`` includes parameters
        required to calculate the implementation status.

        Expect to receive nans on ``current_value`` column of ``implementation_data``.
        """


def calculate_implementation_status(
    implementation_data: pd.DataFrame,
    method: tp.Literal["deviation", "progress"] | TMethod,
    **kwargs: tp.Any,
) -> pd.DataFrame:
    """
    Calculates the implementation percentage (between 0 and 100) for each recommendation
    and run id based on the method selected.

    Args:
        implementation_data: Data ready for implementation status calculations.
        method: Method to use for implementation status calculations. It can be a name
            that maps to a function through ``IMPLEMENTATION_STATUS_METHODS``. It can
            also be a function with ``TMethod```signature
        **kwargs: Additional kwargs that are passed to ``method``.

    Returns:
        Implementation status per feature and run.
    """
    supported_methods = ", ".join(IMPLEMENTATION_STATUS_METHODS.keys())
    if isinstance(method, str):
        if method not in IMPLEMENTATION_STATUS_METHODS:
            raise ValueError(
                f"Provided implementation status method {method} is not supported:"
                f" supported methods are {supported_methods}",
            )
        implementation_func = IMPLEMENTATION_STATUS_METHODS[method]
        return implementation_func(implementation_data, **kwargs)

    elif callable(method):
        return method(implementation_data, **kwargs)

    raise ValueError(
        f"Method must be {supported_methods} or a function that satisfies"
        " ``TMethod`` protocol.",
    )
