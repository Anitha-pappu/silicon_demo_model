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

"""Global reporting package settings"""

import typing as tp
from functools import wraps

import matplotlib

# _P - parameters, _R - result
_P = tp.ParamSpec("_P")  # noqa: WPS111
_R = tp.TypeVar("_R")  # noqa: WPS111

COLORS = (
    "#9267D3",
    "#30C0A5",
    "#C030B0",
    "#C09D30",
    "#3062C0",
    "#00B4FF",
    "#FF9222",
    "#3949AB",
    "#FF5267",
    "#08BDBA",
    "#FDC935",
    "#689F38",
)

DEFAULT_BACKEND = "agg"


def with_default_pyplot_backend(function: tp.Callable[_P, _R]) -> tp.Callable[_P, _R]:
    """
    Acts as a context manager for function:
    switches the backend to ``reporting.config.DEFAULT_BACKEND`` before function call
    and then back to initial backend.

    So ``result = function(*args, **kwargs)`` call equals to::

        backend = matplotlib.get_backend()
        matplotlib.use(DEFAULT_BACKEND)
        result = function(*args, **kwargs)
        matplotlib.use(backend)
    """

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:  # noqa: WPS430
        backend = matplotlib.get_backend()
        matplotlib.use(DEFAULT_BACKEND)
        result = function(*args, **kwargs)  # noqa: WPS110
        matplotlib.use(backend)
        return result

    return wrapped
