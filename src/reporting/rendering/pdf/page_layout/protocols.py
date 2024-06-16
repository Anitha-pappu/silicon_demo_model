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

# TODO: Check what is written here
"""Defines protocols for functions used in the customization of the page.

As of now they are just defined as generic callables, while they require specific inputs
and outputs

- after-page functions

    - need to take the doc as input (and additional inputs if needed)
    - e.g. ``fn(doc: SimpleDocTemplate, **kwargs)``
- after-flowable functions

    - need to take the doc and the flowable as input (and
additional inputs if needed)
    - e.g. ``fn(doc: SimpleDocTemplate, flowable: Flowable, **kwargs)``

Prepares the types here for now, so they can be used to identify what the functions are
meant to do in the code.
We should check how to define callable types correctly before changing the type
definition here.

These function should not returen anything (double chek if this is true though).

Both functions should support kwargs, or at least support the arguments ``page_layout``
and ``content_style``, since in the current implementation these kwargs will be passed
to these functions.

``TStyledAfterPage`` and ``TStyledAfterFlowable`` describe the signature of the
functions above, after the style has been applied. They do not support kwargs and are
meant to be called while rendering the pdf. The signatures are
    - ``TStyledAfterPage``: ``fn(doc: SimpleDocTemplate)``
    - ``TStyledAfterFlowable``:  ``fn(doc: SimpleDocTemplate, flowable: Flowable)``

``TStylableCallable`` is a function that should accept the following as kwargs:
- ``page_layout``
- ``content_style``
"""

import typing as tp

TAfterPage = tp.Callable[..., None]
TAfterFlowable = tp.Callable[..., None]
TStyledAfterPage = tp.Callable[..., None]
TStyledAfterFlowable = tp.Callable[..., None]
TStylableCallable = tp.Callable[..., None]
