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
Contains types for rendering-agnostic content:

- identifiers, which are used to provide info that a content element should be treated
  in a special way
- rendering-agnostic constants, conveniently grouping relevant types
- rendering-agnostic types, describing the types of rendering-agnostic structures that
  can be rendered by any of the renderers

Identifiers
-----------

The identifiers are classes that enable identifying a content element as an element that
has to be rendered in a special way (e.g. identifying that a string should be rendered
as code).
Each identifier is recognized by all the renderers, and each renderer is capable of
handling all the identifiers in the appropriate way.

These identifiers are

- ``ReportElementIdentifier``
  - base class for all the identifiers
- ``Text``
  - identifies that a given string should be treated as text
- ``Code``
  - identifies that a given string should be treated as code
- ``Table``
  - to identify some objects as table
  - allows storing information to customize how the table will look like when rendered

Each identifier has attributes that can store information, which will be available at
rendering time and will allow for customizing how the content will look like when
rendered.
Each renderer will handle these attribute in its own way. Some attributes might only be
relevant for one or some of the renderers.

All the currently defined identifiers at the moment have a ``_repr_html_``, which
enables a nice visualization when displayed in jupyter notebooks. This is not (at the
moment) a requirement (but might become one in the future).

Rendering-agnostic constants
----------------------------

The following constants are provided

- ``SUPPORTED_COMMON_TYPES``
  - common python types that are guaranteed to be rendered by all the renderers when
  used in a report structure
  - these types are ``pd.DataFrame``, ``matplotlib.figure.Figure``, ``go.Figure``
  - each renderer will convert these supported common types to the appropriate
  renderable type
- ``SUPPORTED_IDENTIFIERS``
  - a tuple of the identifiers provided by the package
  - these types are ``Code``, ``Table``
- ``SUPPORTED_RENDERING_AGNOSTIC_TYPES``
  - a tuple of all the rendering-agnostic types that are guaranteed to be rendered by
  all the renderers when used in a report structure
  - consists of the ``SUPPORTED_IDENTIFIERS`` and the ``SUPPORTED_COMMON_TYPES``

Rendering-agnostic types
------------------------

- ``TRenderingAgnosticDict``
  - type of nested dict that can be rendered into a report by all the renderers
- ``TRenderingAgnosticDictKey``
  - type of the keys of ``TRenderingAgnosticDict``
- ``TRenderingAgnosticContent``
  - type of the single final-level content elements of ``TRenderingAgnosticDict``
  - consist of the types in ``SUPPORTED_RENDERING_AGNOSTIC_TYPES``
- ``TRenderingAgnosticDictValue``
  - type of final (non-nested) content of the section of ``TRenderingAgnosticDict``
  - consists of ``TRenderingAgnosticDictValue``,``TRenderingAgnosticDictKey``  and list
  of these
"""


import typing as tp

import pandas as pd
from matplotlib.figure import Figure as MatplotlibFigure
from plotly.graph_objs import Figure as PlotlyFigure

from . import identifiers
from .common_protocols import (
    MatplotlibLike,
    PlotlyLike,
    ReprHtmlCompatible,
    SavefigCompatible,
    ToHtmlCompatible,
    ToImageCompatible,
)

SUPPORTED_COMMON_TYPES = (MatplotlibFigure, PlotlyFigure, pd.DataFrame)
SUPPORTED_RENDERING_AGNOSTIC_TYPES = (
    identifiers.SUPPORTED_IDENTIFIERS,
    *SUPPORTED_COMMON_TYPES,
)
TRenderingAgnosticContent = tp.Union[
    MatplotlibFigure,
    PlotlyFigure,
    pd.DataFrame,
    identifiers.Code,
    identifiers.Table,
    identifiers.Text,
]
TRenderingAgnosticDictKey = tp.Union[str, int, identifiers.SectionHeader]
TRenderingAgnosticDictValue = tp.Union[
    TRenderingAgnosticDictKey,
    TRenderingAgnosticContent,
    tp.List[TRenderingAgnosticContent],
]
TRenderingAgnosticDict = tp.Dict[
    TRenderingAgnosticDictKey, tp.Union[
        TRenderingAgnosticDictValue, "TRenderingAgnosticDict",
    ],
]
