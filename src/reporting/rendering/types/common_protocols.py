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

import io
import typing as tp


@tp.runtime_checkable
class SavefigCompatible(tp.Protocol):  # todo: move out to rendering level
    def savefig(
        self,
        fname: tp.BinaryIO,
        *args: tp.Any,
        format: tp.Optional[str] = None,  # noqa: WPS125
        **kwargs: tp.Any,
    ) -> None:
        """
        Saves the current figure to ``fname`` buffer

        Args:
            fname: binary file-like object to save fig to
            format: The file format must support 'png'
        """


@tp.runtime_checkable
class ToHtmlCompatible(tp.Protocol):
    def to_html(self, *args: tp.Optional[tp.Any], **kwargs: tp.Optional[tp.Any]) -> str:
        """Produces html representation for report rendering purposes"""


@tp.runtime_checkable
class ReprHtmlCompatible(tp.Protocol):
    """Represents protocol for objects with html repr in jupyter"""

    def _repr_html_(self) -> str:
        """Produces jupyter compatible html representation"""


@tp.runtime_checkable
class ToImageCompatible(tp.Protocol):

    def to_image(self, *args, **kwargs) -> io.BytesIO:
        """ A method that returns an image as bytes object"""


class PlotlyLike(ReprHtmlCompatible, ToHtmlCompatible, tp.Protocol):
    """
    Protocol for plotly-like figures.
    We use two parent interfaces to show their properties:
    jupyter and html compatibility.
    """


class MatplotlibLike(ReprHtmlCompatible, SavefigCompatible, tp.Protocol):
    """
    Protocol for matplotlib-like figures.
    We use two parent interfaces to show their properties:
    jupyter and image compatibility.
    """
