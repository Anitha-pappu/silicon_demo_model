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

from __future__ import annotations

import io
import typing as tp

import plotly.graph_objects as go
from reportlab.platypus import Image

from ._base import (  # noqa: WPS436
    ElementInfo,
    ElementType,
    PdfRenderableElement,
    PdfRenderableReportProtocol,
    TUid,
)

TNum = tp.Union[int, float]


class PdfRenderableImage(PdfRenderableElement):
    """
    A connector that makes plotly and matplotlib images compatible with pdf rendering.

    More in general this should be a unified way to deal with sopported images for the
    pdf renderer
    """

    def __init__(self, image, uid: tp.Optional[TUid] = None):
        super().__init__(uid=uid)
        self._image = image
        self.to_image = self._find_suitable_to_image()

    def to_reportlab(self, width: TNum, height: TNum):
        """ For images, the values should be:
            width=doc.drawable_width,
            height=doc.drawable_height,

            For now increasing the resolution of plotly ``go.Figure``s by default
        """
        img_bytes = io.BytesIO()

        # TODO: See if there is a more reasonable way to do this
        #  Consider passing the scale as an optional param to the `to_reportlab method``
        kwargs = {"scale": 2} if isinstance(self._image, go.Figure) else {}
        # TODO: Using width and height here? Investigate if it would make sense or not
        img_bytes.write(self.to_image(format="png", **kwargs))
        return Image(img_bytes, width=width, height=height, kind="proportional")

    @property
    def element_type(self) -> ElementType:
        return ElementType.IMAGE

    def __repr__(self) -> str:
        image_repr = repr(self._image)
        input_element_repr = f"image={image_repr}"
        return self._basic_repr(input_element_repr=input_element_repr)

    @property
    def _element_info_class(self) -> tp.Type[ImageElementInfo]:
        return ImageElementInfo

    def _find_suitable_to_image(self):
        to_image = getattr(self._image, "to_image", None)
        if to_image is not None:
            return self._image.to_image
        savefig = getattr(self._image, "savefig", None)
        if savefig is not None:
            return self._to_image_from_savefig
        input_object_type = type(self._image)
        raise NotImplementedError(
            f"Input object fo type {input_object_type} does not have any implemented "
            "method to be image compatible",
        )

    # Using the name "format" and "noqa: WPS125" because the method is called format in
    #  the original `savefig` method in `matplotlib.pyplot`
    def _to_image_from_savefig(self, format="png"):  # noqa: WPS125
        """ Assumes ``self._image`` has a ``savefig`` method """
        buffer = io.BytesIO()
        self._image.savefig(buffer, format=format)
        image_bytes = buffer.getvalue()
        return image_bytes  # noqa: WPS331  # Naming makes meaning clearer

    def _get_kwargs_to_reportlab(
        self,
        renderable_report: PdfRenderableReportProtocol,
    ) -> tp.Dict[str, tp.Any]:
        return dict(
            width=renderable_report.drawable_width,
            height=renderable_report.drawable_height,
        )


class ImageElementInfo(ElementInfo):
    """Store info about a flowable image that can be used when building the pdf"""

    def _store_info(self, element: PdfRenderableImage):
        """ Does not store any additional info"""
