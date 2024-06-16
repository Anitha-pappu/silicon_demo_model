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

import typing as tp
from abc import ABC, abstractmethod
from enum import Enum

from reportlab.platypus import Flowable

from reporting.rendering.pdf.content_styles import ContentStyleBase

TUid = tp.Tuple[int, ...]


class ElementType(Enum):
    """Types of elements in the final document."""
    HEADER = "header"
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    FLOWABLE = "flowable"
    TABLE = "table"
    UNSPECIFIED = None


class PdfRenderableElement(ABC):
    """An abstract base class for pdf-renderable elements"""

    def __init__(self, uid: tp.Optional[TUid] = None):
        """Initialization of common properties.

        Notes:
            The common properties are
            - ``uid``: a unique-identifier tuple, assumed containing integers.
            - ``uid_string``: the ``uid`` made into a string with standard formatting
            (this means separating the int in ``uid`` with dots ``.``)

        """
        self.update_uid(uid=uid)

    @abstractmethod
    def to_reportlab(self, *args, **kwargs) -> Flowable:
        """Produces a reportlab ``Flowable``"""

    # TODO: Should be ``renderable_report`` instead of ``doc``
    #  Because it uses ``renderable_report``, not ``renderable_report._doc``
    #  Should be renamed everywhere!!!
    def to_reportlab_from_renderable_report(
        self,
        renderable_report: PdfRenderableReportProtocol,
    ) -> Flowable:
        """Converts to ``Flowable`` getting kwargs from ``renderable_report``"""
        kwargs = self._get_kwargs_to_reportlab(renderable_report)
        return self.to_reportlab(**kwargs)

    def update_uid(self, uid: TUid):
        """Update uid in both raw and string format"""
        self._uid = uid
        self._uid_string = _format_uid_as_string(self._uid)

    def get_info(self) -> ElementInfo:
        """Gets info that can be used to customize the ``Flowable``"""
        element_info_class = self._element_info_class
        return element_info_class(element=self)

    @property
    @abstractmethod
    def element_type(self) -> ElementType:
        """A property returning the element type"""

    @property
    def uid_string(self) -> tp.Union[str, None]:
        """Unique identifier as a string"""
        return self._uid_string

    @property
    def uid(self) -> tp.Union[TUid, None]:
        """Unique identifier as a tuple"""
        return self._uid

    def _basic_repr(self, input_element_repr: str) -> str:
        """Create a basic string that can be used to generate ``__repr__`` for the child
        classes, provided that the child class only has ``uid`` as optional input.

        ``input_element_repr`` is a repr for the input element of the child class.
        It is meant to includes also the specific name of the parameter used in the
        child class e.g. ``text='The input text'``, not just ``'The input text'``.
        """
        class_name = self.__class__.__name__
        uid_ = "" if self._uid is None else f" uid={self._uid}"
        return (
            f"{class_name}("
            f"{input_element_repr},"
            f"{uid_}"
            ")"
        )

    @property
    @abstractmethod
    def _element_info_class(self) -> tp.Type[ElementInfo]:
        """A property returning the appropriate elment info class"""

    @abstractmethod
    def _get_kwargs_to_reportlab(
        self,
        renderable_report: tp.Optional[PdfRenderableReportProtocol],
    ) -> tp.Dict[str, tp.Any]:
        """Returns the kwargs that can be passed to the ``to_reportlab`` method"""


def _format_uid_as_string(uid: tp.Union[TUid, None]):
    """Creates a uid string from a uid tuple.

    Notes:
        ``uid`` is a unique identifier tuple. The tuple is assumed to contain integers.
        The unique id string is make by concatenating these integers, separating them
        with a ``.``.
    """
    if uid is None:
        return None
    return ".".join([str(uid_element) for uid_element in uid])


class ElementInfo(ABC):
    """Define info produced by a renderable.

    Notes:
        These info are intended to be produced by a ``PdfRenderableElement``, and stored
        in an enhanced flowable object, so they can be used for when building the pdf.
    """

    def __init__(self, element: PdfRenderableElement) -> None:
        self.uid = element.uid
        self.uid_string = element.uid_string
        self.element_type = element.element_type
        self._store_info(element=element)

    @abstractmethod
    def _store_info(self, element: PdfRenderableElement) -> None:
        """Stores appropriate info for the given element"""


@tp.runtime_checkable
class PdfRenderableReportProtocol(tp.Protocol):
    content_style: ContentStyleBase
    drawable_width: tp.Union[int, float]
    drawable_height: tp.Union[int, float]
