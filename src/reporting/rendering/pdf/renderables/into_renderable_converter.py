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

""" Utils to identify and convert objects into the appropriate pdf-renderable objects"""
import typing as tp

from reporting.rendering import identifiers
from reporting.rendering.types import TRenderingAgnosticDict

from ._base import PdfRenderableElement, TUid  # noqa: WPS436
from ._header import PdfRenderableHeader  # noqa: WPS436
from ._into_renderable_conversion_strategy import (  # noqa: WPS436
    APPROPRIATE_CONVERSION_CLASS_SEARCH_STRATEGY,
)
from .protocols import TPdfRenderableContent, TReportStructureHeader

TPdfRenderableStory = tp.List[PdfRenderableElement]


_TTreeCoords = tp.Tuple[int, ...]
_TVarOptionalTreeCoords = tp.TypeVar("_TVarOptionalTreeCoords", _TTreeCoords, None)


def convert_recursively_into_pdf_renderable(
    section_data: TRenderingAgnosticDict,
    header_level: int,
) -> TPdfRenderableStory:
    """
    Produce a list of pdf-renderable.

    ``section_data`` is a rendering-agnostic dict that will be recursively parsed in
    order to generate a renderable story.

    Parses ``section_data`` recursively and generates ``renderable_objects``, a list of
    elements, in the order with which they will appear in the final pdf.
    The objects in ``renderable_objects`` are in pdf-renderable format.

    ``header_level`` is used to know at which level of the hierarchy the objects are.
    This can be used to set object properties according to their location in the
    document hierarchy (e.g. the style of the header, depending on how important the
    header is).

    Returns ``renderable_objects``, which is a list of pdf-renderable objects.
    """
    renderable_objects = []
    _handle_dict(
        renderable_objects=renderable_objects,
        section_data=section_data,
        header_level=header_level,
        starting_tree_coords=(0, ),
    )
    return renderable_objects


def _handle_section_content(
    renderable_objects: TPdfRenderableStory,
    section_content,
    header_level,
    starting_tree_coords: tp.Optional[_TTreeCoords] = (0, ),
):
    """ Handle the content of a section """
    if isinstance(section_content, dict):
        _handle_dict(
            renderable_objects,
            section_data=section_content,
            header_level=None if header_level is None else header_level + 1,
            starting_tree_coords=starting_tree_coords,
        )
    elif isinstance(section_content, list):
        _handle_list(
            renderable_objects=renderable_objects,
            elements=section_content,
            header_level=header_level,
            starting_tree_coords=starting_tree_coords,
        )
    else:
        # Get the tree coords for the element, then use them as uid
        element_uid = _increment_tree_coords(tree_coords=starting_tree_coords)
        _handle_single_element(
            renderable_objects=renderable_objects,
            element=section_content,
            uid=element_uid,
        )


# TODO: Should the list be allowed to contain dicts?
#  Maybe best not, in order to have a single way to define a structure, and dicts should
#  not be put into a list.
def _handle_list(
    renderable_objects: TPdfRenderableStory,
    elements: list,
    header_level: int,
    starting_tree_coords: tp.Optional[_TTreeCoords] = None,
):
    """
    Handle a list of content elements.

    Takes care of
    - keeping track of the tree coords of the elements
    - handling each element of the list by passing it to ``_handle_section_content``

    Notes:
        Notice that this implementation will break if a list of lists is provided.
        List of lists are not supposed to be found in the report structure. The intended
        result would anyway be the same as if they were in a single list, so they should
        be merged into a single list.
    """
    element_tree_coords = starting_tree_coords
    for element in elements:
        element_tree_coords = _increment_tree_coords(element_tree_coords)
        section_content_tree_coords = _deepen_tree_coords(element_tree_coords)
        _handle_section_content(
            renderable_objects=renderable_objects,
            section_content=element,
            header_level=header_level,
            starting_tree_coords=section_content_tree_coords,
        )


def _deepen_tree_coords(
    tree_coords: _TVarOptionalTreeCoords,
) -> _TVarOptionalTreeCoords:
    """Add an additional level to the tree coords

    Notes:
        If ``tree_coords`` is ``None``, returns ``None``.
        The coordinate of the added level of ``tree_coords`` is ``0`` (zero).
    """
    if tree_coords is None:
        return None
    return (*tree_coords, 0)


def _increment_tree_coords(
    tree_coords: _TVarOptionalTreeCoords,
) -> _TVarOptionalTreeCoords:
    """Increment the lowest level of the tree coord by 1

    Notes:
        If ``tree_coords`` is ``None``, returns ``None``.
    """
    if tree_coords is None:
        return None
    if not tree_coords:
        raise ValueError("Invalid coordinates of length zero were provided.")
    return (*tree_coords[:-1], tree_coords[-1] + 1)


def _convert_to_pdf_renderable(element) -> PdfRenderableElement:
    """ Convert supported ``element`` into the preferred pdf-renderable one.

    Takes a (supported) low-level object ``element`` and returns an instance of the
    preferred pdf-renderable class for that type of object (e.g. for a ``go.Figure``
    returns an instance of ``PdfRenderableImage,`` for a string one of
    ``PdfRenderableText``).

    The logic used follows the ``APPROPRIATE_CONVERSION_CLASS_SEARCH_STRATEGY``, where
    the class/protocol of ``element`` is checked sequentially against those specified in
    the conversion strategy, and when there is a match, then the matching pdf-renderable
    class is chosen.
    The (hopefully robust) strategy for choosing a suitable pdf-renderable class starts
    from the most sure cases, then continues to some less sure but potentially good
    choices for the conversion class.

    Raises ``NotImplementedError`` if ``element`` is not supported.
    """
    for conversion_class, classes_ in APPROPRIATE_CONVERSION_CLASS_SEARCH_STRATEGY:
        for class_ in classes_:
            if isinstance(element, class_):
                return conversion_class(element)
    element_class_name = element.__class__.__name__
    raise NotImplementedError(
        f"Pdf rendering of instances of '{element_class_name}' not supported",
    )


def _handle_single_element(
    renderable_objects: TPdfRenderableStory,
    element: TPdfRenderableContent,
    uid: tp.Optional[TUid] = None,
):
    """ Handle the conversion of a single element to a pdf-renderable object.

    Modifies ``renderable_objects`` in place.
    """
    pdf_renderable_element = _convert_to_pdf_renderable(element)
    pdf_renderable_element.update_uid(uid=uid)
    renderable_objects.append(pdf_renderable_element)


def _handle_dict(
    renderable_objects: TPdfRenderableStory,
    section_data,
    header_level: tp.Union[int, None],
    starting_tree_coords: tp.Optional[_TTreeCoords] = None,
) -> None:
    """ Handle the conversion of a dict into reportlab-friendly objects.

    Modify ``renderable_objects`` in place.
    """
    renderable_objects_ = []
    section_tree_coords = starting_tree_coords
    for header, section_content in section_data.items():
        section_tree_coords = _increment_tree_coords(tree_coords=section_tree_coords)
        header_starting_tree_coords = _deepen_tree_coords(
            tree_coords=section_tree_coords,
        )
        _handle_header(
            renderable_objects_,
            header=header,
            header_level=header_level,
            starting_tree_coords=header_starting_tree_coords,
        )
        section_tree_coords = _increment_tree_coords(tree_coords=section_tree_coords)
        section_content_starting_tree_coords = _deepen_tree_coords(
            tree_coords=section_tree_coords,
        )
        _handle_section_content(
            renderable_objects_,
            section_content,
            header_level,
            starting_tree_coords=section_content_starting_tree_coords,
        )
    renderable_objects.extend(renderable_objects_)


def _handle_header(
    renderable_objects: TPdfRenderableStory,
    header: TReportStructureHeader,
    header_level: tp.Union[int, None],
    starting_tree_coords: tp.Optional[_TTreeCoords] = None,
) -> None:
    """ Handle the conversion of a header into a pdf-renderable object.

    Modifies ``renderable_objects`` in place.

    Headers do not use the usual conversion logic with
    ``renderable_header = _convert_to_pdf_renderable(header)``
    and instead uses by design the ``PdfRenderableHeader``

    At the moment this should handle the ``SectionHeader`` and also some text.
    Techinically the PdfRenderableHeader does convert text into a string using
    ``str(text)`` internally.
    """
    tree_coords = _increment_tree_coords(tree_coords=starting_tree_coords)
    if not isinstance(header, identifiers.SectionHeader):
        # Then it is either str or int
        header = identifiers.SectionHeader(header_text=str(header))
    renderable_header = PdfRenderableHeader(
        header_text=header.header_text,
        header_level=header_level,
        uid=tree_coords,
    )
    renderable_objects.append(renderable_header)
