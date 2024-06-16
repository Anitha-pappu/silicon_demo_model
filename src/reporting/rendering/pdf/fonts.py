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
Check here for info on how to register a font family.
https://stackoverflow.com/questions/14370630/reportlab-pdfgen-support-for-bold-truetype-fonts
"""

import typing as tp
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONTS_FOLDER = Path(__file__).parent / "font_library"
PTSANS_FOLDER = (FONTS_FOLDER / "PT_Sans")
PTSERIF_FOLDER = (FONTS_FOLDER / "PT_Serif")
PTMONO_FOLDER = (FONTS_FOLDER / "PT_Mono")
PTMONO_TTF = (PTMONO_FOLDER / "PTMono-Regular.ttf").resolve()


def register_fonts() -> None:
    """Makes sure the needed fonts are registered"""
    pdfmetrics.registerFont(TTFont("PTMono-Regular", PTMONO_TTF))
    register_ptsans_font_family()
    _register_font_family(
        font_folder=PTSERIF_FOLDER,
        font_family="PTSerif",
        normal_ttf="PTSerif-Regular.ttf",
        bold_ttf="PTSerif-Bold.ttf",
        italic_ttf="PTSerif-Italic.ttf",
        bold_italic_ttf="PTSerif-BoldItalic.ttf",
    )


def register_ptsans_font_family() -> None:
    """Register the PTSans font family.

    Notes:
        Font files from: https://fonts.google.com/specimen/PT+Sans

        Includes regular, bold, italic and bold-italic styles.
    """
    font_folder = PTSANS_FOLDER
    normal_path = (font_folder / "PTSans-Regular.ttf").resolve()
    bold_path = (font_folder / "PTSans-Bold.ttf").resolve()
    italic_path = (font_folder / "PTSans-Italic.ttf").resolve()
    bold_italic_path = (font_folder / "PTSans-BoldItalic.ttf").resolve()

    pdfmetrics.registerFont(TTFont("PTSans-Regular", normal_path))
    pdfmetrics.registerFont(TTFont("PTSans-Bold", bold_path))
    pdfmetrics.registerFont(TTFont("PTSans-Italic", italic_path))
    pdfmetrics.registerFont(TTFont("PTSans-BoldItalic", bold_italic_path))

    pdfmetrics.registerFontFamily(
        "PTSans",
        normal="PTSans-Regular",
        bold="PTSans-Bold",
        italic="PTSans-Italic",
        boldItalic="PTSans-BoldItalic",
    )


def _register_font_family(
    font_folder: Path,
    font_family: str,
    normal_ttf: str,
    bold_ttf: tp.Optional[str] = None,
    italic_ttf: tp.Optional[str] = None,
    bold_italic_ttf: tp.Optional[str] = None,
) -> None:
    """ Register a family of fonts.

    Notes:
        Assumes the .ttf files are inside ``font_folder``.
        Requires that the ``normal_ttf`` is provided (so it can be used as a fallback
        option if any of the other types is missing)
    """
    normal_path = (font_folder / normal_ttf).resolve()
    bold_path = (font_folder / bold_ttf).resolve() if bold_ttf else None
    italic_path = (font_folder / italic_ttf).resolve() if italic_ttf else None
    bold_italic_path = (
        font_folder / bold_italic_ttf
    ).resolve() if bold_italic_ttf else None

    normal_name = f"{font_family}-Regular"
    pdfmetrics.registerFont(TTFont(normal_name, normal_path))

    bold_name, italic_name, bold_italic_name = None, None, None
    if bold_path:
        bold_name = f"{font_family}-Bold"
        pdfmetrics.registerFont(TTFont(bold_name, bold_path))
    if italic_path:
        italic_name = f"{font_family}-Italic"
        pdfmetrics.registerFont(TTFont(italic_name, italic_path))
    if bold_italic_path:
        bold_italic_name = f"{font_family}-BoldItalic"
        pdfmetrics.registerFont(TTFont(bold_italic_name, bold_italic_path))

    pdfmetrics.registerFontFamily(
        family=font_family,
        normal=normal_name,
        bold=bold_name,
        italic=italic_name,
        boldItalic=bold_italic_name,
    )
