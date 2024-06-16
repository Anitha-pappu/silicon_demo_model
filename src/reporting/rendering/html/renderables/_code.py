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

import logging
import typing as tp
from uuid import uuid4

from reporting.rendering.html.notebook_mode_inits import (
    init_notebook_mode_for_code,
)

from ._base import InteractiveHtmlContentBase  # noqa: WPS436

logger = logging.getLogger(__name__)


class InteractiveHtmlRenderableCode(InteractiveHtmlContentBase):
    def __init__(
        self,
        code: str,
        language: tp.Optional[str] = None,
    ) -> None:
        """
        Creates code object that has html representation and is convertable to html

        Args:
            code: code to plot
            language: this will be used in representation to highlight the code.
                Use this for setting proper code highlighting in html repr in case
                auto recognised language is incorrect.
                Most common args are {'python', 'sql', 'json', 'js', 'java', 'c++'}.
                See `https://github.com/EnlighterJS/EnlighterJS#languages`
                for the full list of available languages.
        Raises:
            KeyError in case formatting is `code_formatter` is string
            and not found in available formatters
        """
        self._code = code
        self._language = language

    def to_html(self) -> str:
        return f'<div class="code-block">{self._repr_html_enlighter()}</div>'

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(code={self._code}, "
            f"language={self._language})"
        )

    def _repr_html_enlighter(self, code_tag_id: tp.Union[int, str, None] = None) -> str:
        id_ = uuid4() if code_tag_id is None else str(code_tag_id)
        code = self._code
        lang_attribute = (
            f'data-enlighter-language="{self._language}"'
            if self._language is not None
            else ""
        )
        code_class = 'class="enlight_js"'
        return f"<pre><code {code_class} id={id_} {lang_attribute}>{code}</code></pre>"

    def _repr_html_(self) -> str:
        """Returns HTML representation for code block"""

        # include init by default to simplify user experience (increases notebook size)
        init_notebook_mode_for_code(run_highlighter_for_all_code_blocks=False)

        uid = str(uuid4())
        activation_script = (
            "<script>"
            "EnlighterJS.enlight("
            f"  document.getElementById('{uid}'),"
            "   {rawcodeDbclick: true, toolbarBottom: false}"
            ");"
            "</script>"
        )
        return self._repr_html_enlighter(uid) + activation_script


def plot_code(
    code: str,
    language: tp.Optional[str] = None,
) -> InteractiveHtmlRenderableCode:
    """
    Creates a code object that can be shown as html object and converted to html

    Args:
        code: code to plot
        language: this will be used in representation to highlight the code

   Raises:
        KeyError in case formatting is `code_formatter` is string
        and not found in available formatters
    """
    return InteractiveHtmlRenderableCode(code=code, language=language)
