"""SGLang tool-call parser for Flint: SGLang's parsing of Flint's tool-call format plus Flint's turn grammar.

The main additions are SGLang's structural-tag hooks: every request with tools (``tool_choice`` ``auto``,
``required`` or a named function) is constrained by ``build_flint_tool_grammar``; the ``flint`` grammar backend
lets Flint reason before it. As in OpenAI's API, a tool's argument schema is enforced only when the tool is sent with
``strict: true`` (SGLang fills an omitted ``strict`` with False); other tools keep the call format but not the
schema. ``parallel_tool_calls`` reaches the grammar through ``pass_parallel_tool_calls``, a plugin hook, because
SGLang's structural-tag hooks don't receive it.
"""

import re
from typing import Literal

from flint_sglang.tool_grammar import build_flint_tool_grammar
from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from xgrammar import StructuralTag

TOOL_CALL_PARSER_NAME = "flint"


class FlintDetector(Glm47MoeDetector):
    """Tool-call detector for Flint's format that constrains every tool choice to Flint's turn grammar."""

    parallel_tool_calls: bool = True  # the request's, set by pass_parallel_tool_calls

    def __init__(self) -> None:
        super().__init__()
        # XGrammar's tool-call tag allows whitespace after a tool's name. SGLang strips it when streaming; strip it
        # here too, before the name is used to type the arguments.
        self.func_detail_regex = re.compile(r"<tool_call>(.*?)\s*(<arg_key>.*?)?</tool_call>", re.DOTALL)

    def get_auto_tool_call_structural_tag(self, tools: list[Tool] | None = None) -> StructuralTag:
        return self.get_structural_tag(tools)

    def get_structural_tag(
        self,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | Literal["auto", "required"] = "auto",
        thinking_mode: bool = False,
    ) -> StructuralTag:
        choice = tool_choice.model_dump() if isinstance(tool_choice, ToolChoice) else tool_choice
        return build_flint_tool_grammar([tool.model_dump() for tool in tools or []], choice, self.parallel_tool_calls)


def pass_parallel_tool_calls(parser: FunctionCallParser, *args, parallel_tool_calls: bool = True, **kwargs) -> None:
    """``BEFORE`` hook on ``FunctionCallParser.get_structure_constraint``: give Flint's detector the request's
    ``parallel_tool_calls``, which SGLang passes to that method but not to the detector's structural-tag hooks."""
    if isinstance(parser.detector, FlintDetector):
        parser.detector.parallel_tool_calls = parallel_tool_calls
