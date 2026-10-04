"""Text cleaning for Markdown / MyST documentation sources.

Sphinx documentation written in MyST Markdown (as PyTorch's is) contains markup
that only makes sense to the doc builder. Left in place it pollutes BM25 terms
and embeddings, so we normalize it to plain Markdown while keeping prose,
headings, code and math intact:

* ``{class}`~torch.utils.data.DataLoader``  ->  ```torch.utils.data.DataLoader```
* ``{ref}`automatic batching <label>``      ->  ``automatic batching``
* ``:func:`torch.foo``` (RST role)           ->  ```torch.foo```
* ``(label)=`` anchors and ``% comment`` lines ->  removed
* ``{eval-rst}`` / ``{toctree}`` / ``{image}`` blocks -> removed (no prose)
* ``{note}`` / ``{warning}`` admonitions     ->  "Note:" / "Warning:" + their content
* ``{code-block} python`` / ``{math}``       ->  plain ``python`` / ``math`` code fences
"""

from __future__ import annotations

import re

# Directives whose body is human-readable prose worth keeping, with the label to show.
_PROSE_DIRECTIVE_LABELS = {
    "admonition": "", "attention": "Attention", "caution": "Caution", "danger": "Danger",
    "deprecated": "Deprecated since version", "error": "Error", "hint": "Hint",
    "important": "Important", "list-table": "", "note": "Note", "seealso": "See also",
    "tip": "Tip", "versionadded": "New in version", "versionchanged": "Changed in version",
    "warning": "Warning",
}  # fmt: skip
_CODE_DIRECTIVES = {"code", "code-block", "sourcecode"}

_FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|:{3,})\s*(?P<info>.*)$")
_DIRECTIVE_RE = re.compile(r"^\{(?P<name>[\w:-]+)\}\s*(?P<arg>.*)$")
_DIRECTIVE_OPTION_RE = re.compile(r"^\s*:[\w-]+:")
_ANCHOR_RE = re.compile(r"^\s*\([^()\s]+\)=\s*$")
_COMMENT_RE = re.compile(r"^\s*%")  # MyST comment line
# MyST roles {role}`body` and RST roles :role:`body` / :py:func:`body` (bodies may wrap lines),
# or an inline code span. Matching code spans too means a role-like `{__name__}` *inside*
# code is consumed as code and left untouched instead of being mistaken for a role.
_INLINE_RE = re.compile(
    r"(?P<role>(?:\{[\w:-]+\}|:(?:[\w-]+:)+)`(?P<body>[^`]*)`)"
    r"|(?P<code>(?P<ticks>`+)[^`]+?(?P=ticks))"
)
_EXPLICIT_TITLE_RE = re.compile(r"^(?P<text>.+?)\s*<[^>]+>$", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLANK_RUNS_RE = re.compile(r"\n{3,}")

# What to do with an open fenced block.
KEEP, UNWRAP, DROP = "keep", "unwrap", "drop"


def clean_inline(text: str) -> str:
    """Replace MyST / RST roles with readable text."""

    def render(match: re.Match[str]) -> str:
        if match.group("code"):
            return match.group("code")
        body = match.group("body").strip()
        explicit = _EXPLICIT_TITLE_RE.match(body)
        if explicit:  # {ref}`human text <target-label>`
            return explicit.group("text")
        return f"`{body.lstrip('~!')}`"

    return _INLINE_RE.sub(render, text)


def clean_markdown(text: str) -> str:
    """Normalize MyST/Markdown documentation into plain Markdown.

    A line-based pass with a stack of open fences decides what is code, what is
    prose and what is dropped, which handles nested blocks (a ``{note}``
    containing a code block). Prose is then cleaned in runs rather than per
    line, because roles may wrap across lines.
    """
    text = _HTML_COMMENT_RE.sub("", text.replace("\r\n", "\n"))
    output: list[str] = []
    prose: list[str] = []
    stack: list[tuple[str, str, str]] = []  # (opening marker, action, marker to emit on close)
    skip_directive_options = False

    def flush_prose() -> None:
        if prose:
            output.extend(line.rstrip() for line in clean_inline("\n".join(prose)).split("\n"))
            prose.clear()

    def emit_verbatim(line: str) -> None:
        flush_prose()
        output.append(line)

    for line in text.split("\n"):
        dropping = any(action == DROP for _, action, _ in stack)
        in_code = bool(stack) and stack[-1][1] == KEEP
        fence = _FENCE_RE.match(line)

        if fence and _closes_innermost(fence, stack):
            _, action, close_marker = stack.pop()
            if action == KEEP and not dropping:
                emit_verbatim(close_marker)
            continue

        if fence and not in_code:  # inside code, fence-looking lines are literal text
            marker = fence.group("fence")
            action, opener, is_directive = _classify(marker, fence.group("info").strip())
            close_marker = "`" * max(3, len(marker)) if is_directive else marker
            stack.append((marker, action, close_marker))
            skip_directive_options = is_directive
            if dropping:
                continue
            if action == KEEP:
                emit_verbatim(opener if is_directive else line)
            elif action == UNWRAP and opener:
                prose.append(opener)
            continue

        if dropping:
            continue
        if skip_directive_options and _DIRECTIVE_OPTION_RE.match(line):
            continue  # e.g. ":class: dropdown" or ":linenos:" right after a directive opener
        skip_directive_options = False

        if in_code:
            emit_verbatim(line)  # never rewrite code or math
        elif not (_ANCHOR_RE.match(line) or _COMMENT_RE.match(line)):
            prose.append(line)

    flush_prose()
    cleaned = _BLANK_RUNS_RE.sub("\n\n", "\n".join(output))
    return cleaned.strip() + "\n"


def _classify(marker: str, info: str) -> tuple[str, str, bool]:
    """Map a fence to (action, line to emit when opening, whether it is a MyST directive)."""
    directive = _DIRECTIVE_RE.match(info)
    if not directive:
        return KEEP, "", False  # ordinary code block such as ```python
    name, arg = directive.group("name").lower(), directive.group("arg").strip()
    backticks = "`" * max(3, len(marker))
    if name in _CODE_DIRECTIVES:
        return KEEP, f"{backticks}{arg.split()[0] if arg else ''}", True
    if name == "math":
        return KEEP, f"{backticks}math", True
    if name in _PROSE_DIRECTIVE_LABELS:
        label = " ".join(part for part in (_PROSE_DIRECTIVE_LABELS[name], arg) if part)
        return UNWRAP, f"{label}:" if label else "", True
    return DROP, "", True  # eval-rst, toctree, image, contents, currentmodule, ...


def _closes_innermost(fence: re.Match[str], stack: list[tuple[str, str, str]]) -> bool:
    """A bare fence closes the innermost block if it uses the same marker, at least as long."""
    if not stack or fence.group("info").strip():
        return False
    marker, open_marker = fence.group("fence"), stack[-1][0]
    return marker[0] == open_marker[0] and len(marker) >= len(open_marker)
