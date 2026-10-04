from app.ingestion.cleaning import clean_inline, clean_markdown


def test_roles_are_rendered_as_plain_names():
    text = "Use {class}`~torch.utils.data.DataLoader` and {func}`torch.manual_seed`."
    assert clean_inline(text) == "Use `torch.utils.data.DataLoader` and `torch.manual_seed`."


def test_roles_with_explicit_title_keep_only_the_title():
    text = "See {ref}`automatic batching <loading-batched-and-non-batched-data>`."
    assert clean_inline(text) == "See automatic batching."


def test_anchor_lines_and_autodoc_blocks_are_removed():
    source = "\n".join(
        [
            "# torch.optim",
            "```{eval-rst}",
            ".. automodule:: torch.optim",
            "```",
            "(optimizer-algorithms)=",
            "## Algorithms",
            "Prose stays.",
        ]
    )
    cleaned = clean_markdown(source)
    assert "automodule" not in cleaned
    assert "(optimizer-algorithms)=" not in cleaned
    assert "## Algorithms" in cleaned
    assert "Prose stays." in cleaned


def test_admonitions_are_unwrapped_with_a_label():
    source = "```{note}\n:class: dropdown\nPin memory for faster transfers.\n```\n"
    cleaned = clean_markdown(source)
    assert cleaned.strip() == "Note:\nPin memory for faster transfers."


def test_code_blocks_are_preserved_verbatim_including_indentation_and_roles():
    source = "\n".join(
        [
            "```python",
            "for x in loader:",
            "    # {class}`not a role in code`",
            "    loss.backward()",
            "```",
        ]
    )
    assert clean_markdown(source).strip() == source


def test_code_block_nested_inside_admonition_survives():
    source = "\n".join(
        [
            "````{warning}",
            "Do not do this:",
            "```python",
            "    model.cuda()",
            "```",
            "````",
            "After.",
        ]
    )
    cleaned = clean_markdown(source)
    assert cleaned.startswith("Warning:\nDo not do this:\n```python\n    model.cuda()\n```")
    assert cleaned.rstrip().endswith("After.")
    assert "````" not in cleaned


def test_fence_lines_inside_code_are_literal():
    source = "````markdown\n```{eval-rst}\nshown as an example\n```\n````\n"
    assert clean_markdown(source).strip() == source.strip()


def test_blank_line_runs_are_collapsed_and_comments_removed():
    cleaned = clean_markdown("a\n\n\n\n<!-- hidden -->\nb\n")
    assert cleaned == "a\n\nb\n"


def test_roles_wrapping_across_lines_are_cleaned():
    source = "You can control {ref}`how saved tensors are packed\n<saved-tensors-doc>` by hooks.\n"
    assert clean_markdown(source) == "You can control how saved tensors are packed by hooks.\n"


def test_rst_roles_embedded_in_markdown_are_cleaned():
    assert clean_inline("See :func:`torch.cuda.memory.CUDAPluggableAllocator`.") == (
        "See `torch.cuda.memory.CUDAPluggableAllocator`."
    )


def test_anchor_labels_with_special_characters_are_removed():
    assert clean_markdown("(extending-torch-c++)=\n## Extending\n") == "## Extending\n"


def test_code_block_and_math_directives_become_plain_fences():
    source = (
        "```{code-block} python\n:linenos:\nx = 1\n```\n\n```{math}\n:label: eq\ny = f(x)\n```\n"
    )
    assert clean_markdown(source) == "```python\nx = 1\n```\n\n```math\ny = f(x)\n```\n"


def test_colon_fenced_admonitions_are_unwrapped():
    source = ":::{warning}\n`torch.cuda.amp.autocast` is deprecated.\n:::\nAfter.\n"
    assert clean_markdown(source) == "Warning:\n`torch.cuda.amp.autocast` is deprecated.\nAfter.\n"


def test_image_and_toc_directives_are_dropped():
    source = "```{image} img.png\n:alt: diagram\n```\n```{contents}\n:local:\n```\nText.\n"
    assert clean_markdown(source) == "Text.\n"


def test_role_like_text_inside_inline_code_is_not_a_role():
    source = "form `{__module__}.{__name__}`. Allowlist via {func}`torch.add_safe_globals`."
    assert clean_inline(source) == (
        "form `{__module__}.{__name__}`. Allowlist via `torch.add_safe_globals`."
    )
