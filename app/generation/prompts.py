"""Prompt construction for grounded, cited answers.

The model sees numbered context passages and must cite them as ``[n]``. The
numbering is ours, so citations can be validated mechanically after generation
(see `app.generation.citations`).
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import RetrievedChunk

REFUSAL_MESSAGE = "I couldn't find enough evidence in the knowledge base to answer this reliably."

SYSTEM_PROMPT = f"""You are RAGForge, an assistant that answers questions about technical documentation.
You must answer using ONLY the numbered context passages in the user message.

Rules:
1. Use only information stated in the context passages. Do not use prior knowledge, even if you are confident it is correct.
2. Cite the supporting passage number after every factual statement, like [1] or [2][3]. Only cite passage numbers that appear in the context.
3. If the passages do not contain enough information to answer the question, reply with exactly this sentence and nothing else:
{REFUSAL_MESSAGE}
4. Answer directly and start with the answer itself. Use the prefix "Inference:" ONLY for a sentence that is not written in the passages but logically follows from them; never use it for information the passages state directly.
5. Never invent API names, arguments, default values, file names or sources.
6. Be concise and technical. Format code identifiers with backticks."""

USER_PROMPT_TEMPLATE = """Context passages:

{context}

Question: {question}

Answer using only the context passages above, with [n] citations."""

# LangChain prompt: fixed system message + templated user message (context, question).
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=SYSTEM_PROMPT), ("human", USER_PROMPT_TEMPLATE)]
)


def format_passage(index: int, chunk: RetrievedChunk) -> str:
    """One numbered context block, labelled with its provenance."""
    meta = chunk.metadata
    page = f" | Page: {meta.page}" if meta.page is not None else ""
    # Chunks embed their section path as a first line for retrieval; the label already shows it.
    body = chunk.content.removeprefix(f"{meta.section}\n\n").strip()
    return f"[{index}] Source: {meta.source} | Section: {meta.section}{page}\n{body}"


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(format_passage(i, chunk) for i, chunk in enumerate(chunks, start=1))


def build_user_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    return USER_PROMPT_TEMPLATE.format(context=format_context(chunks), question=question.strip())
