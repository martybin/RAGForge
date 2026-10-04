"""Streamlit front-end for RAGForge.

A thin client of the FastAPI service: models load once in the API process,
and the UI only renders what the API returns. Debug mode shows every stage of
the retrieval pipeline for each answer.

Run:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
import re

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
QUERY_TIMEOUT_S = float(os.getenv("UI_QUERY_TIMEOUT_SECONDS", "600"))

st.set_page_config(page_title="RAGForge", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .rf-subtitle {color: #9AA3B2; margin-top: -0.6rem;}
      .rf-flow {display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin: .4rem 0 1rem;}
      .rf-step {background: #161B26; border: 1px solid #2A3242; border-radius: 8px;
                padding: .35rem .6rem; font-size: .82rem; color: #E6E8EE;}
      .rf-step b {color: #4FD1C5;}
      .rf-arrow {color: #5B6475;}
      .rf-source {border-left: 3px solid #4FD1C5; padding: .2rem .7rem; margin: .35rem 0;
                  background: #131824; border-radius: 0 6px 6px 0; font-size: .88rem;}
      .rf-muted {color: #9AA3B2; font-size: .8rem;}
      .rf-cite {color: #4FD1C5; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- API access -----------------------------------------------------------------


@st.cache_data(ttl=15, show_spinner=False)
def fetch_health() -> dict | None:
    try:
        response = httpx.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def ask(question: str) -> dict:
    response = httpx.post(f"{API_URL}/query", json={"question": question}, timeout=QUERY_TIMEOUT_S)
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.content else response.text
        raise RuntimeError(f"API error {response.status_code}: {detail}")
    return response.json()


# --- Rendering helpers ---------------------------------------------------------------


def highlight_citations(answer: str) -> str:
    return re.sub(r"(\[\d+(?:,\s*\d+)*\])", r'<span class="rf-cite">\1</span>', answer)


def results_table(results: list[dict], extra_scores: tuple[str, ...] = ()) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "rank": r["rank"],
            "score": round(r["score"], 4),
            "chunk_id": r["chunk_id"],
            "section": r["metadata"]["section"],
        }
        for key in extra_scores:
            if key in r["component_scores"]:
                row[key] = round(r["component_scores"][key], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def render_stage(title: str, results: list[dict], extra_scores: tuple[str, ...] = ()) -> None:
    st.markdown(f"**{title}** · {len(results)} results")
    if not results:
        st.caption("No results.")
        return
    st.dataframe(results_table(results, extra_scores), hide_index=True, use_container_width=True)


def render_flow(response: dict) -> None:
    trace, timings = response["retrieval"], response["timings"]
    steps = [
        ("Question", ""),
        ("Dense", f"{len(trace['dense'])}"),
        ("BM25", f"{len(trace['bm25'])}"),
        ("Hybrid", f"{len(trace['hybrid'])} · {timings['retrieval_ms']:.0f} ms"),
        ("Reranked", f"{len(trace['reranked'])} · {timings['rerank_ms']:.0f} ms"),
        ("Context", f"{len(trace['context'])}"),
        ("LLM answer", f"{timings['generation_ms'] / 1000:.1f} s"),
    ]
    html = '<span class="rf-arrow">→</span>'.join(
        f'<span class="rf-step">{name} <b>{value}</b></span>' for name, value in steps
    )
    st.markdown(f'<div class="rf-flow">{html}</div>', unsafe_allow_html=True)


def render_sources(response: dict) -> None:
    if not response["sources"]:
        return
    with st.expander(f"Sources ({len(response['sources'])})", expanded=True):
        context = {c["chunk_id"]: c for c in response["retrieval"]["context"]}
        for source in response["sources"]:
            st.markdown(
                f'<div class="rf-source"><span class="rf-cite">[{source["index"]}]</span> '
                f"<code>{source['source']}</code><br>"
                f'<span class="rf-muted">{source["section"]} · rerank score {source["score"]:.3f}</span></div>',
                unsafe_allow_html=True,
            )
            chunk = context.get(source["chunk_id"])
            if chunk:
                with st.popover(f"View chunk [{source['index']}]"):
                    st.markdown(chunk["content"])


def render_debug(response: dict) -> None:
    trace = response["retrieval"]
    with st.expander("Retrieval pipeline (debug)", expanded=False):
        render_flow(response)
        dense_tab, bm25_tab, hybrid_tab, rerank_tab, context_tab = st.tabs(
            ["Dense", "BM25", "Hybrid", "Reranked", "Final context"]
        )
        with dense_tab:
            render_stage("Dense (cosine similarity)", trace["dense"])
        with bm25_tab:
            render_stage("BM25 (Okapi score)", trace["bm25"])
        with hybrid_tab:
            render_stage(
                "Hybrid (α·dense_norm + (1-α)·bm25_norm)",
                trace["hybrid"],
                ("dense", "bm25", "dense_norm", "bm25_norm"),
            )
        with rerank_tab:
            render_stage(
                "Cross-encoder reranked (sigmoid score)",
                trace["reranked"],
                ("hybrid", "hybrid_rank"),
            )
        with context_tab:
            for i, chunk in enumerate(trace["context"], start=1):
                st.markdown(f"**[{i}] {chunk['chunk_id']}** · score {chunk['score']:.3f}")
                st.code(chunk["content"], language="markdown")
            for dropped in trace["filtered_out"]:
                st.caption(f"Filtered out {dropped['chunk_id']}: {dropped['reason']}")


def render_response(response: dict, debug: bool) -> None:
    if response["insufficient_evidence"]:
        st.warning(response["answer"])
    else:
        st.markdown(highlight_citations(response["answer"]), unsafe_allow_html=True)
    for warning in response.get("warnings", []):
        st.caption(f"⚠️ {warning}")
    render_sources(response)
    t = response["timings"]
    st.caption(
        f"retrieval {t['retrieval_ms']:.0f} ms · rerank {t['rerank_ms']:.0f} ms · "
        f"generation {t['generation_ms'] / 1000:.1f} s · total {t['total_ms'] / 1000:.1f} s"
    )
    if debug:
        render_debug(response)


# --- Sidebar --------------------------------------------------------------------------


def render_sidebar() -> bool:
    with st.sidebar:
        st.header("System")
        health = fetch_health()
        if health is None:
            st.error(f"API unreachable at {API_URL}")
        else:
            status = health["status"]
            icon = {"ok": "🟢", "degraded": "🟠", "unavailable": "🔴"}[status]
            st.markdown(f"{icon} API **{status}** · {health['indexed_chunks']} chunks indexed")
            if health.get("detail"):
                st.caption(health["detail"])
            config = health["config"]
            st.subheader("Models")
            st.markdown(
                f"- **LLM:** `{config['llm_model']}` ({config['llm_provider']})\n"
                f"- **Embeddings:** `{config['embedding_model']}`\n"
                f"- **Reranker:** `{config['reranker_model']}`"
            )
            st.subheader("Retrieval")
            st.markdown(
                f"- Chunks: {config['chunk_size']} tokens, {config['chunk_overlap']} overlap\n"
                f"- Dense top-k {config['top_k_dense']} · BM25 top-k {config['top_k_bm25']}\n"
                f"- Hybrid α = {config['hybrid_alpha']} → top {config['top_k_hybrid']}\n"
                f"- Rerank → top {config['top_k_reranked']} (min score {config['min_rerank_score']})\n"
                f"- Context budget {config['max_context_tokens']} tokens"
            )
        st.divider()
        debug = st.toggle(
            "Debug mode", value=False, help="Show every stage of the retrieval pipeline."
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    return debug


# --- Main -----------------------------------------------------------------------------


def main() -> None:
    st.title("🔎 RAGForge")
    st.markdown(
        '<p class="rf-subtitle">Answers from the indexed PyTorch documentation only, with citations. '
        "Hybrid retrieval → BGE reranking → grounded generation.</p>",
        unsafe_allow_html=True,
    )
    debug = render_sidebar()
    st.session_state.setdefault("history", [])

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            render_response(turn["response"], debug)

    question = st.chat_input(
        "Ask about the PyTorch docs, e.g. How do I make DataLoader workers reproducible?"
    )
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking and generating…"):
            try:
                response = ask(question)
            except (httpx.HTTPError, RuntimeError) as exc:
                st.error(str(exc))
                return
        render_response(response, debug)
    st.session_state.history.append({"question": question, "response": response})


main()
