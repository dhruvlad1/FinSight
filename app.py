import pandas as pd
import streamlit as st

from finsight.extraction import extract_guidance
from finsight.rag import generate_answer, retrieve_context
from finsight.retriever import FaissRetriever


st.set_page_config(
    page_title="FinSight | Earnings Call Intelligence",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: #f4f6f8;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    [data-testid="stSidebar"] * {
        color: #dbe3ed;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        background: #1b2638;
    }

    .hero {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 2rem 2.25rem 1.75rem;
        margin-bottom: 1.25rem;
    }

    .hero-kicker,
    .section-kicker {
        color: #5bb7a5;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero-title {
        color: #f8fafc;
        font-size: clamp(2rem, 4vw, 3.1rem);
        font-weight: 750;
        letter-spacing: 0;
        line-height: 1.05;
        margin: 0.35rem 0 0.6rem;
    }

    .hero-tagline {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 600;
        line-height: 1.35;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: #9eacbd;
        font-size: 0.95rem;
        margin: 0;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1.25rem;
    }

    .hero-pill {
        border: 1px solid #405166;
        border-radius: 999px;
        color: #cbd5e1;
        font-size: 0.76rem;
        padding: 0.32rem 0.65rem;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 750;
        line-height: 1.1;
        margin-top: 0.25rem;
    }

    .context-strip {
        background: #e8f3f0;
        border-left: 3px solid #258f7c;
        border-radius: 6px;
        color: #193b37;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1.25rem;
    }

    .context-company {
        font-size: 1.05rem;
        font-weight: 700;
    }

    .context-period {
        color: #42635e;
        font-size: 0.88rem;
        margin-top: 0.2rem;
    }

    .answer-card {
        background: #ffffff;
        border: 1px solid #d8e0e8;
        border-left: 4px solid #258f7c;
        border-radius: 8px;
        color: #172033;
        font-size: 1.03rem;
        line-height: 1.65;
        padding: 1.25rem 1.4rem;
    }

    .evidence-summary {
        color: #475569;
        font-size: 0.92rem;
        margin: -0.4rem 0 0.8rem;
    }

    .source-meta {
        color: #334155;
        font-size: 0.9rem;
        font-weight: 650;
    }

    .source-score {
        color: #258f7c;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .sidebar-label {
        color: #91a3b8;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin: 1.2rem 0 0.5rem;
        text-transform: uppercase;
    }

    .sidebar-summary {
        border-top: 1px solid #334155;
        color: #aebdcd;
        font-size: 0.82rem;
        line-height: 1.75;
        margin-top: 1rem;
        padding-top: 0.85rem;
    }

    .stButton > button[kind="primary"] {
        background: #1f806f;
        border-color: #1f806f;
    }

    .stButton > button[kind="primary"]:hover {
        background: #176757;
        border-color: #176757;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


COMPANY_NAMES = {
    "A": "Agilent Technologies, Inc.",
    "AAPL": "Apple Inc.",
    "ABBV": "AbbVie Inc.",
    "ABNB": "Airbnb, Inc.",
    "ABT": "Abbott Laboratories",
}


@st.cache_data
def load_chunks() -> pd.DataFrame:
    return pd.read_pickle("vectorstore/chunks.pkl")


@st.cache_resource
def load_retriever() -> FaissRetriever:
    return FaissRetriever.load("vectorstore/index.faiss")


def company_label(ticker: str) -> str:
    return f"{COMPANY_NAMES.get(ticker, ticker)} ({ticker})"


def transcript_count(data: pd.DataFrame) -> int:
    return int(data[["ticker", "quarter", "year"]].drop_duplicates().shape[0])


def render_metric(label: str, value: int) -> None:
    st.markdown(
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value:,}</div>',
        unsafe_allow_html=True,
    )


def build_guidance_rows(data: pd.DataFrame) -> list[dict[str, str]]:
    rows = []

    for _, row in data.iterrows():
        for statement in extract_guidance(row["text"]):
            rows.append(
                {
                    "Quarter": f"Q{row['quarter']} {row['year']}",
                    "Speaker": row["speaker"],
                    "Guidance": statement,
                }
            )

    return rows


def build_comparison_rows(data: pd.DataFrame) -> list[dict[str, str]]:
    rows = []

    for _, row in data.iterrows():
        for statement in extract_guidance(row["text"]):
            rows.append(
                {
                    "Quarter": f"Q{row['quarter']} {row['year']}",
                    "Speaker": row["speaker"],
                    "Management Statement": statement,
                }
            )

    return rows


def render_sources(sources: list[dict]) -> None:
    st.markdown('<div class="section-kicker">Source evidence</div>', unsafe_allow_html=True)
    st.subheader("Retrieved transcript evidence")
    st.markdown(
        f'<div class="evidence-summary">{len(sources)} transcript '
        "chunk(s) retrieved. Similarity scores describe retrieval relevance, "
        "not answer confidence.</div>",
        unsafe_allow_html=True,
    )

    if not sources:
        st.info("No transcript evidence was retrieved for this question.")
        return

    for index, source in enumerate(sources, start=1):
        display_company = COMPANY_NAMES.get(
            source["ticker"],
            source.get("company", source["ticker"]),
        )
        label = (
            f"Source {index} | {display_company} | "
            f"Q{source['quarter']} {source['year']} | {source['speaker']}"
        )

        with st.expander(label, expanded=index == 1):
            st.markdown(
                f'<div class="source-meta">{display_company} · '
                f"Q{source['quarter']} {source['year']} · "
                f"{source['speaker']}</div>"
                f'<div class="source-score">Retrieval similarity: '
                f"{source['score']:.3f}</div>",
                unsafe_allow_html=True,
            )
            st.write(source["text"])


chunks = load_chunks()
retriever = load_retriever()

total_transcripts = transcript_count(chunks)
total_chunks = len(chunks)
total_companies = int(chunks["ticker"].nunique())
total_quarters = int(chunks[["quarter", "year"]].drop_duplicates().shape[0])


with st.sidebar:
    st.markdown('<div class="sidebar-label">Research controls</div>', unsafe_allow_html=True)

    tickers = sorted(chunks["ticker"].unique())
    selected_ticker = st.selectbox(
        "Company",
        tickers,
        format_func=company_label,
    )

    quarter_options = ["All"] + sorted(
        chunks.loc[
            chunks["ticker"].eq(selected_ticker),
            "quarter",
        ].unique().tolist()
    )
    selected_quarter = st.selectbox("Quarter", quarter_options)

    top_k = st.slider(
        "Retrieved sources",
        min_value=3,
        max_value=10,
        value=5,
    )

    st.markdown(
        f'<div class="sidebar-label">Dataset</div>'
        f'<div class="sidebar-summary">'
        f"{total_companies} companies<br>"
        f"{total_transcripts} transcripts<br>"
        f"{total_chunks:,} chunks<br>"
        f"{total_quarters} quarter periods"
        "</div>",
        unsafe_allow_html=True,
    )


answer_context = (selected_ticker, selected_quarter, top_k)
if st.session_state.get("answer_context") != answer_context:
    st.session_state["answer_context"] = answer_context
    st.session_state["answer_result"] = None
    st.session_state["answer_error"] = None


st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Financial research workspace</div>
        <div class="hero-title">FinSight</div>
        <div class="hero-tagline">Evidence-grounded intelligence from earnings calls.</div>
        <p class="hero-subtitle">Temporal Earnings Call Intelligence &amp; RAG</p>
        <div class="hero-pills">
            <span class="hero-pill">Evidence-grounded</span>
            <span class="hero-pill">Metadata-aware</span>
            <span class="hero-pill">Multi-quarter</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


overview_columns = st.columns(4)
with overview_columns[0]:
    render_metric("Transcripts", total_transcripts)
with overview_columns[1]:
    render_metric("Chunks", total_chunks)
with overview_columns[2]:
    render_metric("Companies", total_companies)
with overview_columns[3]:
    render_metric("Quarter periods", total_quarters)


ask_tab, guidance_tab, compare_tab = st.tabs(
    ["Ask the Call", "Guidance", "Compare Quarters"]
)


with ask_tab:
    st.markdown('<div class="section-kicker">Research question</div>', unsafe_allow_html=True)
    st.header("Ask the Call")
    st.write(
        "Ask about management commentary, financial results, guidance, and outlook."
    )

    period_label = (
        "All available quarters"
        if selected_quarter == "All"
        else f"Q{selected_quarter}"
    )
    st.markdown(
        f'<div class="context-strip">'
        f'<div class="context-company">{company_label(selected_ticker)}</div>'
        f'<div class="context-period">Selected period: {period_label}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Question",
        placeholder=(
            "Ask about revenue, margins, guidance, outlook, or management commentary..."
        ),
        height=120,
        label_visibility="collapsed",
        key="research_question",
    )

    if st.button("Ask the call", type="primary", use_container_width=False):
        if not question.strip():
            st.warning("Enter a question to search the selected earnings call.")
        else:
            quarter_filter = (
                None if selected_quarter == "All" else int(selected_quarter)
            )

            try:
                with st.spinner(
                    "Searching transcript evidence and generating grounded response..."
                ):
                    context, retrieved = retrieve_context(
                        question=question,
                        chunks=chunks,
                        retriever=retriever,
                        top_k=top_k,
                        ticker=selected_ticker,
                        quarter=quarter_filter,
                    )
                    result = generate_answer(
                        question=question,
                        context=context,
                        sources=retrieved,
                    )

                st.session_state["answer_result"] = result
                st.session_state["answer_error"] = None
            except ValueError as error:
                st.session_state["answer_result"] = None
                st.session_state["answer_error"] = str(error)
            except Exception:
                st.session_state["answer_result"] = None
                st.session_state["answer_error"] = (
                    "The grounded response could not be generated. "
                    "Check the application configuration and try again."
                )

    answer_error = st.session_state.get("answer_error")
    answer_result = st.session_state.get("answer_result")

    if answer_error:
        st.error(answer_error)

    if answer_result:
        st.markdown('<div class="section-kicker">Grounded answer</div>', unsafe_allow_html=True)
        st.subheader("Answer")
        st.markdown(
            f'<div class="answer-card">{answer_result["answer"]}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Answer generated from the retrieved transcript context.")
        render_sources(answer_result["sources"])


with guidance_tab:
    st.markdown('<div class="section-kicker">Forward-looking statements</div>', unsafe_allow_html=True)
    st.header("Management Guidance")
    st.write("Track guidance-like statements across the selected earnings calls.")

    guidance_chunks = chunks[chunks["ticker"].eq(selected_ticker)].copy()
    if selected_quarter != "All":
        guidance_chunks = guidance_chunks[
            guidance_chunks["quarter"].eq(int(selected_quarter))
        ]

    guidance_rows = build_guidance_rows(guidance_chunks)
    guidance_df = pd.DataFrame(guidance_rows)

    summary_columns = st.columns(2)
    with summary_columns[0]:
        render_metric("Guidance statements", len(guidance_rows))
    with summary_columns[1]:
        covered_quarters = (
            int(guidance_df["Quarter"].nunique()) if guidance_rows else 0
        )
        render_metric("Quarter periods covered", covered_quarters)

    if guidance_rows:
        st.dataframe(
            guidance_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Quarter": st.column_config.TextColumn(width="small"),
                "Speaker": st.column_config.TextColumn(width="medium"),
                "Guidance": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.info("No management guidance statements were found for the selected period.")


with compare_tab:
    st.markdown('<div class="section-kicker">Temporal analysis</div>', unsafe_allow_html=True)
    st.header("Multi-Quarter Comparison")
    st.write("Compare management commentary across earnings calls.")
    st.markdown(
        f'<div class="context-strip">'
        f'<div class="context-company">{company_label(selected_ticker)}</div>'
        '<div class="context-period">Select the quarter periods to compare.</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    available_quarters = sorted(
        chunks.loc[
            chunks["ticker"].eq(selected_ticker),
            "quarter",
        ].unique().tolist()
    )
    comparison_quarters = st.multiselect(
        "Quarter periods",
        available_quarters,
        default=available_quarters,
        format_func=lambda quarter: f"Q{quarter}",
    )

    if comparison_quarters:
        comparison_chunks = chunks[
            chunks["ticker"].eq(selected_ticker)
            & chunks["quarter"].isin(comparison_quarters)
        ].copy()
        comparison_rows = build_comparison_rows(comparison_chunks)

        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            st.caption(
                f"{len(comparison_rows):,} management statements across "
                f"{len(comparison_quarters)} selected quarter periods."
            )
            st.dataframe(
                comparison_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Quarter": st.column_config.TextColumn(width="small"),
                    "Speaker": st.column_config.TextColumn(width="medium"),
                    "Management Statement": st.column_config.TextColumn(width="large"),
                },
            )
        else:
            st.info("No management statements were found for the selected quarters.")
    else:
        st.info("Select at least one quarter to compare.")