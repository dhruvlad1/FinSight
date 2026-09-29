import pandas as pd
import streamlit as st

from finsight.extraction import extract_guidance
from finsight.rag import format_sources, generate_answer, retrieve_context
from finsight.retriever import FaissRetriever


st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="wide",
)

st.title("FinSight")
st.caption("Temporal Earnings Call Intelligence & RAG")


@st.cache_data
def load_chunks() -> pd.DataFrame:
    return pd.read_pickle("vectorstore/chunks.pkl")


@st.cache_resource
def load_retriever() -> FaissRetriever:
    return FaissRetriever.load("vectorstore/index.faiss")


chunks = load_chunks()
retriever = load_retriever()


COMPANY_NAMES = {
    "A": "Agilent Technologies, Inc.",
    "AAPL": "Apple Inc.",
    "ABBV": "AbbVie Inc.",
    "ABNB": "Airbnb, Inc.",
    "ABT": "Abbott Laboratories",
}


st.sidebar.header("Transcript Selection")

tickers = sorted(chunks["ticker"].unique())

selected_ticker = st.sidebar.selectbox(
    "Company",
    tickers,
    format_func=lambda ticker: f"{COMPANY_NAMES.get(ticker, ticker)} ({ticker})",
)

ticker = selected_ticker

quarter_options = ["All"] + sorted(
    chunks.loc[
        chunks["ticker"].eq(ticker),
        "quarter",
    ].unique().tolist()
)

quarter = st.sidebar.selectbox("Quarter", quarter_options)

question = st.text_area(
    "Ask about the earnings call",
    placeholder="Example: What was the company's revenue?",
)

top_k = st.sidebar.slider(
    "Retrieved sources",
    min_value=3,
    max_value=10,
    value=5,
)


if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        selected_quarter = None if quarter == "All" else int(quarter)

        with st.spinner("Retrieving evidence and generating answer..."):
            context, retrieved = retrieve_context(
                question=question,
                chunks=chunks,
                retriever=retriever,
                top_k=top_k,
                ticker=ticker,
                quarter=selected_quarter,
            )

            result = generate_answer(
                question=question,
                context=context,
                sources=retrieved,
            )

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")

        for source in format_sources(result["sources"]):
            st.write(f"- {source}")

        with st.expander("Retrieved transcript evidence"):
            for source in result["sources"]:
                company_name = COMPANY_NAMES.get(
                    source["ticker"],
                    source["ticker"],
                )

                st.markdown(
                    f"**{company_name} ({source['ticker']}) — "
                    f"Q{source['quarter']} {source['year']} — "
                    f"{source['speaker']}**"
                )

                st.write(source["text"])


# Management Guidance Tracker
st.divider()
st.subheader("Management Guidance Tracker")

guidance_chunks = chunks[chunks["ticker"].eq(ticker)].copy()

if quarter != "All":
    guidance_chunks = guidance_chunks[
        guidance_chunks["quarter"].eq(int(quarter))
    ]

guidance_rows = []

for _, row in guidance_chunks.iterrows():
    statements = extract_guidance(row["text"])

    for statement in statements:
        guidance_rows.append(
            {
                "Quarter": f"Q{row['quarter']} {row['year']}",
                "Speaker": row["speaker"],
                "Guidance": statement,
            }
        )

if guidance_rows:
    guidance_df = pd.DataFrame(guidance_rows)

    st.write(
        f"Found **{len(guidance_df)}** guidance statements "
        f"for {COMPANY_NAMES.get(ticker, ticker)}."
    )

    st.dataframe(
        guidance_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No management guidance statements were found.")