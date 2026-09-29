import pandas as pd
import streamlit as st

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

st.sidebar.header("Transcript Selection")

tickers = sorted(chunks["ticker"].unique())
ticker = st.sidebar.selectbox("Company", tickers)

quarter_options = ["All"] + sorted(
    chunks.loc[chunks["ticker"].eq(ticker), "quarter"].unique().tolist()
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
                st.markdown(
                    f"**{source['ticker']} — Q{source['quarter']} "
                    f"{source['year']} — {source['speaker']}**"
                )
                st.write(source["text"])