import streamlit as st

from sentence_transformers import SentenceTransformer

from main import (
    load_dataset, build_bm25_index, compute_dense_embeddings,
    build_faiss_index, load_classifier, retrieve_similar_claim,
    predict_claim_label, fact_check, DATA_PATH
)

@st.cache_resource
def load_resources():
    """Load all the resources we need: dataset, indexes, models."""
    df = load_dataset(DATA_PATH)
    statements = df["statement"].tolist()

    # BM25
    bm25 = build_bm25_index(statements)

    # FAISS with SentenceTransformer
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = compute_dense_embeddings(dense_model, statements)
    faiss_index = build_faiss_index(embeddings)

    # Load classifier (fine-tuned BERT)
    classifier, tokenizer = load_classifier()

    return df, bm25, faiss_index, dense_model, classifier, tokenizer

def main():
    st.title("Fact-Checking System with LIAR Dataset")
    st.write("Enter a claim to verify its veracity.")

    # Load resources (caches after first run)
    df, bm25, faiss_index, dense_model, classifier, tokenizer = load_resources()

    # Text input
    user_query = st.text_input("Enter your claim here:")

    # Checkbox for verbose (optional)
    verbose_mode = st.checkbox("Show detailed evidence (verbose mode)?", value=False)

    # Button to run the fact-check
    if st.button("Check Claim"):
        if not user_query.strip():
            st.warning("Please enter a valid claim.")
        else:
            # Call your fact_check function
            response = fact_check(
                user_query,
                bm25,
                faiss_index,
                dense_model,
                df,
                classifier,
                tokenizer,
                verbose=verbose_mode
            )
            st.markdown("### Result")
            st.write(response)

if __name__ == "__main__":
    main()