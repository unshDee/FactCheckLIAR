"""
A fact‐checking system using the LIAR dataset.
It builds a BM25 index (sparse retrieval) and a FAISS index (dense retrieval)
over claim statements, uses a transformer‐based classifier (BERT) to predict veracity,
and then generates a structured system response.
Usage:
    Command line:
       python main.py --query "Your claim here" [--verbose]
    (If --query is not provided, it will prompt for input.)
    Optionally, use --train_classifier to force retraining of the classifier.
"""

import os
import argparse
import numpy as np
import pandas as pd
import faiss
import torch

# BM25 for sparse retrieval: tokenizes documents and scores them based on term frequency
from rank_bm25 import BM25Okapi

# SentenceTransformer to compute dense embeddings for semantic similarity
from sentence_transformers import SentenceTransformer

# HuggingFace transformers for BERT-based classification
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Datasets library helps with training data preparation
from datasets import Dataset

# Global paths and label mappings
DATA_PATH = "data/train.tsv"  # Path to the training TSV file
LABEL2ID = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_dataset(data_path=DATA_PATH):
    """
    Load the LIAR dataset from a TSV file.
    The columns are defined manually because the file doesn't include a header.
    """
    cols = [
        "id",  # Unique identifier for each claim
        "label",  # Original fact-check labels (e.g., true, false, etc.)
        "statement",  # The claim text
        "subject",  # Topic of discussion
        "speaker",  # Who made the claim
        "job_title",  # Speaker's job title
        "state_info",  # State information of the speaker
        "party_affiliation",  # Political party affiliation of the speaker
        "barely_true_counts",  # Historical count of barely-true statements by the speaker
        "false_counts",  # Historical count of false statements by the speaker
        "half_true_counts",  # Historical count of half-true statements
        "mostly_true_counts",  # Historical count of mostly-true statements
        "pants_onfire_counts",  # Historical count of 'pants on fire' statements
        "context"  # Additional context about the claim (e.g., location or event)
    ]
    df = pd.read_csv(data_path, sep="\t", header=None, names=cols, encoding="utf-8")
    df.fillna("", inplace=True)  # Replace any missing values with an empty string
    return df


def build_bm25_index(documents):
    """
    Build a BM25 index for sparse retrieval.
    - Tokenizes each document by converting to lowercase and splitting on whitespace.
    - BM25Okapi then uses these tokens to compute relevance scores.
    """
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25


def compute_dense_embeddings(model, sentences, batch_size=32):
    """
    Compute dense vector embeddings for a list of sentences using a SentenceTransformer.
    - The embeddings are converted to a float32 numpy array.
    - FAISS expects normalized vectors for cosine similarity; thus, L2 normalization is applied.
    """
    embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)  # Normalize each embedding to unit length
    return embeddings


def build_faiss_index(embeddings):
    """
    Build a FAISS index using inner product (equivalent to cosine similarity on normalized vectors).
    - The index is built on the dimension 'd' of the embeddings.
    - All embeddings are added to the index.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def train_classifier(df, model_name="bert-base-uncased", num_epochs=8, output_dir="classifier_model"):
    """
    Train a BERT-based classifier on the LIAR dataset.
    Steps:
      1. Map the original string labels to numerical IDs.
      2. Create a HuggingFace Dataset from the DataFrame (using only 'statement' and 'label').
      3. Tokenize the statements and prepare the dataset for training.
      4. Fine-tune a pre-trained BERT model for sequence classification.
      5. Save the fine-tuned model and tokenizer.
    """

    def map_labels(example):
        example["label_id"] = LABEL2ID.get(example["label"].strip().lower(), -1)
        return example

    # Convert the DataFrame into a HuggingFace Dataset and process labels
    dataset = Dataset.from_pandas(df[["statement", "label"]])
    dataset = dataset.map(map_labels)
    dataset = dataset.filter(lambda x: x["label_id"] != -1)

    # Load pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize the claim statements with appropriate padding and truncation
    def tokenize_function(examples):
        return tokenizer(examples["statement"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["statement", "label"])
    tokenized_dataset = tokenized_dataset.rename_column("label_id", "labels")
    tokenized_dataset.set_format("torch")  # Prepare dataset for PyTorch training

    # Load the pre-trained BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(LABEL2ID))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=10,
        logging_dir='./logs',
        disable_tqdm=False,
    )

    # Initialize the Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Training classifier ...")
    trainer.train()  # Fine-tune the classifier
    # Save the trained model and tokenizer for later use
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Classifier saved to {output_dir}.")
    return model, tokenizer


def load_classifier(model_dir="classifier_model"):
    """
    Load a saved classifier. If not found locally, download it from Hugging Face.
    - hf_model: the Hugging Face repository identifier.
    """
    hf_model = "unshDee/liar_qa"
    if os.path.exists(model_dir):
        # Load from local directory if available
        model = BertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        print(f"Loaded classifier from local directory: {model_dir}.")
    else:
        # Download from Hugging Face Hub if local version is not found
        print(f"Local model not found. Downloading from Hugging Face repository: {hf_model}...")
        model = BertForSequenceClassification.from_pretrained(hf_model)
        tokenizer = BertTokenizer.from_pretrained(hf_model)
        # Optionally save the downloaded model locally for future use
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    return model, tokenizer


def predict_claim_label(claim, classifier, tokenizer):
    """
    Predict the veracity label for a given claim.
    - Tokenizes the input claim and moves the tensors to the same device as the classifier.
    - Runs a forward pass through the classifier to get logits.
    - Returns the label corresponding to the highest logit.
    """
    inputs = tokenizer(claim, return_tensors="pt", truncation=True, padding=True, max_length=128)
    device = next(classifier.parameters()).device  # Get model device (CPU or GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
    with torch.no_grad():
        outputs = classifier(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return ID2LABEL[predicted_class]


def retrieve_similar_claim(query, bm25, faiss_index, dense_model, df, alpha=0.5, top_k=3):
    """
    Retrieve similar claims from the dataset using a hybrid approach:
      - BM25 provides sparse (keyword-based) similarity scores.
      - Dense embeddings (from SentenceTransformer) provide semantic similarity scores.
    The scores are normalized and then combined via a weighted sum.
    Returns:
      - best_idx: The index of the most similar claim.
      - top_claims: DataFrame with the top k retrieved claims and their combined scores.
    """
    # Get BM25 scores for the query after tokenizing it
    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)

    # Compute the dense embedding for the query
    query_embedding = dense_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)
    # Search FAISS index for top_k similar embeddings
    D, I = faiss_index.search(query_embedding, top_k)
    dense_scores = D.flatten()
    # Normalize dense scores similarly
    if dense_scores.max() - dense_scores.min() > 0:
        dense_scores_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-6)
    else:
        dense_scores_norm = dense_scores

    # Combine BM25 and dense scores using a weighted sum (alpha controls the contribution)
    combined_scores = []
    for idx in I.flatten():
        pos = np.where(I.flatten() == idx)[0][0]
        combined_score = alpha * bm25_scores_norm[idx] + (1 - alpha) * dense_scores_norm[pos]
        combined_scores.append(combined_score)
    combined_scores = np.array(combined_scores)

    # Select the best match based on the combined score
    best_idx = I.flatten()[np.argmax(combined_scores)]
    top_claims = df.iloc[I.flatten()].copy()
    top_claims["combined_score"] = combined_scores
    return best_idx, top_claims


def fact_check(query, bm25, faiss_index, dense_model, df, classifier, tokenizer, verbose=False):
    """
    Given a user query, this function:
      - Retrieves similar claims from the dataset (using BM25 and FAISS).
      - Uses the classifier to predict the veracity of the query.
      - Formats a system response.

    The response can be:
      - Concise (non-verbose): e.g., "If you are referring to a claim by [speaker] that [statement], it is categorically [predicted label]."
      - Verbose: with additional details about the supporting evidence.
    """
    best_idx, top_claims = retrieve_similar_claim(query, bm25, faiss_index, dense_model, df)
    retrieved_claim = df.iloc[best_idx]
    predicted_label = predict_claim_label(query, classifier, tokenizer)

    # Format the speaker name nicely (e.g., converting "garnet-coleman" to "Garnet Coleman")
    speaker = retrieved_claim['speaker']
    speaker = speaker.replace('-', ' ').title()

    # Dictionary to map raw predicted labels to more natural language phrases
    label_descriptions = {
        "pants-fire": "categorically false",
        "false": "false",
        "barely-true": "mostly false",
        "half-true": "partially true",
        "mostly-true": "mostly true",
        "true": "true"
    }

    if not verbose:
        # Concise response using a template
        response = (
            f"If you are referring to a claim by {speaker} that "
            f"{retrieved_claim['statement']}, it is {label_descriptions[predicted_label]}."
        )
    else:
        # Verbose response with additional supporting details
        response = (
            f"Claim: \"{query}\"\n"
            f"Predicted Label: {predicted_label}\n\n"
            "Supporting Evidence from the Dataset:\n"
            f"- Statement: \"{retrieved_claim['statement']}\"\n"
            f"- Speaker: {retrieved_claim['speaker']} ({retrieved_claim['job_title']})\n"
            f"- Context: {retrieved_claim['context']}\n"
            f"- Dataset Label: {retrieved_claim['label']}\n\n"
            f"If you are referring to the claim above, it is {label_descriptions[predicted_label]}."
        )
    return response


def main():
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="Fact-Checking System with LIAR Dataset")
    parser.add_argument("--query", type=str, help="Claim to fact-check", default=None)
    parser.add_argument(
        "--train_classifier",
        action="store_true",
        help="Train the classifier model from scratch (if not using a saved model)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output a detailed response (for debugging) rather than a succinct answer.",
    )
    args = parser.parse_args()

    # Load the dataset
    df = load_dataset(DATA_PATH)
    if args.verbose:
        print("Dataset loaded.")

    # Build BM25 index using the claim statements
    statements = df["statement"].tolist()
    bm25 = build_bm25_index(statements)
    if args.verbose:
        print("BM25 index built.")

    # Compute dense embeddings and build the FAISS index for semantic retrieval
    if args.verbose:
        print("Computing dense embeddings for FAISS index ...")
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = compute_dense_embeddings(dense_model, statements)
    faiss_index = build_faiss_index(embeddings)
    if args.verbose:
        print("FAISS index built.")

    # Load or train the classifier model
    if args.train_classifier:
        classifier, tokenizer = train_classifier(df)
    else:
        classifier, tokenizer = load_classifier()

    # Get the query either from command-line argument or prompt the user
    if args.query:
        query = args.query
    else:
        query = input("Enter a claim to fact-check: ")

    # Perform fact-checking and generate a response
    result = fact_check(query, bm25, faiss_index, dense_model, df, classifier, tokenizer, verbose=args.verbose)
    print("\n--- Fact-Checking Response ---\n")
    print(result)
    print('-' * 80)


if __name__ == "__main__":
    main()
