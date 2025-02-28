# Fact-Checking System Using the LIAR Dataset

## Model Overview

This model is a fact‐checking system built on the LIAR dataset. It combines both sparse (BM25) and dense (FAISS) retrieval techniques to locate relevant claims and uses a fine-tuned BERT-based classifier to predict the veracity of user-provided statements. The final output is generated via a template that provides a clear, concise verdict along with supporting details such as the original claim, speaker, and context.

## Training Data

- Dataset: LIAR dataset, which contains thousands of labeled political statements along with metadata such as speaker information, job titles, and context.
- Labels: The system classifies statements into six categories: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, and `true`.

## Model Details

### Retrieval Component:

- Sparse Retrieval: BM25 index over claim statements.
- Dense Retrieval: FAISS index built on embeddings from the "all-MiniLM-L6-v2" SentenceTransformer.
- Score Fusion: A weighted sum of BM25 and dense retrieval scores is used to identify the most relevant similar claim.

### Classification Component:

- Model: [unshDee/liar_qa](https://huggingface.co/unshDee/liar_qa) - BERT (bert-base-uncased) fine-tuned on the LIAR dataset for six-class veracity prediction. 
- Output: The predicted label (e.g., "false") is used in the final fact-checking response.

### Response Generation:
A template-based system formats the final output.

## To run

### Install dependencies

```commandline
pip install -r requirements.txt
```

### Classification Model

On the first run, the model will be downloaded (to a folder named `classifier_model`) from the Hugging Face model hub.

[unshDee/liar_qa](https://huggingface.co/unshDee/liar_qa)

### via Terminal

```commandline
python main.py --query "Is it true that Barack Obama was born in Kenya?"
python main.py --query "Is it true that the COVID-19 vaccine contains microchips?"
python main.py --query "Is it true that climate change is a hoax?"
python main.py --query "Is it true that 5G networks cause severe health issues?"
python main.py --query "Is it true that illegal immigrants are the primary cause of crime in the United States?"
python main.py --query "Is it true that transgender individuals in the U.S. have a 1-in-12 chance of being murdered?"
python main.py --query "Is it true that increasing the minimum wage will lead to massive job losses?"
python main.py --query "Is it true that the U.S. government is hiding evidence of alien encounters?"
python main.py --query "Is it true that global warming is just a natural cycle and not influenced by human activity?"
python main.py --query "Is it true that immigrants drain public resources and are a burden on the economy?"

```
### via Streamlit

```commandline
streamlit run app.py
```

---

## To evaluate the performance of the fact-checking system:

- **Assess the Classifier**: Use accuracy, precision, recall, and F1-score on a held-out test set from the LIAR dataset to measure how well the model distinguishes between the six labels.
- **Evaluate Retrieval Effectiveness**: Compute retrieval metrics like recall@k and mean reciprocal rank (MRR) to ensure that the BM25+FAISS hybrid returns the most relevant supporting claims.
- **User Testing**: Perform usability studies with real users to determine if the final natural language responses are clear, informative, and useful in verifying claims.