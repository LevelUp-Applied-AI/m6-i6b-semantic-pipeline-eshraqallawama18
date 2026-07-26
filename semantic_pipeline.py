"""
Module 6 Week B — Integration: NER + Embeddings Semantic Pipeline

Build an end-to-end NLP pipeline that combines named entity recognition
(Week A) with embedding-based semantic search (Week B) on a climate
article corpus.
"""

import numpy as np
import pandas as pd
import spacy
import torch

from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess(filepath):
    """Load the climate articles dataset and prepare texts for processing."""

    df = pd.read_csv(filepath)

    df = df.dropna(subset=["text"])

    df["text"] = df["text"].astype(str)

    if "language" in df.columns:
        df = df[df["language"] == "en"]

    df = df.reset_index(drop=True)

    return df


def run_ner(texts):

    nlp = spacy.load("en_core_web_sm")

    entities = []

    for index, text in enumerate(texts):

        doc = nlp(text)

        for ent in doc.ents:
            entities.append(
                {
                    "text_index": index,
                    "entity_text": ent.text,
                    "entity_label": ent.label_
                }
            )

    return pd.DataFrame(
        entities,
        columns=[
            "text_index",
            "entity_text",
            "entity_label"
        ]
    )

def compute_embeddings(texts, tokenizer, model):

    embeddings = []

    model.eval()

    with torch.no_grad():

        for text in texts:

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            outputs = model(**inputs)

            hidden = outputs.last_hidden_state

            embedding = hidden.mean(dim=1)

            embeddings.append(
                embedding.squeeze().numpy()
            )

    return np.array(embeddings)

def semantic_search(query, corpus_embeddings, corpus_texts, top_k=5):

    scores = cosine_similarity(
        query.reshape(1, -1),
        corpus_embeddings
    )[0]


    indices = np.argsort(scores)[::-1][:top_k]


    results = []

    for i in indices:
        results.append(
            (
                corpus_texts[i],
                float(scores[i])
            )
        )

    return results

def enrich_with_entities(search_results, entity_df, corpus_texts):

    enriched = []

    for text, score in search_results:

        index = corpus_texts.index(text)

        rows = entity_df[
            entity_df["text_index"] == index
        ]

        entities = []

        for _, row in rows.iterrows():
            entities.append(
                {
                    "text": row["entity_text"],
                    "label": row["entity_label"]
                }
            )


        enriched.append(
            {
                "text": text,
                "similarity": score,
                "entities": entities
            }
        )


    return enriched

def demonstrate_pipeline(corpus_df, entity_df, embeddings, queries,
                         tokenizer, model):

    results = {}

    corpus_texts = corpus_df["text"].tolist()


    for query in queries:

        query_embedding = compute_embeddings(
            [query],
            tokenizer,
            model
        )[0]


        search_results = semantic_search(
            query_embedding,
            embeddings,
            corpus_texts
        )


        enriched = enrich_with_entities(
            search_results,
            entity_df,
            corpus_texts
        )


        results[query] = enriched


    return results

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    # Load and preprocess
    df = load_and_preprocess("data/climate_articles.csv")
    if df is not None:
        texts = df["text"].tolist()
        print(f"Loaded {len(texts)} texts")

        # NER
        entities = run_ner(texts)
        if entities is not None:
            print(f"Extracted {len(entities)} entities")

        # Embeddings
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        model.eval()
        embs = compute_embeddings(texts, tokenizer, model)
        if embs is not None:
            print(f"Embedding matrix shape: {embs.shape}")

        # Demo queries
        with open("data/example_queries.txt") as f:
            queries = [line.strip() for line in f if line.strip()]

        if embs is not None and entities is not None:
            results = demonstrate_pipeline(
                df, entities, embs, queries, tokenizer, model
            )
            if results:
                for q, enriched in results.items():
                    print(f"\nQuery: {q}")
                    for r in enriched[:3]:
                        print(f"  Score: {r['similarity']:.4f}")
                        print(f"  Text: {r['text'][:100]}...")
                        print(f"  Entities: {r['entities'][:5]}")
