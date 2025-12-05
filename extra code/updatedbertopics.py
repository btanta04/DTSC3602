import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


def main():
    df = pd.read_csv("outseer_articles.csv")
    df = df.iloc[1:].reset_index(drop=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "published" in df.columns:
        df["date"] = pd.to_datetime(df["published"], errors="coerce")
    else:
        df["date"] = pd.NaT

    #Build combined document text: title + summary + full_text
    for col in ["title", "summary", "full_text"]:
        if col not in df.columns:
            df[col] = ""

    df["doc"] = (
        df["title"].fillna("").astype(str)
        + ". "
        + df["summary"].fillna("").astype(str)
        + ". "
        + df["full_text"].fillna("").astype(str)
    )

    df = df[df["doc"].str.strip() != ""].reset_index(drop=True)

    # Stopword set (for keyword post-filtering, not for docs)
    stop_words = set(ENGLISH_STOP_WORDS)
    custom_words = {
        "the", "to", "and", "of", "in", "for", "is", "that",
        "with", "on", "as", "by", "it", "this", "these", "those",
        "are", "be", "from", "at", "an", "or", "we", "you", "our",
        "outseer", "blog", "article",
    }
    stop_words.update(custom_words)

    # lowercase + remove non-letters
    df["doc_clean"] = (
        df["doc"]
        .fillna("")
        .str.replace(r"[^a-zA-Z ]+", " ", regex=True)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    docs_clean = df["doc_clean"].tolist()

    print(f"Number of documents used for BERTopic: {len(docs_clean)}")

    # Load embedding model
    print("Loading sentence transformer model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    #  Vectorizer 
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )

    # using BERTopic
    print("Fitting BERTopic model on documents...")
    topic_model = BERTopic(
        embedding_model=embed_model,
        vectorizer_model=vectorizer_model,
        top_n_words=50,   
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs_clean)

    # Attach topics and probabilities back to each article
    df["topic"] = topics
    df["probability"] = probs
    df.to_csv("outseer_articles_with_topics.csv", index=False)
    print("Saved article-level topics → outseer_articles_with_topics.csv")

    # Topic overview
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv("bertopic_topics_overview.csv", index=False)
    print("Saved topic overview → bertopic_topics_overview.csv")

    # Extract keywords + weights for each topic
    keyword_rows = []
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)   
        for rank, (word, weight) in enumerate(words, start=1):
            keyword_rows.append(
                {
                    "topic": topic_id,
                    "rank": rank,
                    "keyword": word,
                    "weight": weight,
                }
            )

    kw_df = pd.DataFrame(keyword_rows)

    kw_df["keyword"] = (
        kw_df["keyword"]
        .str.lower()
        .str.replace(r"[^a-zA-Z ]+", "", regex=True)
        .str.strip()
    )
    kw_df = kw_df[kw_df["keyword"] != ""]
    kw_df = kw_df[~kw_df["keyword"].isin(stop_words)]

    # Save ALL keywords
    kw_df.to_csv("bertopic_keywords_all.csv", index=False)
    print("All topic keywords → bertopic_keywords_all.csv")

    # Fraud-only subset
    fraud_pattern = (
        r"(fraud|frauds?|fraudster[s]?|"
        r"scam[s]?|"
        r"phish(?:ing)?|"
        r"mule[s]?|"
        r"launder(?:ing)?|"
        r"chargeback[s]?|"
        r"breach(?:es)?|"
        r"identity|"
        r"theft)"
    )
    kw_fraud = kw_df[kw_df["keyword"].str.contains(fraud_pattern, case=False, regex=True)]

    kw_fraud.to_csv("bertopic_keywords_weights.csv", index=False)
    print("Fraud-only keyword weights → bertopic_keywords_weights.csv")

    print("\nTop fraud-related keywords:")
    print(
        kw_fraud.groupby("keyword")["weight"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
    )


if __name__ == "__main__":
    main()