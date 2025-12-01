import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# 1. Load data and basic info
# -----------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv('outseer_articles.csv')

    # Make sure published is a datetime
    df["published"] = pd.to_datetime(df["published"], errors="coerce")

    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nDate range:")
    print("  Min:", df["published"].min())
    print("  Max:", df["published"].max())
    print()
    return df


# -----------------------------
# 2. Tokenization + utilities
# -----------------------------
STOPWORDS = {
    "the", "and", "to", "of", "a", "in", "for", "on", "with", "is", "this",
    "that", "as", "it", "at", "by", "an", "are", "be", "from", "more",
    "your", "you", "our", "outseer", "blog", "what", "these", "need",
    "post", "into", "can", "how", "why", "will", "have", "has"
}


def combine_text(df: pd.DataFrame) -> pd.Series:
    # Combine title + summary into one text field
    return (df["title"].astype(str) + " " + df["summary"].astype(str)).str.lower()


def tokenize(text_series: pd.Series):
    all_text = " ".join(text_series)
    tokens = re.findall(r"[a-zA-Z0-9\-]+", all_text)
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 3]
    return filtered


# -----------------------------
# 3. Keyword frequency plot
# -----------------------------
def plot_keyword_frequencies(tokens, top_n: int = 20):
    freq = Counter(tokens).most_common(top_n)
    labels, counts = zip(*freq)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)  # simple bar chart
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Keyword Frequencies")
    plt.tight_layout()
    plt.show()

    print("\nTop keyword frequencies:")
    for word, count in freq:
        print(f"{word:15} {count}")


# -----------------------------
# 4. TF-IDF + KMeans clustering
# -----------------------------
def cluster_articles(text_series: pd.Series, n_clusters: int = 3):
    print("\nRunning TF-IDF + KMeans clustering...")

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(text_series)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    return labels


def print_cluster_examples(df: pd.DataFrame, cluster_labels, n_examples: int = 5):
    df = df.copy()
    df["cluster"] = cluster_labels

    print("\nCluster examples:")
    for c in sorted(df["cluster"].unique()):
        print(f"\n=== Cluster {c} ===")
        sample = df[df["cluster"] == c].head(n_examples)
        for i, row in sample.iterrows():
            print("-", row["title"])


# -----------------------------
# 5. Top 10 terms per year
# -----------------------------
def top_terms_per_year(df: pd.DataFrame, top_n: int = 10):
    print("\nTop terms per year:")

    df_year = df.dropna(subset=["published"]).copy()
    df_year["year"] = df_year["published"].dt.year

    for year, group in df_year.groupby("year"):
        text_series = combine_text(group)
        tokens = tokenize(text_series)
        freq = Counter(tokens).most_common(top_n)

        print(f"\nYear {year}:")
        for word, count in freq:
            print(f"  {word:15} {count}")


# -----------------------------
# 6. Simple summary paragraph
# -----------------------------
def print_summary():
    summary = (
        "After analyzing the Outseer articles, three major fraud trends stand out. "
        "First, there is a strong focus on digital payment fraud and the need for "
        "secure authentication around online transactions, especially using tools "
        "like EMV 3-D Secure. Second, many posts highlight social-engineering scams "
        "such as phishing, refund scams, and account takeover attacks that target "
        "consumers directly. Third, there is a growing emphasis on AI and machine "
        "learning for fraud detection, where models analyze behavior and user intent "
        "to stop fraud before it happens."
    )
    print("\n=== Summary Paragraph ===\n")
    print(summary)
    print()


# -----------------------------
# Main runner
# -----------------------------
def main():
    # 1. Load data
    df = load_data("outseer_articles.csv")

    # 2. Combine and tokenize text
    text_series = combine_text(df)
    tokens = tokenize(text_series)

    # 3. Plot keyword frequencies
    plot_keyword_frequencies(tokens, top_n=20)

    # 4. Cluster articles and print examples
    cluster_labels = cluster_articles(text_series, n_clusters=3)
    print_cluster_examples(df, cluster_labels, n_examples=5)

    # 5. Top terms per year
    top_terms_per_year(df, top_n=10)

    # 6. Print summary paragraph
    print_summary()


if __name__ == "__main__":
    main()