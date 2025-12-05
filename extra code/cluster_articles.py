import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---- CONFIG ----
# Embeddings for the fraud articles
EMBEDDINGS_PATH = "embeddings/fraud_articles_embeddings.npy"

# Metadata for those articles (same order as embeddings)
METADATA_PATH = "fraud_articles.csv"

# Output file with cluster labels added
OUTPUT_PATH = "clustered_articles.csv"
# ---- END CONFIG ----


def load_data():
    """
    Load embeddings from .npy and article metadata from CSV.
    Ensures the lengths match.
    """
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(METADATA_PATH)

    if len(df) != embeddings.shape[0]:
        raise ValueError(
            "Row mismatch between metadata and embeddings: "
            f"{len(df)} rows in CSV vs {embeddings.shape[0]} "
            "rows in embeddings."
        )

    return df, embeddings


def find_best_k(embeddings, k_values=None):
    """
    Try several k values and print silhouette scores
    so you can pick a reasonable number of clusters.
    """
    if k_values is None:
        k_values = [3, 4, 5, 6, 7, 8]

    print("Testing k values:", k_values)
    scores = {}

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            n_init="auto",
            random_state=42
        )
        labels = model.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores[k] = score
        print("k =", k, "silhouette =", round(score, 3))

    return scores


def run_clustering(k=5):
    """
    Run final KMeans clustering with chosen k, attach labels
    to the dataframe, and save to CSV.
    """
    df, embeddings = load_data()

    model = KMeans(
        n_clusters=k,
        n_init="auto",
        random_state=42
    )
    labels = model.fit_predict(embeddings)

    df["cluster"] = labels
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n✅ Clustering complete.")
    print("Saved file:", OUTPUT_PATH)
    print("\nCluster sizes:")
    print(df["cluster"].value_counts().sort_index())

    # If there is a text/title column, show a few samples per cluster
    for col in ["title", "summary", "content", "article_text"]:
        if col in df.columns:
            print(f"\nShowing sample {col} values per cluster:")
            for cluster_id in sorted(df["cluster"].unique()):
                print("\n--- Cluster", cluster_id, "---")
                sample = df[df["cluster"] == cluster_id][col].head(5)
                for t in sample:
                    print("-", str(t)[:200])
            break


if __name__ == "__main__":
    df_meta, emb = load_data()

    # 1) Help you choose k
    find_best_k(emb)

    # 2) Run final clustering (change k after you see scores)
    print("\nRunning final clustering with k = 5\n")
    run_clustering(k=5)
