import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


# CSV files to embed (adjust if names differ)
DATA_FILES = [
    "fraud_articles.csv",
    "outseer_articles.csv",
    "Fraud_and_Scam_Data_Sharing_Outseer.csv",
]

# Sentence-transformers model
MODEL_NAME = "all-MiniLM-L6-v2"

# Where to save embedding outputs
OUTPUT_DIR = Path("embeddings")


def get_text_columns(path: Path, df: pd.DataFrame):
    """
    Decide which columns to use as text for a given file.
    This handles differences between fraud_articles and the others.
    """
    filename = path.name

    # File-specific overrides (tweak as needed)
    if filename == "fraud_articles.csv":
        # from your error: ['title', 'url', 'is_fraud_related']
        # here we just embed the title text
        return ["title"]

    # Generic rule for other files: prefer these if present
    preferred_candidates = [
        "content",
        "body",
        "text",
        "article_text",
        "description",
    ]

    text_cols = [c for c in preferred_candidates if c in df.columns]

    # Always include title if present
    if "title" in df.columns and "title" not in text_cols:
        text_cols.insert(0, "title")

    # If still nothing, fall back to "anything that looks text-like"
    if not text_cols:
        non_text = {
            "url",
            "link",
            "id",
            "article_id",
            "is_fraud_related",
            "label",
        }
        text_cols = [c for c in df.columns if c not in non_text]

    if not text_cols:
        raise ValueError(
            f"Could not determine any text columns for {filename}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"Using text columns for {filename}: {text_cols}")
    return text_cols


def load_articles(path: Path) -> pd.DataFrame:
    """Load a CSV and create a single text column for embedding."""
    df = pd.read_csv(path)

    text_cols = get_text_columns(path, df)

    # Combine chosen text columns into one field
    df["text_for_embedding"] = (
        df[text_cols].fillna("").agg(" ".join, axis=1).str.strip()
    )

    # Drop rows with completely empty text
    df = df[df["text_for_embedding"] != ""]

    # Ensure each row has some kind of ID
    if "article_id" not in df.columns:
        df["article_id"] = df.index.astype(str)

    return df


def create_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for the text column."""
    texts = df["text_for_embedding"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    for filename in DATA_FILES:
        path = Path(filename)
        if not path.exists():
            print(f"Skipping {filename}: file not found")
            continue

        print(f"\nProcessing {filename}...")
        df = load_articles(path)
        embeddings = create_embeddings(df, model)

        print(f"Generated embeddings shape: {embeddings.shape}")

        stem = path.stem
        npy_path = OUTPUT_DIR / f"{stem}_embeddings.npy"
        parquet_path = OUTPUT_DIR / f"{stem}_with_embeddings.parquet"

        # Save raw embeddings as .npy
        np.save(npy_path, embeddings)

        # Save combined data + embeddings as parquet
        emb_df = pd.DataFrame(embeddings)
        emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]

        combined = pd.concat(
            [df.reset_index(drop=True), emb_df.reset_index(drop=True)],
            axis=1,
        )
        combined.to_parquet(parquet_path, index=False)

        print(
            f"Saved embeddings:\n"
            f"  - {npy_path}\n"
            f"  - {parquet_path}"
        )


if __name__ == "__main__":
    main()
