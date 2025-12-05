import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# PATHS 
LOCAL_BASE = Path(__file__).resolve().parent

if Path("/root").exists():
    BASE = Path("/root")
else:
    BASE = LOCAL_BASE

EMBED_PARQUET = BASE / "embeddings" / "outseer_articles_with_embeddings.parquet"
RAW_CSV = BASE / "outseer_articles.csv"
KW_CSV = BASE / "bertopic_keywords_weights.csv"

# LOADING DATA
@st.cache_data
def load_articles():
    if EMBED_PARQUET.exists():
        df = pd.read_parquet(EMBED_PARQUET)
    else:
        df = pd.read_csv(RAW_CSV)

    df = df.iloc[1:].reset_index(drop=True)

    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
    elif "date" in df.columns:
        df["published"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["published"] = pd.NaT

    return df


@st.cache_data
def load_keywords():
    if KW_CSV.exists():
        kw_df = pd.read_csv(KW_CSV)
        expected_cols = {"keyword", "weight"}
        missing = expected_cols - set(kw_df.columns)
        if missing:
            st.warning(f"Keyword file missing columns: {missing}")
        return kw_df
    else:
        st.warning(
            "bertopic_keywords_weights.csv not found. "
            "Top keyword chart will be empty."
        )
        return pd.DataFrame(columns=["keyword", "weight"])


df = load_articles()
kw_df = load_keywords()

# EMBEDDING-BASED FRAUD SCORE
EMB_COLS = [c for c in df.columns if c.startswith("emb_")]

if not EMB_COLS:
    st.warning(
        "No embedding columns (emb_0, emb_1, ...) found. "
        "Fraud scores will not be computed. "
        "Make sure you ran the embeddings script and "
            "are loading embeddings/outseer_articles_with_embeddings.parquet."
    )
    df["fraud_score"] = None
else:
    text_cols = [
        c for c in ["summary", "full_text", "doc", "text_for_embedding"]
        if c in df.columns
    ]
    if text_cols:
        combined_text = (
            df[text_cols]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.lower()
        )
    else:
        combined_text = pd.Series([""] * len(df))

    # Articles that look "fraud-y" based on text
    fraud_mask = combined_text.str.contains("fraud", na=False)
    fraud_df = df[fraud_mask]

    if fraud_df.empty:
        st.warning(
            "No articles containing the word 'fraud' were found. "
            "Fraud reference embedding cannot be computed."
        )
        df["fraud_score"] = None
    else:
        fraud_embs = fraud_df[EMB_COLS].to_numpy(dtype=float)
        fraud_reference = fraud_embs.mean(axis=0)

        norm = np.linalg.norm(fraud_reference)
        if norm == 0:
            fraud_reference = None
            st.warning("Fraud reference embedding has zero norm.")
            df["fraud_score"] = None
        else:
            fraud_reference = fraud_reference / norm
            all_embs = df[EMB_COLS].to_numpy(dtype=float)
            sims = all_embs @ fraud_reference  

            sims = np.clip(sims, 0, 1)
            df["fraud_score"] = np.round(sims * 100, 2)

# TEXT FIELD FOR CLUSTERING 
TEXT_COLS = ["summary", "full_text", "doc", "text_for_embedding"]
available_cols = [c for c in TEXT_COLS if c in df.columns]

if available_cols:
    df["text_for_clustering"] = (
        df[available_cols]
        .fillna("")
        .agg(" ".join, axis=1)
    )
else:
    df["text_for_clustering"] = ""


@st.cache_data
def compute_kmeans_clusters(input_df: pd.DataFrame):
    work_df = input_df.copy()
    work_df["text_for_clustering"] = work_df["text_for_clustering"].fillna("")
    non_empty_mask = work_df["text_for_clustering"].str.strip() != ""
    work_df = work_df[non_empty_mask].reset_index(drop=True)

    if work_df.empty:
        return input_df.assign(cluster=np.nan), {}

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    X = vectorizer.fit_transform(work_df["text_for_clustering"])

    k = 3
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)

    work_df["cluster"] = km.labels_

    # Map back to original 
    result_df = input_df.copy()
    result_df["cluster"] = np.nan
    result_df.loc[work_df.index, "cluster"] = work_df["cluster"].values

    # Extract keywords per cluster
    terms = vectorizer.get_feature_names_out()
    order = km.cluster_centers_.argsort()[:, ::-1]

    cluster_keywords = {}
    for i in range(k):
        top_terms = [terms[ind] for ind in order[i, :15]]
        cluster_keywords[i] = top_terms

    return result_df, cluster_keywords


# SIDEBAR 

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    ["Overview", "Article Explorer"]
)

# OVERVIEW PAGE

if page == "Overview":
    st.title("Outseer Overview")

    # Top Keywords (from BERTopic fraud keywords)
    st.subheader("Top Fraud-Related Keywords")

    if not kw_df.empty and "keyword" in kw_df.columns and "weight" in kw_df.columns:
        # Aggregate weights across topics
        kw_agg = (
            kw_df.groupby("keyword")["weight"]
            .sum()
            .reset_index()
        )
        top_kw = kw_agg.sort_values("weight", ascending=False).head(20)

        chart_keywords = (
            alt.Chart(top_kw)
            .mark_bar()
            .encode(
                x=alt.X("weight:Q", title="Total Topic Weight"),
                y=alt.Y("keyword:N", sort="-x", title="Keyword"),
                tooltip=["keyword", "weight"]
            )
        )

        st.altair_chart(chart_keywords, use_container_width=True)
        st.caption(
            "Keywords extracted from BERTopic, filtered to fraud-related terms."
        )
    else:
        st.info("No BERTopic keyword data available.")

    # Fraud Score Distribution

    st.subheader("Fraud Likelihood Distribution")

    df_fraud = df[df["fraud_score"].notna()]

    if not df_fraud.empty:
        fraud_chart = (
            alt.Chart(df_fraud)
            .mark_bar()
            .encode(
                x=alt.X(
                    "fraud_score:Q",
                    bin=alt.Bin(maxbins=20),
                    title="Fraud Score"
                ),
                y=alt.Y("count()", title="Number of Articles"),
                tooltip=["count()"]
            )
        )
        st.altair_chart(fraud_chart, use_container_width=True)
    else:
        st.info("Fraud scores are not available.")


    # KMEANS CLUSTERING 

    st.subheader("KMeans Topic Clusters (k = 3)")

    with st.spinner("Computing clusters (k=3)..."):
        clustered_df, cluster_keywords = compute_kmeans_clusters(df)

    if not cluster_keywords:
        st.info("No text available for clustering.")
    else:
        st.markdown("##### Cluster Themes & Top Keywords")

        cluster_names = {}
        for cid, kw_list in cluster_keywords.items():
            suggested_name = " / ".join([w.title() for w in kw_list[:3]])
            cluster_names[cid] = suggested_name

            st.markdown(f"**Cluster {cid} — {suggested_name}**")
            st.write(", ".join(kw_list[:15]))

        # Cluster sizes
        st.markdown("##### Cluster Sizes (Total Articles)")

        size_df = (
            clustered_df.dropna(subset=["cluster"])
            .groupby("cluster")["title"]
            .count()
            .reset_index()
            .rename(columns={"title": "count"})
        )

        if not size_df.empty:
            size_chart = (
                alt.Chart(size_df)
                .mark_bar()
                .encode(
                    x=alt.X("cluster:N", title="Cluster"),
                    y=alt.Y("count:Q", title="Number of Articles"),
                    tooltip=["cluster", "count"],
                )
            )
            st.altair_chart(size_chart, use_container_width=True)
        else:
            st.info("No cluster assignments to show.")


        st.markdown("##### Cluster Trends Over Time")

        df_ct = clustered_df.copy()
        df_ct = df_ct.dropna(subset=["cluster", "published"])

        if not df_ct.empty:
            # Month-level time bucket
            df_ct["year_month"] = df_ct["published"].dt.to_period("M").dt.to_timestamp()

            time_cluster_counts = (
                df_ct.groupby(["year_month", "cluster"])["title"]
                .count()
                .reset_index()
                .rename(columns={"title": "count"})
            )

            # per-cluster article counts over time
            line_chart = (
                alt.Chart(time_cluster_counts)
                .mark_line(point=True)
                .encode(
                    x=alt.X("year_month:T", title="Month"),
                    y=alt.Y("count:Q", title="Number of Articles"),
                    color=alt.Color(
                        "cluster:N",
                        title="Cluster",
                        legend=alt.Legend(
                            labelExpr="datum.label",
                        ),
                    ),
                    tooltip=["year_month:T", "cluster:N", "count:Q"],
                )
            )
            st.altair_chart(line_chart, use_container_width=True)


# ARTICLE EXPLORER PAGE
elif page == "Article Explorer":
    st.title("Article Explorer")

    # Sidebar search
    st.sidebar.subheader("Search Articles")
    search_title = st.sidebar.text_input("Search by Title")
    search_keyword = st.sidebar.text_input("Search by Keyword (title / text)")

    df_display = df.copy()

    #  Filter by title 
    if search_title:
        df_display = df_display[
            df_display["title"].str.contains(search_title, case=False, na=False)
        ]

    # Filter by keyword in title + text fields 
    if search_keyword:
        keyword = search_keyword.strip()
        text_cols = [
            c for c in ["title", "summary", "full_text", "doc", "text_for_embedding"]
            if c in df_display.columns
        ]

        if text_cols:
            combined = (
                df_display[text_cols]
                .fillna("")
                .agg(" ".join, axis=1)
                .str.contains(keyword, case=False, na=False)
            )
            df_display = df_display[combined]

    # Sort by published date
    df_display = df_display.sort_values("published", ascending=False)

    if df_display.empty:
        st.info("No articles match your search.")
    else:
        selected_title = st.selectbox("Choose an article", df_display["title"])
        article = df_display[df_display["title"] == selected_title].iloc[0]

        st.header(article["title"])

        if "url" in article and pd.notna(article["url"]):
            st.markdown(f"[Read Full Article]({article['url']})")

        if pd.notna(article.get("published", pd.NaT)):
            st.write(f"Published: {article['published'].date()}")

        st.subheader("Fraud Likelihood")
        score = article.get("fraud_score", None)
        if score is None or (isinstance(score, float) and np.isnan(score)):
            st.info("Fraud reference embedding not available.")
        else:
            st.metric("Fraud Score", f"{score}%")
            st.progress(float(score) / 100.0)


        st.subheader("Full Text")
        with st.expander("Show full article text"):
            if "full_text" in article:
                st.write(article["full_text"])
            else:
                st.write("No full text available.")

    
        st.subheader("Summary")
        if "summary" in article:
            st.write(article["summary"])
        else:
            st.write("No summary available.")

    