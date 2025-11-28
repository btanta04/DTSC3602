Cyber Risk Analyst 

Authors: 

Tyler Buck
Jorge Andrade
Alli B
btanta04

This project explores the intersection of cybersecurity, NLP, and applied data science, using real world fraud related datasets to simulate industry level fraud detection scenarios. The work was developed as part of a collaborative academic initiative where teams were tasked with exploring analytical approaches that could aid financial institutions such as USAA in identifying emerging fraud patterns across digital platforms. 


Using a dataset inspired by USAA's internal fraud concerns alongside public data from Outseer's Fraud & Payment Blog, we built an end-to-end pipeline for data collection, analysis, topic modeling, and interactive visualization. Web scraping was performed using both BeautifulSoup-Based scrapers and a Gemini-powered AI scraper, which produced concise summaries for rapid insight generation. The processed articles were then structured into CSV files and analyzed using NLP methods for fraud keywords detection, summarization, pattern recognition, and topic emergence.


To uncover deeper fraud trends, BERTopic was used to perform unsupervised topic modeling, revealing high-level patterns across the USAA dataset and industry wide fraud articles. Additionally, embed-articles.py converted each record into vector embeddings using SentenceTransformer(all-mini-L6-v2), enabling clustering, semantic similarity searches, and machine learning integration. 
A Streamlit dashboard was then built to allow analysts or future stakeholders to interact with fraud topics, AI summaries, emerging trends, and embedding clusters in real time. This final component transforms the work from static research into an interactive decision-support tool. 


In summary: Highlighting the potential of data science to support real cyber risk analysis. 

git clone https://github.com/btanta04/DTSC3602
cd DTSC3602

git checkout nlp-analysis

Branch              Purpose
main            Final and Merged work
nlp-analysis    Fraud detection using NLP & embeddings
Scraper-test    Web scraping modules
dashboard       Streamlit UI development

# Clone the repo
git clone https://github.com/btanta04/DTSC3602
cd DTSC3602
git checkout nlp-analysis

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp example.env .env

# Run the Streamlit dashboard
streamlit run dashboard.py



GEMINI_API_KEY=your_api_key_here
SCRAPER_MODE=gemini     # or bs4
MODEL_NAME=all-mini-L6-v2



USAA Inspired problem -> Data Scraping -> AI Summaries -> NLP Exploratory Analysis -> BERTopic Modeling -> Embeddings -> Streamlit Visualization


Folder Structure

fraud-detection/
│── data/
│   ├── fraud_articles.csv
│   ├── outseer_articles.csv
│   ├── articles_with_smart_summaries.csv
│── embeddings/
│   └── *.npy
│── bs4Scraper.py
│── aiscrapper.py
│── embed_articles.py
│── pdfscraper.py
│── nlp_analyzer.py
│── nlp_summarizer.py
│── dashboard.py
│── requirements.txt
│── README.md
│── README_NLP.md

