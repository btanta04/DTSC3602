<<<<<<< HEAD
# Fraud Risk Analysis

**GitHub Repository:** https://github.com/btanta04/DTSC3602

## Authors: 

- Tyler Buck
- Jorge Andrade
- Alli Borland
- Bisher Tanta


## Project Overview
This project explores the intersection of cybersecurity, NLP, and applied data science, using real world fraud related datasets to simulate industry level fraud detection scenarios. The work was developed as part of a collaborative academic initiative where teams were tasked with exploring analytical approaches that could aid financial institutions such as USAA in identifying emerging fraud patterns across digital platforms. 


## Technical Approach
Using a dataset inspired by USAA's internal fraud concerns alongside public data from Outseer's Fraud & Payment Blog, we built an end-to-end pipeline for data collection, analysis, topic modeling, and interactive visualization. Web scraping was performed using both BeautifulSoup-Based scrapers and a Gemini-powered AI scraper, which produced concise summaries for rapid insight generation. The processed articles were then structured into CSV files and analyzed using NLP methods for fraud keywords detection, summarization, pattern recognition, and topic emergence.


To uncover deeper fraud trends, BERTopic was used to perform unsupervised topic modeling, revealing high-level patterns across the USAA dataset and industry wide fraud articles. Additionally, embed-articles.py converted each record into vector embeddings using SentenceTransformer(all-mini-L6-v2), enabling clustering, semantic similarity searches, and machine learning integration. A Streamlit dashboard was then built to allow analysts or future stakeholders to interact with fraud topics, AI summaries, emerging trends, and embedding clusters in real time. This final component transforms the work from static research into an interactive decision-support tool. 


In summary: Highlighting the potential of data science to support real fraud/scam risk analysis. 

# 1. Clone the Repository
git clone https://github.com/btanta04/DTSC3602
cd DTSC3602
git checkout nlp-analysis

Branch              Purpose
main            Final and Merged work
nlp-analysis    Fraud detection using NLP & embeddings
Scraper-test    Web scraping modules
dashboard       Streamlit UI development


# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment variables
echo "GEMINI_API_KEY=your_actual_gemini_key_here" > .env

# 4. Run the Streamlit dashboard
streamlit run dashboard.py

## Application Demo 

![Fraud & Scam Analyzer Dashboard](https://github.com/user-attachments/assets/8ab48fe8-31a3-4489-b91c-a7e2a51873e5)

## Public Demo Analytics & Findings!

### Top Fraud-Related Keywords
https://aborland123--streamlit-outseer-deployment-serve.modal.run/

![Top Keywords](https://github.com/user-attachments/assets/16d90cb9-3c02-43dd-ac61-2d67c173bc4a)

### Fraud Likelihood Distribution  
![Fraud Scores](https://github.com/user-attachments/assets/cd5a550f-9823-4048-a07b-84a65619948c)

### Publication Timeline
![Articles Timeline](https://github.com/user-attachments/assets/85f99451-1957-4f11-8c8e-6b2b82f74040)

