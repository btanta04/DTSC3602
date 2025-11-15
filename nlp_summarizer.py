import pandas as pd
import spacy
import pytextrank

# Load NLP model
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# Load your CSV
df = pd.read_csv('outseer_articles.csv')

print(f" Total articles: {len(df)}")

# Filter only articles with good content in summary column
articles_with_content = df[df['summary'].str.len() > 50]

print(f" Articles with good content: {len(articles_with_content)}")
print(f" Empty articles skipped: {len(df) - len(articles_with_content)}")

results = []

print("\n🚀 Processing articles with NLP...")

for i, row in articles_with_content.iterrows():
    article_text = str(row['summary'])
    
    print(f" Processing: {row['title'][:50]}...")
    
    # Use TextRank to extract key sentences
    doc = nlp(article_text)
    
    key_sentences = []
    for sent in doc._.textrank.summary(limit_sentences=2):
        key_sentences.append(str(sent))
    
    # Create NLP summary
    nlp_summary = ' '.join(key_sentences)
    
    results.append({
        'title': row['title'],
        'url': row['url'],
        'original_summary': article_text,
        'nlp_summary': nlp_summary,
        'original_length': len(article_text),
        'nlp_length': len(nlp_summary)
    })

# Save results
output_df = pd.DataFrame(results)
output_df.to_csv('quality_nlp_summaries.csv', index=False)

print(f" Done! Processed {len(results)} quality articles")
print(f" Saved to: quality_nlp_summaries.csv")