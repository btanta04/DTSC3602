import pandas as pd
import spacy
import pytextrank

# Load spaCy with TextRank
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# Load your articles
print("Loading articles from CSV...")
df = pd.read_csv('fraud_articles.csv')

# Function to generate smart summaries
def generate_smart_summary(text, max_sentences=2):
    """Generate summary using TextRank algorithm"""
    if not isinstance(text, str) or len(text) < 100:
        return "Not enough content to summarize"
    
    try:
        doc = nlp(text)
        
        # Get the most important sentences
        summary_sentences = []
        for sent in doc._.textrank.summary(limit_phrases=3, limit_sentences=max_sentences):
            summary_sentences.append(str(sent).strip())
        
        return ' '.join(summary_sentences)
    
    except Exception as e:
        return f"Error generating summary: {e}"

# Process each article
print("\n Generating Smart Summaries...")
print("=" * 50)

results = []

for i, row in df.iterrows():
    title = row['title']
    content = row.get('content', '') or row.get('text', '')
    
    print(f"\n📰 Article {i+1}: {title[:60]}...")
    print(f"   Original length: {len(content)} characters")
    
    # Generate smart summary
    summary = generate_smart_summary(content)
    
    print(f"   Summary: {summary[:100]}...")
    print(f"   Summary length: {len(summary)} characters")
    
    # Save results
    results.append({
        'title': title,
        'url': row['url'],
        'original_content_length': len(content),
        'smart_summary': summary,
        'summary_length': len(summary)
    })

# Create new DataFrame with summaries
summary_df = pd.DataFrame(results)

# Save to new CSV
summary_df.to_csv('articles_with_smart_summaries.csv', index=False)
print(f"\n Saved {len(summary_df)} articles with smart summaries to 'articles_with_smart_summaries.csv'")

# Show some stats
print(f"\n Summary Statistics:")
print(f"   Average original length: {summary_df['original_content_length'].mean():.0f} chars")
print(f"   Average summary length: {summary_df['summary_length'].mean():.0f} chars")
print(f"   Compression ratio: {(summary_df['summary_length'] / summary_df['original_content_length']).mean():.1%}")