import pandas as pd
import re
from collections import Counter

print("🚀 NLP Analysis Started!")

# Load your data
try:
    df = pd.read_csv('articles_with_content.csv')
    print(f"📊 Loaded {len(df)} articles with content")
except:
    df = pd.read_csv('fraud_articles.csv')  
    print(f"📊 Loaded {len(df)} articles")

# YOUR NLP MISSIONS:

def smart_fraud_detector(text):
    """Make this smarter - detect fraud better"""
    if not isinstance(text, str):
        return 0
        
    text = text.lower()
    
    # Current simple version - IMPROVE THIS!
    fraud_words = ['fraud', 'scam', 'phishing', 'hack', 'theft', 'identity', 'compromise']
    score = sum(1 for word in fraud_words if word in text)
    
    return score

def make_summary(text):
    """Make better summaries of articles"""
    if not isinstance(text, str):
        return "No content"
    
    # Simple: first 2 sentences - IMPROVE THIS!
    sentences = re.split(r'[.!?]+', text)
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    summary = '. '.join(clean_sentences[:2]) + '.'
    
    return summary

def find_trending_topics(articles):
    """Find most common fraud topics"""
    all_text = ' '.join([str(text) for text in articles])
    words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
    
    # Remove common words
    stop_words = {'this', 'that', 'with', 'from', 'have', 'were', 'they', 'their'}
    meaningful_words = [w for w in words if w not in stop_words]
    
    return Counter(meaningful_words).most_common(10)

# TEST YOUR NLP SKILLS
print("\n🎯 Testing NLP Functions:")
print("=" * 40)

# Test on first 3 articles
for i, row in df.head(3).iterrows():
    content = row.get('content', row.get('text', ''))
    score = smart_fraud_detector(content)
    summary = make_summary(content)
    
    print(f"\n📰 Article {i+1}:")
    print(f"   Title: {row['title'][:50]}...")
    print(f"   Fraud Score: {score}")
    print(f"   Summary: {summary[:100]}...")

# Find trending topics
if 'content' in df.columns:
    topics = find_trending_topics(df['content'].dropna())
    print(f"\n🔥 Top 5 Trending Topics:")
    for word, count in topics[:5]:
        print(f"   {word}: {count} times")

print("\n✅ Ready to improve these NLP functions!")