import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_fraud_articles():
    url = "https://www.outseer.com/fraud-and-payment-blog"
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        
        # Look for all text elements that could be articles
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'div', 'a']):
            text = element.get_text(strip=True)
            
            # Filter for article-like text (reasonable length, not navigation)
            if text and len(text) > 20 and len(text) < 200:
                # Get URL if available
                if element.name == 'a' and element.get('href'):
                    link = element['href']
                    if not link.startswith('http'):
                        link = f"https://www.outseer.com{link}"
                else:
                    link = "https://www.outseer.com/fraud-and-payment-blog"
                
                # Check for fraud keywords
                fraud_keywords = ['fraud', 'scam', 'phishing', 'identity theft', 'social engineering', 'malware', 'payment threats']
                is_fraud = any(keyword in text.lower() for keyword in fraud_keywords)
                
                articles.append({
                    'title': text,
                    'url': link,
                    'is_fraud_related': is_fraud
                })
        
        # Remove duplicates and limit
        unique_articles = []
        seen_titles = set()
        for article in articles:
            if article['title'] not in seen_titles:
                unique_articles.append(article)
                seen_titles.add(article['title'])
        
        return unique_articles[:20]  # Return first 20 articles
        
    except Exception as e:
        print(f"Error: {e}")
        return []

# Run the scraper
articles = scrape_fraud_articles()

if articles:
    df = pd.DataFrame(articles)
    df.to_csv('fraud_articles.csv', index=False)
    print(f"Scraped {len(df)} articles")
    print("Saved to fraud_articles.csv")
    
    # Show what we found
    print("\n Articles found:")
    for i, article in enumerate(articles):
        fraud_flag = "🚨" if article['is_fraud_related'] else "  "
        print(f"{fraud_flag} {i+1}. {article['title']}")
        
else:
    print("No articles found")