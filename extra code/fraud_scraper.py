import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_articles_with_content():
    url = "https://www.outseer.com/fraud-and-payment-blog"
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        
        # First, find article links
        for element in soup.find_all('a', href=True):
            text = element.get_text(strip=True)
            link = element['href']
            
            if text and len(text) > 20 and len(text) < 200 and '/blog/' in link:
                if not link.startswith('http'):
                    link = f"https://www.outseer.com{link}"
                
                articles.append({
                    'title': text,
                    'url': link
                })
        
        # Remove duplicates
        unique_articles = []
        seen_urls = set()
        for article in articles:
            if article['url'] not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(article['url'])
        
        print(f"Found {len(unique_articles)} articles, now getting content...")
        
        # Now get the full text from each article page
        for article in unique_articles[:15]:  # Get first 15 articles
            print(f"Getting content from: {article['title']}")
            article_content = get_article_text(article['url'])
            article['content'] = article_content
            time.sleep(1)  # Be nice to the server
        
        return unique_articles[:15]
        
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_article_text(article_url):
    """Get the full text from an article page"""
    try:
        response = requests.get(article_url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get all text from the page
        full_text = soup.get_text()
        # Clean it up
        cleaned_text = ' '.join(full_text.split())
        return cleaned_text
        
    except Exception as e:
        return f"Could not extract content: {e}"

# Run the scraper
articles = scrape_articles_with_content()

if articles:
    df = pd.DataFrame(articles)
    df.to_csv('articles_with_content.csv', index=False)
    print(f"✅ Scraped {len(df)} articles with full content!")
    print("📁 Saved to articles_with_content.csv")
    
    # Show sample
    print(f"\nSample content length: {len(articles[0]['content'])} characters")
    print(f"Preview: {articles[0]['content'][:200]}...")
else:
    print("No articles found")