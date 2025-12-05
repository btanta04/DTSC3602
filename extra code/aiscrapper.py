import os
import csv
import sys
import json
from urllib.parse import urlparse
import trafilatura
from trafilatura import sitemaps
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_KEY)

# Picks what Gemini model because I was having problems with that
def pick_model(preferred="gemini-2.0-flash"):
    try:
        models = list(genai.list_models())
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                if preferred in m.name:
                    return m.name
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                return m.name
    except Exception as e:
        print("Model listing failed:", e)
    return "gemini-2.0-flash"  

MODEL = pick_model()
model = genai.GenerativeModel(model_name=MODEL)

print(f"Using Gemini model: {MODEL}")

START_URL = "https://www.outseer.com/"
BLOG_PATH_HINTS = ("/fraud-and-payment-blog", "/blog/")
OUT_CSV = "outseer_articles.csv"
MAX_URLS = 150


# Only getting articles
def looks_like_blog(u: str) -> bool:
    lower = u.lower()
    if any(p in lower for p in ("/tag/", "/category/", "/author/", "/page/")):
        return False
    return any(h in lower for h in BLOG_PATH_HINTS)


def summarize_text(title, text):
    if not text.strip():
        return ""
    prompt = f"""
    Summarize this Outseer blog post in 2–3 short sentences.
    Focus on the main topic and insights.
    Title: {title}
    Text:
    {text[:8000]}
    """
    try:
        r = model.generate_content(prompt)
        return r.text.strip()
    except Exception as e:
        print("  !! Gemini summary failed:", e)
        return ""


def main():
    domain = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(START_URL))

    try:
        urls = sitemaps.sitemap_search(domain) or []
    except Exception:
        urls = []
    urls = [u for u in urls if looks_like_blog(u)]
    urls = list(dict.fromkeys(urls))[:MAX_URLS]

    if not urls:
        print("No candidate URLs found.")
        sys.exit(0)

    rows = []
    for i, url in enumerate(urls, 1):
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            continue
        try:
            try:
                data = trafilatura.extract(downloaded, format="json", with_metadata=True)
            except TypeError:
                data = trafilatura.extract(downloaded, output_format="json", with_metadata=True)
        except Exception:
            continue

        if not data:
            continue

        j = json.loads(data)
        title = (j.get("title") or "").strip()
        published = (j.get("date") or "").strip()
        text = (j.get("text") or "").strip()

        if not (title or text):
            continue

        summary = summarize_text(title, text)
        rows.append({
            "title": title,
            "url": url,
            "published": published,
            "full_text": text,
            "summary": summary,
        })

        print(f"[{i}/{len(urls)}] ✓ {title[:80]} — summary done")

    if not rows:
        print("Nothing extracted.")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "url", "published", "full_text", "summary"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n Saved {len(rows)} articles → {OUT_CSV}")


if __name__ == "__main__":
    main()