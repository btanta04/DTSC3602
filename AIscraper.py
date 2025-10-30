"""
Outseer Fraud & Payment Blog → CSV scraper

- Discovers article URLs from listing pages (regex, no BeautifulSoup).
- Follows pagination.
- Extracts clean article text:
    1) First try Gemini (JSON: title + paragraphs array)  ← reliable escaping
    2) Fallback to readability-lxml (deterministic, no LLM)
- Writes CSV: url,title,full_text

Requires:
  pip install google-generativeai requests python-dotenv readability-lxml lxml
"""

import os
import re
import csv
import json
import time
import random
import requests
from urllib.parse import urljoin
from typing import Dict, Any, Optional, List, Tuple

# ----------------- ENV -----------------
from dotenv import load_dotenv
load_dotenv()

START_URL   = os.getenv("START_URL", "https://www.outseer.com/fraud-and-payment-blog")
OUT_CSV     = os.getenv("OUT_CSV", "outseer_fraud_payment_blog.csv")
MAX_PAGES   = int(float(os.getenv("MAX_PAGES", "10")))
SLEEP_SEC   = float(os.getenv("SLEEP_SEC", "0.6"))
GEMINI_KEY  = os.getenv("GEMINI_API_KEY", "")
MODEL_PREF  = os.getenv("MODEL", "gemini-1.5-flash-latest")

UA_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OutseerScraper/1.0)"}

# ----------------- OPTIONAL LLM (Gemini) -----------------
GEMINI_ENABLED = bool(GEMINI_KEY)

if GEMINI_ENABLED:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEY)

    def pick_model(preferred: str) -> str:
        """
        Pick a model that supports generateContent.
        Handles older SDKs requiring 'models/...' names.
        """
        models = list(genai.list_models())
        name_map = {m.name: m for m in models}

        candidates = [
            preferred,
            f"models/{preferred}" if not preferred.startswith("models/") else preferred,
            "gemini-1.5-flash-latest",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-001",
            "models/gemini-1.5-pro",
            "models/gemini-1.0-pro",
        ]
        for c in candidates:
            m = name_map.get(c)
            if m and "generateContent" in getattr(m, "supported_generation_methods", []):
                return c

        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                return m.name
        raise RuntimeError("No Gemini model supports generateContent. Try upgrading google-generativeai.")

    MODEL_NAME = pick_model(MODEL_PREF)
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    print(f"[Gemini] Using model: {MODEL_NAME}")
else:
    print("[Gemini] Disabled (no GEMINI_API_KEY). Fallback extractor will be used for all posts.")

# ----------------- FALLBACK EXTRACTOR -----------------
from readability import Document
from lxml import html as lh

def extract_with_readability(html_text: str) -> Tuple[str, List[str]]:
    """
    Deterministic extraction of title + paragraphs.
    Returns: (title, [paragraphs])
    """
    doc = Document(html_text)
    title = doc.short_title() or ""
    article_html = doc.summary(html_partial=True)
    root = lh.fromstring(article_html)
    # Grab paragraphs and subheads
    texts = root.xpath("//p//text()|//h2//text()|//h3//text()")
    paragraphs = [t.strip() for t in texts if t and t.strip()]
    return title.strip(), paragraphs

# ----------------- HTTP + UTILS -----------------
def http_get(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout, headers=UA_HEADERS)
    r.raise_for_status()
    return r.text

def backoff_sleep(i: int, base: float = 1.5, jitter: float = 0.25):
    """Exponential backoff with jitter (for rate limits)."""
    t = base ** min(i, 6)
    t += random.uniform(0, jitter)
    time.sleep(t)

def clean_ws(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

# ----------------- LISTING PARSE (no LLM) -----------------
def find_post_links(listing_html: str, base_url: str) -> List[str]:
    """
    Heuristic regex to capture post permalinks on Outseer blog pages.
    Excludes categories/tags/authors/pagination links.
    """
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', listing_html, flags=re.I)
    links: List[str] = []
    for h in hrefs:
        if h.startswith("#") or h.startswith("mailto:") or h.startswith("tel:"):
            continue
        absu = urljoin(base_url, h)
        # Keep post-like paths; skip listing helpers
        if re.search(r"/(fraud-and-payment-blog|blog)/", absu, re.I) and not re.search(r"/(tag|category|author|page)/", absu, re.I):
            links.append(absu)
    # de-dupe preserve order
    seen: set = set()
    out: List[str] = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def find_next_page(listing_html: str, base_url: str) -> Optional[str]:
    # rel=next
    m = re.search(r'rel=["\']next["\'][^>]*href=["\']([^"\']+)["\']', listing_html, flags=re.I)
    if m:
        return urljoin(base_url, m.group(1))
    # link text
    m = re.search(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(?:Next|Older|Older Posts|Next Page)</a>', listing_html, flags=re.I)
    if m:
        return urljoin(base_url, m.group(1))
    # naive /page/N
    m = re.search(r'href=["\']([^"\']*page/[0-9]+/?)["\']', listing_html, flags=re.I)
    if m:
        return urljoin(base_url, m.group(1))
    return None

# ----------------- GEMINI ARTICLE EXTRACT -----------------
ARTICLE_SYS = (
    "Extract the main article from this HTML. Return JSON with keys:\n"
    '  "title": string (concise article title without site suffix)\n'
    '  "paragraphs": array of strings (each a cleaned paragraph of the main body)\n'
    "Rules:\n"
    "- Remove navigation, footers, sidebars, cookie banners, share/comments, unrelated captions.\n"
    "- Keep paragraph order; decode entities; no markdown or HTML in the text.\n"
    "- No extra keys. Return strictly valid minified JSON."
)

def extract_with_gemini(article_html: str) -> Dict[str, Any]:
    """
    Returns {"title": str, "paragraphs": [str, ...]}
    Raises on persistent failure (caller can fallback).
    """
    if not GEMINI_ENABLED:
        raise RuntimeError("Gemini disabled")

    # Keep input small to avoid truncation
    capped_html = article_html[:120_000]
    prompt = ARTICLE_SYS + "\n\nHTML:\n" + capped_html

    last_err: Optional[Exception] = None
    for i in range(3):
        try:
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                )
            )
            text = (resp.text or "").strip()

            # Guard: keep first {...} block if stray tokens appear
            l = text.find("{")
            r = text.rfind("}")
            if l != -1 and r != -1:
                text = text[l:r+1]

            data = json.loads(text)
            title = (data.get("title") or "").strip()
            paras = [p.strip() for p in (data.get("paragraphs") or []) if p and p.strip()]
            return {"title": title, "paragraphs": paras}
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "rate" in msg or "finish_reason" in msg:
                backoff_sleep(i)
                continue
            break
    raise RuntimeError(f"Gemini parse failed: {last_err}")

# ----------------- MAIN -----------------
def main():
    page_url = START_URL
    pages = 0
    seen: set = set()
    rows: List[Dict[str, str]] = []

    while page_url and pages < MAX_PAGES:
        print("[Listing] GET", page_url)
        try:
            listing_html = http_get(page_url)
        except Exception as e:
            print("  !! Listing fetch failed:", e)
            break

        links = find_post_links(listing_html, base_url=page_url)
        print(f"  Found {len(links)} article URLs")

        for url in links:
            if url in seen:
                continue
            seen.add(url)

            try:
                time.sleep(SLEEP_SEC)
                print("[Article] GET", url)
                art_html = http_get(url)

                # Try Gemini first (if enabled), else go straight to readability
                title = ""
                full_text = ""
                if GEMINI_ENABLED:
                    try:
                        parsed = extract_with_gemini(art_html)
                        title = clean_ws(parsed.get("title", ""))
                        paragraphs = parsed.get("paragraphs") or []
                        full_text = clean_ws("\n\n".join(paragraphs))
                    except Exception as ge:
                        print("  !! Gemini parse failed; falling back:", ge)

                if not (title or full_text):
                    # Fallback (or primary if Gemini disabled)
                    rb_title, rb_paras = extract_with_readability(art_html)
                    title = clean_ws(rb_title)
                    full_text = clean_ws("\n\n".join(rb_paras))

                if not (title or full_text):
                    print("  !! empty parse; skipped")
                    continue

                rows.append({"url": url, "title": title, "full_text": full_text})
                print("  ✓", title[:100])

            except Exception as e:
                print("  !! Article error:", e)

        pages += 1
        next_page = find_next_page(listing_html, base_url=page_url)
        page_url = next_page
        if page_url:
            time.sleep(SLEEP_SEC)

    if rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["url", "title", "full_text"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {len(rows)} articles → {OUT_CSV}")
    else:
        print("\nNo articles extracted.")

if __name__ == "__main__":
    main()