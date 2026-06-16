import html
import hashlib
import logging
import re
import unicodedata
from datetime import datetime
import time
from typing import List, Dict, Optional
import feedparser
import requests
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# User-Agent header to avoid blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

RSS_FEEDS = {
    "Tagesschau": "https://www.tagesschau.de/xml/rss2/",
    "Deutsche Welle": "https://rss.dw.com/xml/rss-de-all",
    "Spiegel": "https://www.spiegel.de/index.rss"
}

def clean_german_text(text: str) -> str:
    """
    Cleans HTML tags, unescapes HTML entities, normalizes German Umlauts
    to Unicode NFC form, and formats the German ß cleanly.
    """
    if not text:
        return ""
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Strip HTML tags just in case
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize to Unicode NFC form (ensures characters like ä, ö, ü, ß are single codepoints)
    text = unicodedata.normalize('NFC', text)
    
    # Clean hyphens in German compound words (e.g. "Corona - Krise" -> "Corona-Krise")
    text = re.sub(r'(\b\w+)\s*-\s*(\w+\b)', r'\1-\2', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def generate_article_id(url: str, pub_date_str: str) -> str:
    """
    Generates a unique SHA-256 hash from URL and publication date.
    """
    unique_string = f"{url}||{pub_date_str}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

def scrape_full_article_body(url: str, source: str) -> str:
    """
    Fetches the article URL and extracts the main body text based on publisher rules,
    falling back to generic paragraph parsing if needed.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {url} (Status {response.status_code})")
            return ""
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts, styles, forms, headers, footers, and navigation
        for element in soup(["script", "style", "form", "footer", "header", "nav", "aside"]):
            element.decompose()
            
        paragraphs = []
        
        # Site-specific selectors
        if source == "Tagesschau":
            # Tagesschau uses paragraphs with class textabschnitt or subheadings
            text_blocks = soup.find_all(["p", "h2"], class_=lambda x: x and "textabschnitt" in x)
            if text_blocks:
                paragraphs = [block.get_text() for block in text_blocks]
            else:
                # Fallback to article tag
                article = soup.find("article")
                if article:
                    paragraphs = [p.get_text() for p in article.find_all("p")]
                    
        elif source == "Spiegel":
            # Spiegel content is often in sections/divs within article body
            article = soup.find("article")
            if article:
                # Find paragraphs inside the article container, excluding related stories
                paragraphs = [p.get_text() for p in article.find_all("p") 
                              if not p.find_parent(class_=lambda x: x and ("infobox" in x or "related" in x or "commercial" in x))]
            else:
                # Fallback to specific article paragraphs class
                text_blocks = soup.find_all("p", class_=lambda x: x and ("article-paragraph" in x or "paragraph" in x))
                paragraphs = [block.get_text() for block in text_blocks]
                
        elif source == "Deutsche Welle":
            # DW uses longText class or general article paragraphs
            main_content = soup.find(class_=lambda x: x and ("longText" in x or "article-text" in x or "main-content" in x))
            if main_content:
                paragraphs = [p.get_text() for p in main_content.find_all("p")]
            else:
                article = soup.find("article")
                if article:
                    paragraphs = [p.get_text() for p in article.find_all("p")]
        
        # Generic fallback if specific scraping yielded nothing
        if not paragraphs:
            article = soup.find("article")
            if article:
                paragraphs = [p.get_text() for p in article.find_all("p")]
            else:
                # Last resort: all paragraphs longer than 50 chars that are not links
                for p in soup.find_all("p"):
                    p_text = p.get_text().strip()
                    if len(p_text) > 50 and not p.find("a", recursive=False):
                        paragraphs.append(p_text)
                        
        # Clean and join paragraphs
        cleaned_paragraphs = []
        for p in paragraphs:
            cleaned = clean_german_text(p)
            # Filter out boilerplate, short sentences, or social media links
            if len(cleaned) > 30 and not any(bp in cleaned.lower() for bp in ["folgen sie uns", "datenschutz", "cookie", "impressum", "all rights reserved"]):
                cleaned_paragraphs.append(cleaned)
                
        return "\n\n".join(cleaned_paragraphs)
        
    except Exception as e:
        logger.error(f"Error scraping article body from {url}: {e}")
        return ""

def scrape_news_feeds(limit_per_feed: int = 5) -> List[Dict]:
    """
    Scrapes the RSS feeds and retrieves article details, fetching the full body
    for new/valid items.
    """
    scraped_articles = []
    
    for source, feed_url in RSS_FEEDS.items():
        logger.info(f"Parsing RSS feed for {source}: {feed_url}")
        try:
            # Fetch RSS feed using requests with browser headers to avoid user-agent blocks
            response = requests.get(feed_url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
            else:
                logger.warning(f"Failed to fetch RSS feed for {source} (Status {response.status_code}). Falling back to direct parsing.")
                feed = feedparser.parse(feed_url)
                
            entries = feed.entries[:limit_per_feed]
            logger.info(f"Found {len(feed.entries)} entries. Processing top {len(entries)}")
            
            for entry in entries:
                title = clean_german_text(entry.get("title", ""))
                url = entry.get("link", "")
                
                if not url:
                    continue
                    
                # Normalize URL (remove query params)
                url = url.split("?")[0]
                
                # Parse date
                pub_date_struct = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub_date_struct:
                    pub_date = datetime.fromtimestamp(time.mktime(pub_date_struct))
                else:
                    pub_date = datetime.utcnow()
                    
                pub_date_str = pub_date.isoformat()
                article_id = generate_article_id(url, pub_date_str)
                
                logger.info(f"Scraping full text for article: {title} ({url})")
                body_de = scrape_full_article_body(url, source)
                
                # Fallback to RSS summary if full body could not be fetched
                if not body_de:
                    body_de = clean_german_text(entry.get("summary", ""))
                    logger.info("Fell back to RSS summary for body text.")
                    
                if not body_de:
                    logger.warning(f"Skipping article {url} due to empty body text.")
                    continue
                
                scraped_articles.append({
                    "article_id": article_id,
                    "timestamp": pub_date_str,
                    "source": source,
                    "url": url,
                    "title_de": title,
                    "body_de": body_de
                })
                
                # Polite scraping interval
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error parsing feed {source}: {e}")
            
    return scraped_articles

if __name__ == "__main__":
    # Quick test of scraper
    test_articles = scrape_news_feeds(limit_per_feed=1)
    print(f"Scraped {len(test_articles)} articles.")
    if test_articles:
        print(f"Sample Article Source: {test_articles[0]['source']}")
        print(f"Sample Article Title: {test_articles[0]['title_de']}")
        print(f"Sample Article Body (truncated): {test_articles[0]['body_de'][:300]}...")
