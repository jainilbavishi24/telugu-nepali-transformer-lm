import requests
import trafilatura
import unicodedata
import json
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

OUTPUT_FILE = 'Nepalese/data.nosync/nepali_corpus.jsonl'

def clean_nepali_text(text):
    text = text.replace('\u200C', '').replace('\u200D', '')
    return unicodedata.normalize('NFC', text).strip()

def scrape_wordpress_api():
    base_api_url = "https://ratopati.com/feed"
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        # Loop through pages (e.g., pages 1 to 50, yielding ~10 items per page)
        for page in range(1, 51):
            print(f"Fetching API page {page}...")
            params = {'page': page, 'per_page': 10}
            
            try:
                response = requests.get(base_api_url, headers=HEADERS, params=params, timeout=10)
                if response.status_code == 400:
                    print("Reached the end of available pages.")
                    break
                response.raise_for_status()
                
                posts = response.json()
                if not posts:
                    break
                    
                for post in posts:
                    url = post.get('link')
                    if not url:
                        continue
                        
                    # Extract article text using trafilatura
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        raw_text = trafilatura.extract(downloaded, target_language='ne')
                        if raw_text:
                            clean_text = clean_nepali_text(raw_text)
                            record = {"url": url, "text": clean_text}
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            print(f"Saved: {url}")
                            
            except Exception as e:
                print(f"Error on page {page}: {e}")
                
            time.sleep(1)

if __name__ == "__main__":
    scrape_wordpress_api()