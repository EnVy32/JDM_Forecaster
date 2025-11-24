import requests
from bs4 import BeautifulSoup
import re
import time

def get_usd_jpy_rate():
    """
    Fetches the live USD -> JPY exchange rate from a public API.
    Fallback to 150.0 if the API fails.
    """
    print("--- [FINANCE] Fetching Live Exchange Rate ---")
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            rate = data['rates']['JPY']
            print(f"--> Live Rate: 1 USD = {rate} JPY")
            return rate
    except Exception as e:
        print(f"--> Warning: API failed ({e}). Using fallback rate.")
    
    return 150.0 # Fallback

def fetch_page(url):
    """
    Advanced fetcher using a Session to mimic a real browser session.
    """
    print(f"--- [SCRAPER] Connecting to: {url} ---")
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.text
        else:
            print(f"--> Error: Server returned status {response.status_code}")
            return None
    except Exception as e:
        print(f"--> Critical Network Error: {e}")
        return None

def scrape_listings(base_url, max_pages=20, progress_callback=None):
    """
    Iterates through multiple pages to build a large dataset.
    Accepts a progress_callback(current_page, total_pages) to update UI.
    """
    all_cars = []
    current_rate = get_usd_jpy_rate()
    
    for page in range(1, max_pages + 1):
        # Update UI if callback is provided
        if progress_callback:
            progress_callback(page, max_pages)
            
        target_url = f"{base_url}?pn={page}"
        print(f"\n--- SCRAPING PAGE {page}/{max_pages} ---")
        
        html = fetch_page(target_url)
        if not html:
            print("--> Page fetch failed. Stopping.")
            break
            
        cars = parse_search_results(html, current_rate)
        all_cars.extend(cars)
        
        # Polite delay
        time.sleep(1.5)
        
    print(f"--- SCRAPING COMPLETE. Total Cars: {len(all_cars)} ---")
    return all_cars

def extract_price(container, text_content, exchange_rate):
    """
    Surgical logic to find the price and convert to JPY.
    """
    price_usd = 0
    price_tag = container.find(['p', 'span', 'div'], class_=re.compile(r'(price|fob)', re.IGNORECASE))
    if price_tag:
        raw_price = price_tag.get_text(strip=True)
        digits = re.sub(r'[^\d]', '', raw_price)
        if digits:
            price_usd = int(digits)

    if price_usd == 0:
        match = re.search(r'US\$\s*([\d,]+)', text_content)
        if match:
            price_clean = match.group(1).replace(',', '')
            price_usd = int(price_clean)

    if price_usd > 0:
        price_jpy_k = int((price_usd * exchange_rate) / 1000)
        return price_jpy_k
    return 0

def parse_search_results(html_text, exchange_rate=150.0):
    """
    Parses HTML to extract car data.
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    cars_data = []
    
    containers = soup.find_all(['li', 'div'], class_=re.compile(r'car-item'))
    if not containers:
        containers = soup.find_all('div', class_=re.compile(r'(product|item|listing)'))

    for container in containers:
        try:
            full_text = container.get_text(separator=' ', strip=True)
            
            link_tag = container.find('a', href=True)
            car_link = None
            if link_tag:
                href = link_tag['href']
                if href.startswith('http'):
                    car_link = href
                else:
                    car_link = f"https://www.tc-v.com{href}"

            price = extract_price(container, full_text, exchange_rate)
            
            year = None
            year_match = re.search(r'\b(199\d|200\d|201\d|202\d)\b', full_text)
            if year_match:
                year = int(year_match.group(0))

            mileage = 0
            mile_match = re.search(r'([\d,]+)\s*km', full_text, re.IGNORECASE)
            if mile_match:
                mileage = int(mile_match.group(1).replace(',', ''))
            
            engine = 1300
            eng_match = re.search(r'([\d,]+)\s*cc', full_text, re.IGNORECASE)
            if eng_match:
                engine = int(eng_match.group(1).replace(',', ''))

            is_mt = bool(re.search(r'\b(MT|Manual|F5|F6)\b', full_text, re.IGNORECASE))
            is_4wd = bool(re.search(r'\b(4WD|4x4|AWD)\b', full_text, re.IGNORECASE))
            transmission = 'mt' if is_mt else 'at'
            drive = '4wd' if is_4wd else '2wd'

            if price > 0 and year:
                cars_data.append({
                    'price': price,
                    'year': year,
                    'mileage': mileage,
                    'engine_capacity': engine,
                    'transmission': transmission,
                    'drive': drive,
                    'mark': 'honda',
                    'model': 'fit',
                    'link': car_link
                })
        except Exception:
            continue

    return cars_data