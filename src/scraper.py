import requests
from bs4 import BeautifulSoup
import re

def fetch_page(url):
    """
    Advanced fetcher using a Session to mimic a real browser session.
    Maintains cookies and headers to avoid 403 Forbidden errors.
    """
    print(f"--- [SCRAPER] Connecting to: {url} ---")
    
    # Use a Session object to persist cookies (looks more human)
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
            print(f"--> Success: Downloaded {len(response.text)} bytes.")
            return response.text
        else:
            print(f"--> Error: Server returned status {response.status_code}")
            return None
    except Exception as e:
        print(f"--> Critical Network Error: {e}")
        return None

def extract_price(container, text_content):
    """
    Surgical logic to find the price.
    Strategy 1: Search for specific 'price' or 'fob' tags.
    Strategy 2: Regex search on the entire text for 'US$'.
    """
    price = 0
    
    # 1. Try finding a dedicated price tag
    price_tag = container.find(['p', 'span', 'div'], class_=re.compile(r'(price|fob)', re.IGNORECASE))
    if price_tag:
        raw_price = price_tag.get_text(strip=True)
        # Extract digits
        digits = re.sub(r'[^\d]', '', raw_price)
        if digits:
            price = int(digits)

    # 2. Fallback: Search the entire text for pattern "US$ 1,200"
    if price == 0:
        match = re.search(r'US\$\s*([\d,]+)', text_content)
        if match:
            price_clean = match.group(1).replace(',', '')
            price = int(price_clean)

    # Currency Conversion (USD -> '000 JPY)
    # Assumption: 1 USD = 150 JPY. We divide by 1000 for the model scale.
    if price > 0:
        price_jpy_k = int((price * 150) / 1000)
        return price_jpy_k
    
    return 0

def parse_search_results(html_text):
    """
    Super-Advanced Parser.
    Deconstructs HTML elements to find data even if structure changes.
    """
    print("--- [PARSER] Analyzing HTML Structure ---")
    soup = BeautifulSoup(html_text, 'html.parser')
    
    cars_data = []
    
    # Find car containers. 'li.car-item' is standard, but we allow fallbacks.
    containers = soup.find_all(['li', 'div'], class_=re.compile(r'car-item'))
    
    if not containers:
        print("--> WARNING: No 'car-item' found. Trying fallback to generic cards...")
        containers = soup.find_all('div', class_=re.compile(r'(product|item|listing)'))

    print(f"--> Found {len(containers)} potential listings.")
    
    success_count = 0
    
    for container in containers:
        try:
            # Get full text for regex mining
            full_text = container.get_text(separator=' ', strip=True)
            
            # --- 1. LINK (Source of Truth) ---
            link_tag = container.find('a', href=True)
            car_link = None
            if link_tag:
                href = link_tag['href']
                if href.startswith('http'):
                    car_link = href
                else:
                    car_link = f"https://www.tc-v.com{href}"

            # --- 2. PRICE (Helper Function) ---
            price = extract_price(container, full_text)
            
            # --- 3. YEAR ---
            year = None
            # Regex boundaries \b prevent matching '2000cc' as a year
            year_match = re.search(r'\b(199\d|200\d|201\d|202\d)\b', full_text)
            if year_match:
                year = int(year_match.group(0))

            # --- 4. MILEAGE ---
            mileage = 0
            mile_match = re.search(r'([\d,]+)\s*km', full_text, re.IGNORECASE)
            if mile_match:
                mileage = int(mile_match.group(1).replace(',', ''))
            
            # --- 5. ENGINE ---
            engine = 1300 # Default for Fit
            eng_match = re.search(r'([\d,]+)\s*cc', full_text, re.IGNORECASE)
            if eng_match:
                engine = int(eng_match.group(1).replace(',', ''))

            # --- 6. SPECS ---
            is_mt = bool(re.search(r'\b(MT|Manual|F5|F6)\b', full_text, re.IGNORECASE))
            is_4wd = bool(re.search(r'\b(4WD|4x4|AWD)\b', full_text, re.IGNORECASE))
            
            transmission = 'mt' if is_mt else 'at'
            drive = '4wd' if is_4wd else '2wd'

            # --- VALIDATION ---
            # Only add if we have Price and Year (Critical for AI)
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
                success_count += 1
                
        except Exception as e:
            continue

    print(f"--> Successfully extracted data for {success_count} cars.")
    return cars_data