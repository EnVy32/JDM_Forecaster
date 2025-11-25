import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re
import sys
import time
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

# --- [WINDOWS FIX] ---
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- [CORE ASYNC ENGINE] ---

async def get_usd_jpy_rate_async(session, logs):
    """
    Fetches live exchange rate.
    """
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        async with session.get(url, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                rate = data['rates']['JPY']
                logs.append(f"✅ [Finance] Rate fetched: 1 USD = {rate} JPY")
                return rate
    except Exception as e:
        logs.append(f"⚠️ [Finance] API Error: {e}. Using fallback 150.0")
    
    return 150.0

async def fetch_page_async(session, url, semaphore, logs):
    """
    Fetches a single page. Handles 404s gracefully (End of Pagination).
    """
    async with semaphore:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Referer': 'https://www.google.com/'
        }
        try:
            await asyncio.sleep(0.5) # Polite delay
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 404:
                    # 404 is expected when we reach the end of listings
                    return None
                else:
                    logs.append(f"❌ [Network] Error {response.status} at {url}")
                    return None
        except Exception as e:
            # logs.append(f"❌ [Network] Exception: {e}")
            return None

def extract_price(container, text_content, exchange_rate):
    price_usd = 0
    
    price_tag = container.find(['p', 'span', 'div'], class_=re.compile(r'(price|fob)', re.IGNORECASE))
    if price_tag:
        digits = re.sub(r'[^\d]', '', price_tag.get_text(strip=True))
        if digits: price_usd = int(digits)

    if price_usd == 0:
        match = re.search(r'US\$\s*([\d,]+)', text_content)
        if match:
            price_usd = int(match.group(1).replace(',', ''))

    if price_usd > 0:
        return int((price_usd * exchange_rate) / 1000)
    return 0

def parse_search_results(html_text, exchange_rate, target_mark, target_model, logs):
    if not html_text: return []
    
    soup = BeautifulSoup(html_text, 'html.parser')
    cars_data = []
    
    containers = soup.find_all(['li', 'div'], class_=re.compile(r'car-item'))
    if not containers:
        containers = soup.find_all('div', class_=re.compile(r'(product|item|listing)'))

    for container in containers:
        try:
            full_text = container.get_text(separator=' ', strip=True)
            link_tag = container.find('a', href=True)
            car_link = link_tag['href'] if link_tag else None
            if car_link and not car_link.startswith('http'): car_link = f"https://www.tc-v.com{car_link}"

            price = extract_price(container, full_text, exchange_rate)
            
            # Expanded Year Regex (1980s+)
            year_match = re.search(r'\b(198\d|199\d|200\d|201\d|202\d)\b', full_text)
            year = int(year_match.group(0)) if year_match else None

            mile_match = re.search(r'([\d,]+)\s*km', full_text, re.IGNORECASE)
            mileage = int(mile_match.group(1).replace(',', '')) if mile_match else 0
            
            eng_match = re.search(r'([\d,]+)\s*cc', full_text, re.IGNORECASE)
            engine = int(eng_match.group(1).replace(',', '')) if eng_match else 0

            is_mt = bool(re.search(r'\b(MT|Manual|F5|F6|5MT|6MT)\b', full_text, re.IGNORECASE))
            is_4wd = bool(re.search(r'\b(4WD|AWD)\b', full_text, re.IGNORECASE))
            
            grade_tag = container.find('p', class_=re.compile(r'grade'))
            grade = grade_tag.get_text(strip=True) if grade_tag else "Unknown"

            if price > 0 and year:
                cars_data.append({
                    'price': price, 'year': year, 'mileage': mileage, 'engine_capacity': engine,
                    'transmission': 'mt' if is_mt else 'at', 'drive': '4wd' if is_4wd else '2wd',
                    'grade': grade, 'mark': target_mark, 'model': target_model, 'link': car_link
                })
        except Exception:
            continue

    return cars_data

# --- https://learn.microsoft.com/en-us/azure/logic-apps/error-exception-handling ---

def get_clean_base_url(url):
    """
    Removes the 'pn' parameter from the URL to get the true Page 1 URL.
    Preserves other filters (e.g., ?steering=rhd).
    """
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    
    # Remove 'pn' if it exists
    if 'pn' in query_params:
        del query_params['pn']
    
    new_query = urlencode(query_params, doseq=True)
    
    new_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))
    return new_url

def build_pagination_url(base_url, page_number):
    """
    Safely adds 'pn=X' to the URL.
    """
    parsed = urlparse(base_url)
    query_params = parse_qs(parsed.query)
    
    query_params['pn'] = [str(page_number)]
    
    new_query = urlencode(query_params, doseq=True)
    
    new_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))
    return new_url

async def scrape_listings_async_runner(base_url, max_pages, progress_callback, target_mark, target_model):
    logs = []
    all_cars = []
    sem = asyncio.Semaphore(5)
    
    logs.append(f"🚀 [System] Target URL: {base_url}")
    
    async with aiohttp.ClientSession() as session:
        rate = await get_usd_jpy_rate_async(session, logs)
        
        # --- PHASE 1: PAGE 1 (THE ROOT) ---
        # We use the cleaned base URL directly. We DO NOT append pn=1.
        page1_url = get_clean_base_url(base_url)
        logs.append(f"🔎 [System] Fetching Page 1: {page1_url}")
        
        probe_html = await fetch_page_async(session, page1_url, sem, logs)
        
        if not probe_html:
            logs.append("❌ [Critical] Page 1 failed. Check URL or Network.")
            return [], logs
            
        cars_p1 = parse_search_results(probe_html, rate, target_mark, target_model, logs)
        all_cars.extend(cars_p1)
        if progress_callback: progress_callback(1, max_pages)
        
        if not cars_p1:
             logs.append("⚠️ [Warning] Page 1 loaded but NO cars found. Check selectors.")

        # --- PHASE 2: MASS SCRAPE (Pages 2 to N) ---
        # Only proceed if we actually found cars on Page 1
        if max_pages > 1 and len(cars_p1) > 0:
            completed_tasks = 1
            
            async def monitored_fetch(url):
                nonlocal completed_tasks
                html = await fetch_page_async(session, url, sem, logs)
                completed_tasks += 1
                if progress_callback: progress_callback(completed_tasks, max_pages)
                return html

            tasks = []
            for i in range(2, max_pages + 1):
                url = build_pagination_url(page1_url, i)
                tasks.append(monitored_fetch(url))
            
            html_pages = await asyncio.gather(*tasks)
            
            valid_pages_count = 1 
            for html in html_pages:
                if html:
                    valid_pages_count += 1
                    cars = parse_search_results(html, rate, target_mark, target_model, logs)
                    all_cars.extend(cars)
            
            logs.append(f"🏁 [System] Scrape Finished. Pages: {valid_pages_count}. Total Cars: {len(all_cars)}")

    return all_cars, logs

# --- [SYNCHRONOUS API] ---

def scrape_listings(base_url, max_pages=100, progress_callback=None):
    try:
        parts = base_url.strip('/').split('/')
        if 'used_car' in parts:
            idx = parts.index('used_car')
            t_mark = parts[idx+1] if len(parts) > idx+1 else "unknown"
            t_model = parts[idx+2] if len(parts) > idx+2 else "unknown"
        else:
            t_mark, t_model = "unknown", "unknown"
    except:
        t_mark, t_model = "unknown", "unknown"

    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    return asyncio.run(scrape_listings_async_runner(base_url, max_pages, progress_callback, t_mark, t_model))

def fetch_page(url):
    """Sync wrapper for debugging"""
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    async def _run():
        async with aiohttp.ClientSession() as session:
            return await fetch_page_async(session, url, asyncio.Semaphore(1), [])
    return asyncio.run(_run())