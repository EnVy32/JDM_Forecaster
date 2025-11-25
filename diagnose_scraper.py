from src.scraper import fetch_page, parse_search_results
import re

# The Problematic URL
target_url = "https://www.tc-v.com/used_car/mazda/rx-7/"

print(f"--- DIAGNOSING: {target_url} ---")

# 1. Fetch raw HTML
print("1. Fetching HTML...")
html = fetch_page(target_url)

if not html:
    print("❌ FATAL: Could not fetch page. Likely Network/Firewall issue.")
    exit()

print(f"✅ HTML Fetched ({len(html)} bytes).")

# 2. Save HTML for manual inspection
with open("debug_output.html", "w", encoding="utf-8") as f:
    f.write(html)
print("   -> Saved raw HTML to 'debug_output.html'. Open this in Chrome to check for CAPTCHA.")

# 3. Test Parsing
print("2. Testing Parser...")
cars = parse_search_results(html, target_mark="mazda", target_model="rx-7")
print(f"✅ Found {len(cars)} valid cars in this page.")

if len(cars) == 0:
    print("❌ WARNING: No cars passed the filter!")
    
    # Debug WHY they failed
    print("   -> Analyzing raw content for keywords...")
    if "car-item" in html:
        print("   -> 'car-item' class FOUND. The layout seems correct.")
    else:
        print("   -> 'car-item' class NOT FOUND. The website layout might have changed.")

    # Check for Price patterns
    prices = re.findall(r'US\$\s*([\d,]+)', html)
    print(f"   -> Found {len(prices)} price strings (e.g., {prices[:3] if prices else 'None'}).")
    
    if len(prices) > 0 and len(cars) == 0:
        print("   -> CONCLUSION: Prices exist, but the filtering logic (Year/Price>0) is dropping them.")
    elif len(prices) == 0:
        print("   -> CONCLUSION: No prices found. They might be 'ASK' or hidden via JavaScript.")

else:
    print("   -> Success! The scraper logic is working.")
    print(f"   -> Example: {cars[0]}")