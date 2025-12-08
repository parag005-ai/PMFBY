"""Try to access DES official API for crop data."""
import requests

# Try DES official API endpoints
urls = [
    'https://data.desagri.gov.in/api/crops/apy/report',
    'https://data.desagri.gov.in/api/v1/crops',
    'https://data.desagri.gov.in/website/api/crops-apy-report',
]

for url in urls:
    try:
        r = requests.get(url, timeout=15)
        print(f"URL: {url}")
        print(f"  Status: {r.status_code}")
        ct = r.headers.get("content-type", "N/A")
        print(f"  Content-Type: {ct}")
        print(f"  Size: {len(r.content)} bytes")
        if r.status_code == 200 and len(r.content) < 1000:
            print(f"  Preview: {r.text[:300]}")
        print()
    except Exception as e:
        print(f"URL: {url}")
        print(f"  ERROR: {str(e)[:100]}")
        print()
