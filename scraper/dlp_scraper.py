"""
Dubai Legislation Portal (DLP) Scraper
=======================================
Downloads PDF legislations from https://dlp.dubai.gov.ae

Sources:
  1. Browse Legislation page — all legislations listed with PDF links
  2. Official Gazette page  — gazette PDFs by year

Usage:
    python scraper/dlp_scraper.py                    # scrape all
    python scraper/dlp_scraper.py --type law         # only laws
    python scraper/dlp_scraper.py --year 2024        # only 2024
    python scraper/dlp_scraper.py --max 10           # limit to 10
    python scraper/dlp_scraper.py --gazette          # gazette PDFs
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import unquote, urljoin

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
except ImportError:
    sync_playwright = None

# Fix Windows console encoding for unicode output
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import requests
from bs4 import BeautifulSoup

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

BASE_URL = "https://dlp.dubai.gov.ae"
BROWSE_URL = f"{BASE_URL}/en/Pages/BrowseLegislation.aspx"
GAZETTE_URL = f"{BASE_URL}/en/Pages/OfficialGazette.aspx"

DEFAULT_OUTPUT_DIR = "pdf_english"
GAZETTE_OUTPUT_DIR = "pdf_gazette"
MANIFEST_FILE = "scraper/download_manifest.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Rate limiting
REQUEST_DELAY = 1.5  # seconds between requests


# ------------------------------------------------------------------ #
# Legislation list parser
# ------------------------------------------------------------------ #

def fetch_page(url: str, session: requests.Session) -> BeautifulSoup:
    """Fetch a page and return its BeautifulSoup."""
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "html.parser")  # no lxml needed


def fetch_dynamic_page(url: str) -> BeautifulSoup:
    """Fetch a page using Playwright and repeatedly click 'Load More' until all items are loaded."""
    if sync_playwright is None:
        raise ImportError("playwright is not installed. Please run `pip install playwright` and `playwright install chromium`.")
    
    print("\n[*] Launching headless browser to fetch dynamic content...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Add a realistic user agent to avoid blocking
        context = browser.new_context(user_agent=HEADERS["User-Agent"])
        page = context.new_page()
        
        print(f"[*] Navigating to {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        
        # Give initial load extra time
        page.wait_for_timeout(3000)
        
        click_count = 0
        while True:
            try:
                # Based on the page source, the load more link is a span or A tag with "Load More"
                load_more_btn = page.locator("a:has-text('Load More'), span:has-text('Load More')").first
                
                if not load_more_btn.is_visible(timeout=2000):
                    break
                    
                load_more_btn.scroll_into_view_if_needed()
                load_more_btn.click()
                click_count += 1
                
                if click_count % 5 == 0:
                    print(f"    ... clicked 'Load More' {click_count} times")
                
                # Wait for the table to update
                page.wait_for_timeout(2500)
                
            except Exception:
                # If it times out checking visibility or fails to click, assume we're done
                break
                
        print(f"[*] Finished loading all pages. Clicked 'Load More' a total of {click_count} times.")
        
        html_content = page.content()
        browser.close()
        
    return BeautifulSoup(html_content, "html.parser")


def parse_legislation_entries(soup: BeautifulSoup) -> list:
    """
    Parse legislation entries from the Browse Legislation page.

    Each entry has:
      - title
      - type (Law, Resolution, Decree, etc.)
      - issuing_authority
      - date_of_issuance
      - status
      - pdf_url
      - html_url
      - arabic_pdf_url
      - gazette_number
    """
    entries = []

    # The page uses list items with "View as PDF" links
    # Find all PDF links first — they are the most reliable anchors
    pdf_links = soup.find_all("a", href=re.compile(r"Legislation%20Reference.*\.pdf", re.IGNORECASE))

    if not pdf_links:
        # Fallback: look for direct PDF links
        pdf_links = soup.find_all("a", href=re.compile(r"Legislation Reference.*\.pdf", re.IGNORECASE))

    for link in pdf_links:
        href = link.get("href", "")
        if not href:
            continue

        # Build absolute URL
        if href.startswith("/"):
            pdf_url = BASE_URL + href
        elif not href.startswith("http"):
            pdf_url = BASE_URL + "/" + href
        else:
            pdf_url = href

        # Extract filename from URL
        filename = unquote(pdf_url.split("/")[-1])

        # Try to find the parent container for metadata
        parent = link.find_parent("li") or link.find_parent("div")

        entry = {
            "title": _extract_title(filename),
            "pdf_url": pdf_url,
            "filename": filename,
            "type": "",
            "issuing_authority": "",
            "date_of_issuance": "",
            "status": "",
            "gazette_number": "",
            "source": "browse_legislation",
        }

        if parent:
            text = parent.get_text(" ", strip=True)

            # Extract metadata from surrounding text
            type_match = re.search(r"Type\s+(Law|Resolution|Decree|Order|Regulation)", text, re.IGNORECASE)
            if type_match:
                entry["type"] = type_match.group(1)

            auth_match = re.search(r"Issuing Authority\s+(.+?)(?:Source|Date|Type|$)", text)
            if auth_match:
                entry["issuing_authority"] = auth_match.group(1).strip()

            date_match = re.search(r"Date of Issuance\s+(\d{1,2}\s+\w+\s+\d{4})", text)
            if date_match:
                entry["date_of_issuance"] = date_match.group(1)

            status_match = re.search(r"Status\s+(In Force|Repealed|Amended|Partially Repealed)", text, re.IGNORECASE)
            if status_match:
                entry["status"] = status_match.group(1)

        # Extract year from filename or URL
        year_match = re.search(r"(20\d{2}|19\d{2})", filename)
        if year_match:
            entry["year"] = int(year_match.group(1))
        else:
            entry["year"] = None

        entries.append(entry)

    return entries


def _extract_title(filename: str) -> str:
    """Clean up a filename into a human-readable title."""
    name = filename.replace(".pdf", "").replace("%20", " ")
    name = unquote(name)
    # Remove trailing size info like "247 kb"
    name = re.sub(r"\d+\s*kb$", "", name, flags=re.IGNORECASE).strip()
    return name


# ------------------------------------------------------------------ #
# Gazette parser
# ------------------------------------------------------------------ #

def parse_gazette_links(session: requests.Session) -> list:
    """
    Parse gazette PDF links from the Official Gazette page.
    The page requires JavaScript for year selection, so we try
    direct URL patterns for gazette PDFs.
    """
    entries = []

    # The gazette uses a PDF viewer with ?file=<number>
    # Try fetching the browse page which also lists gazette numbers
    soup = fetch_page(BROWSE_URL, session)
    gazette_links = soup.find_all("a", href=re.compile(r"PDFViewer\.aspx\?file=\d+"))

    seen = set()
    for link in gazette_links:
        href = link.get("href", "")
        num_match = re.search(r"file=(\d+)", href)
        if not num_match:
            continue

        gazette_num = num_match.group(1)
        if gazette_num in seen:
            continue
        seen.add(gazette_num)

        # Gazette PDFs follow this URL pattern
        pdf_url = f"{BASE_URL}/en/Pages/PDFViewer.aspx?file={gazette_num}"

        entries.append({
            "title": f"Official Gazette No. {gazette_num}",
            "gazette_number": gazette_num,
            "pdf_url": pdf_url,
            "filename": f"Official_Gazette_{gazette_num}.pdf",
            "source": "official_gazette",
        })

    return entries


# ------------------------------------------------------------------ #
# Downloader
# ------------------------------------------------------------------ #

def extract_gazette_metadata(pdf_path: str) -> dict:
    """Read the first page of the gazette to extract metadata like the year."""
    import pdfplumber
    import re
    
    meta = {"year": None}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text()
                if text:
                    years = re.findall(r"\b((?:19|20)\d{2})\b", text)
                    if years:
                        valid_years = [int(y) for y in years if 1960 <= int(y) <= 2030]
                        if valid_years:
                            # Assume the highest reasonably scoped year is the issue year
                            meta["year"] = int(max(valid_years))
    except Exception:
        pass
    return meta


def download_pdf(
    url: str,
    output_path: str,
    session: requests.Session,
    skip_existing: bool = True,
) -> bool:
    """
    Download a PDF file.
    Returns True if downloaded, False if skipped or failed.
    """
    if skip_existing and os.path.exists(output_path):
        return False

    try:
        resp = session.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()

        # Verify it's actually a PDF (or at least not HTML error)
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type.lower() and "pdf" not in url.lower():
            print(f"  [!] Skipped (HTML response, not PDF): {url}")
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  [OK] Downloaded ({size_kb:.0f} KB): {os.path.basename(output_path)}")
        return True

    except requests.RequestException as e:
        print(f"  [FAIL] Failed: {url} -- {e}")
        return False


def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    # Remove/replace problematic characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    # Cap length
    if len(name) > 200:
        name = name[:200]
    if not name.endswith(".pdf"):
        name += ".pdf"
    return name


# ------------------------------------------------------------------ #
# Manifest (track what we've downloaded)
# ------------------------------------------------------------------ #

def load_manifest(path: str = MANIFEST_FILE) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"downloads": [], "last_run": None}


def save_manifest(manifest: dict, path: str = MANIFEST_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ #
# Main orchestration
# ------------------------------------------------------------------ #

def scrape_legislation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    filter_type: str = None,
    filter_year: int = None,
    max_downloads: int = None,
    skip_existing: bool = True,
):
    """Scrape and download legislation PDFs from the Browse Legislation page."""
    print("=" * 60)
    print("  DLP Legislation Scraper")
    print("=" * 60)
    print(f"  Source:  {BROWSE_URL}")
    print(f"  Output:  {output_dir}/")
    if filter_type:
        print(f"  Filter:  type = {filter_type}")
    if filter_year:
        print(f"  Filter:  year = {filter_year}")
    if max_downloads:
        print(f"  Limit:   {max_downloads} downloads")
    print("=" * 60)

    session = requests.Session()
    manifest = load_manifest()

    print("\n[*] Fetching dynamic legislation list (this may take a minute)...")
    soup = fetch_dynamic_page(BROWSE_URL)
    entries = parse_legislation_entries(soup)
    print(f"   Found {len(entries)} legislation PDF(s)")

    # Apply filters
    if filter_type:
        ft = filter_type.lower()
        entries = [e for e in entries if ft in e.get("type", "").lower() or ft in e.get("title", "").lower()]
        print(f"   After type filter: {len(entries)}")

    if filter_year:
        entries = [e for e in entries if e.get("year") == filter_year]
        print(f"   After year filter: {len(entries)}")

    if max_downloads:
        entries = entries[:max_downloads]

    # Download
    downloaded = 0
    skipped = 0
    failed = 0

    os.makedirs(output_dir, exist_ok=True)

    for i, entry in enumerate(entries, 1):
        filename = sanitize_filename(entry["filename"])
        output_path = os.path.join(output_dir, filename)

        print(f"\n[{i}/{len(entries)}] {entry['title']}")

        if download_pdf(entry["pdf_url"], output_path, session, skip_existing):
            downloaded += 1
            entry["downloaded_at"] = datetime.utcnow().isoformat()
            entry["local_path"] = output_path
            manifest["downloads"].append(entry)
        else:
            if os.path.exists(output_path):
                skipped += 1
                print(f"  [SKIP] Already exists: {filename}")
            else:
                failed += 1

        time.sleep(REQUEST_DELAY)

    manifest["last_run"] = datetime.utcnow().isoformat()
    save_manifest(manifest)

    print("\n" + "=" * 60)
    print(f"  [OK]   Downloaded: {downloaded}")
    print(f"  [SKIP] Skipped:    {skipped}")
    print(f"  [FAIL] Failed:     {failed}")
    print(f"  [LOG]  Manifest:   {MANIFEST_FILE}")
    print("=" * 60)

    return downloaded


def scrape_gazette(
    output_dir: str = GAZETTE_OUTPUT_DIR,
    max_downloads: int = None,
    skip_existing: bool = True,
    start_num: int = 400,
):
    """Scrape and download Official Gazette PDFs via sequential serial numbers."""
    print("=" * 60)
    print("  DLP Official Gazette Scraper (Sequential Mode)")
    print("=" * 60)

    session = requests.Session()
    manifest = load_manifest()
    
    os.makedirs(output_dir, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = 0
    consecutive_fails = 0
    
    # We loop up to a safe upper bound. Currently around O G 800.
    max_issues = max_downloads if max_downloads else 1000
    
    for num in range(start_num, max_issues + 1):
        pdf_url = f"https://dlp.dubai.gov.ae/Official%20Gazette/O%20G%20{num}.pdf"
        filename = f"Official_Gazette_{num}.pdf"
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n[Gazette {num}] Official Gazette No. {num}")

        # If it already exists, skipped
        if skip_existing and os.path.exists(output_path):
            print(f"  [SKIP] Already exists: {filename}")
            skipped += 1
            consecutive_fails = 0
            
            # Ensure it is in the manifest
            pdf_meta = extract_gazette_metadata(output_path)
            entry = {
                "title": f"Official Gazette No. {num}",
                "gazette_number": str(num),
                "pdf_url": pdf_url,
                "filename": filename,
                "source": "official_gazette",
                "year": pdf_meta.get("year"),
                "local_path": output_path
            }
            if not any(e.get("filename") == filename and e.get("source") == "official_gazette" for e in manifest.get("downloads", [])):
                manifest.setdefault("downloads", []).append(entry)
            
            continue

        # Try to download
        if download_pdf(pdf_url, output_path, session, skip_existing):
            downloaded += 1
            consecutive_fails = 0
            
            pdf_meta = extract_gazette_metadata(output_path)
            entry = {
                "title": f"Official Gazette No. {num}",
                "gazette_number": str(num),
                "pdf_url": pdf_url,
                "filename": filename,
                "source": "official_gazette",
                "year": pdf_meta.get("year"),
                "downloaded_at": datetime.utcnow().isoformat(),
                "local_path": output_path
            }
            manifest.setdefault("downloads", []).append(entry)
            save_manifest(manifest)
            
        else:
            failed += 1
            consecutive_fails += 1
            if consecutive_fails >= 100:
                print(f"  --> Reached {consecutive_fails} consecutive failures. Stopping, assuming we hit the maximum issued Gazette.")
                break
                
        time.sleep(REQUEST_DELAY)

    manifest["last_run"] = datetime.utcnow().isoformat()
    save_manifest(manifest)
    
    print("\n" + "=" * 60)
    print(f"  [OK]   Downloaded: {downloaded}")
    print(f"  [SKIP] Skipped:    {skipped}")
    print(f"  [FAIL] Failed:     {failed}")
    print(f"  [LOG]  Manifest:   {MANIFEST_FILE}")
    print("=" * 60)

    return downloaded


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Download UAE Dubai legislations from dlp.dubai.gov.ae"
    )
    parser.add_argument(
        "--gazette",
        action="store_true",
        help="Download Official Gazette PDFs instead of legislation",
    )
    parser.add_argument(
        "--type",
        dest="filter_type",
        type=str,
        default=None,
        help="Filter by legislation type (e.g., 'law', 'resolution', 'decree')",
    )
    parser.add_argument(
        "--year",
        dest="filter_year",
        type=int,
        default=None,
        help="Filter by year (e.g., 2024)",
    )
    parser.add_argument(
        "--max",
        dest="max_downloads",
        type=int,
        default=None,
        help="Maximum number of PDFs to download",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        type=str,
        default=None,
        help="Output directory (default: pdf_english/ or pdf_gazette/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists",
    )

    args = parser.parse_args()

    if args.gazette:
        output = args.output_dir or GAZETTE_OUTPUT_DIR
        scrape_gazette(
            output_dir=output,
            max_downloads=args.max_downloads,
            skip_existing=not args.force,
        )
    else:
        output = args.output_dir or DEFAULT_OUTPUT_DIR
        scrape_legislation(
            output_dir=output,
            filter_type=args.filter_type,
            filter_year=args.filter_year,
            max_downloads=args.max_downloads,
            skip_existing=not args.force,
        )


if __name__ == "__main__":
    main()
