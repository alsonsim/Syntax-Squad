# src/00_download.py
import argparse, time, re, sys, random
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlencode, parse_qs, quote_plus

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd

# ---------- Headers / sessions ----------

UAS = [
    # A few realistic desktop UAs; rotate to reduce blocks
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
]

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(UAS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        # A referer helps on some sites
        "Referer": "https://www.google.com/",
    })
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def save_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # canonical schema
    keep = ["id","author","rating","title","text","date","url","source","page"]
    for k in keep:
        if k not in df.columns: df[k] = None
    df[keep].to_csv(out_path, index=False, encoding="utf-8")
    print(f"[done] Saved {len(df)} rows -> {out_path}")

# -------------------------- TripAdvisor discovery + crawl --------------------------

def normalize_ta_domain(url: str) -> str:
    # Use global domain to reduce country TLD blocks
    return (url
            .replace("tripadvisor.com.sg", "tripadvisor.com")
            .replace("tripadvisor.com.au", "tripadvisor.com")
            .replace("tripadvisor.co.uk", "tripadvisor.com"))

def ta_search_url(query: str, city: Optional[str] = None, country: Optional[str] = None) -> str:
    q = query
    if city:    q += f" {city}"
    if country: q += f" {country}"
    return f"https://www.tripadvisor.com/Search?q={quote_plus(q)}"

def ta_discover(sess: requests.Session, query: str,
                city: Optional[str] = None,
                country: Optional[str] = None) -> Optional[str]:
    url = ta_search_url(query, city, country)
    print(f"[TA] Discover: {url}")
    r = sess.get(url, timeout=30)
    if r.status_code != 200:
        print(f"[TA] HTTP {r.status_code} during discovery")
        return None
    soup = BeautifulSoup(r.text, "lxml")

    candidates = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if any(key in href for key in ("Attraction_Review", "Hotel_Review", "/Restaurant_Review")):
            full = href if href.startswith("http") else f"https://www.tripadvisor.com{href}"
            candidates.append(full)

    if candidates:
        picked = normalize_ta_domain(candidates[0])
        print(f"[TA] Picked: {picked}")
        return picked
    print("[TA] No suitable results found.")
    return None

def ta_build_page_url(base_url: str, page_index: int) -> str:
    # TripAdvisor accepts -or{offset}- before -Reviews-; assume ~10 per page step
    offset = page_index * 10
    return re.sub(r"-Reviews-", f"-Reviews-or{offset}-", base_url)

def parse_tripadvisor(html: str, page_url: str, page_idx: int):
    soup = BeautifulSoup(html, "lxml")
    rows = []
    # TA changes CSS often; try multiple buckets
    cards = soup.select('[data-test-target="HR_CC_CARD"], div.review-container, div.YibKl')
    if not cards:
        cards = soup.select("div.review, div[data-reviewid]")
    for card in cards:
        rid = card.get("data-reviewid") or card.get("data-reviewId")
        # author
        author = None
        a1 = card.select_one('[data-test-target="reviewer-info"] span, a.ui_header_link')
        if a1: author = a1.get_text(strip=True)
        # title
        title = None
        t1 = card.select_one('[data-test-target="review-title"] span, a.review_title, .glasR4aX')
        if t1: title = t1.get_text(" ", strip=True)
        # text
        text = None
        t2 = card.select_one('[data-test-target="review-text"] span, q span, .QewHA, .pIRBV')
        if t2: text = t2.get_text(" ", strip=True)
        # rating
        rating = None
        r1 = card.select_one('.ui_bubble_rating, [class*="bubble_"], [aria-label*="of 5 bubbles"]')
        if r1:
            aria = r1.get("aria-label","")
            m = re.search(r"([0-9.]+)\s*out of 5", aria)
            if m:
                rating = float(m.group(1))
            else:
                m2 = re.search(r"bubble_(\d+)", " ".join(r1.get("class", [])))
                if m2:
                    rating = int(m2.group(1)) / 10.0
        # date
        date = None
        d1 = card.select_one('[data-test-target="review-date"] span, .ratingDate, span.teHYY._R.Me.S4.H3')
        if d1: date = d1.get_text(" ", strip=True)

        if text or title:
            rows.append({
                "id": rid,
                "author": author,
                "rating": rating,
                "title": title,
                "text": text,
                "date": date,
                "url": page_url,
                "source": "tripadvisor",
                "page": page_idx + 1,
            })
    return rows

def crawl_tripadvisor(start_url: str, max_pages: int, delay: float):
    sess = make_session()
    start_url = normalize_ta_domain(start_url)
    all_rows = []
    for p in range(max_pages):
        page_url = ta_build_page_url(start_url, p) if p > 0 else start_url
        print(f"[TA] Page {p+1}: {page_url}")
        resp = sess.get(page_url, timeout=30)
        if resp.status_code == 403:
            print("[warn] 403 Forbidden. Try longer --delay, different UA, or switch to Booking.com for this venue.")
            break
        if resp.status_code != 200:
            print(f"[warn] HTTP {resp.status_code} — stopping.")
            break
        page_rows = parse_tripadvisor(resp.text, page_url, p)
        if not page_rows:
            print("[info] No reviews parsed on this page — stopping.")
            break
        all_rows.extend(page_rows)
        time.sleep(delay + random.uniform(0, 0.8))
    return all_rows

# -------------------------- Booking.com discovery + crawl --------------------------

def booking_search_url(query: str, city: Optional[str] = None, country: Optional[str] = None) -> str:
    q = query
    if city:    q += f" {city}"
    if country: q += f" {country}"
    return f"https://www.booking.com/searchresults.html?ss={quote_plus(q)}"

def booking_discover(sess: requests.Session, query: str,
                     city: Optional[str] = None,
                     country: Optional[str] = None) -> Optional[str]:
    url = booking_search_url(query, city, country)
    print(f"[BK] Discover: {url}")
    r = sess.get(url, timeout=30)
    if r.status_code != 200:
        print(f"[BK] HTTP {r.status_code} during discovery")
        return None
    soup = BeautifulSoup(r.text, "lxml")

    # Look for property links; prefer those that can show reviews tab
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "/hotel/" in href:
            if not href.startswith("http"):
                href = "https://www.booking.com" + href
            if "#tab-reviews" not in href:
                href += "#tab-reviews"
            print(f"[BK] Picked: {href}")
            return href
    print("[BK] No suitable results found.")
    return None

def booking_build_page_url(base_url: str, page_index: int) -> str:
    parsed = urlparse(base_url)
    q = parse_qs(parsed.query)
    step = int(q.get("rows", [10])[0])
    q["offset"] = [str(page_index * step)]
    query_new = urlencode({k: v[0] for k, v in q.items()})
    return parsed._replace(query=query_new).geturl()

def parse_booking(html: str, page_url: str, page_idx: int):
    soup = BeautifulSoup(html, "lxml")
    rows = []

    cards = soup.select('li.review_list_new_item_block, div.review_item, div.c-review-block')
    if not cards:
        cards = soup.select('[data-testid="review-card"]')

    for card in cards:
        # rating
        rating = None
        r1 = card.select_one('[data-testid="review-score"]')
        if r1:
            m = re.search(r"([0-9.]+)", r1.get_text())
            if m: rating = float(m.group(1))
        else:
            r2 = card.select_one('.review-score-badge, .bui-review-score__badge')
            if r2:
                m2 = re.search(r"([0-9.]+)", r2.get_text())
                if m2: rating = float(m2.group(1))
        # author
        author = None
        a1 = card.select_one('[data-testid="review-author"]') or card.select_one('.bui-avatar-block__title')
        if a1: author = a1.get_text(" ", strip=True)
        # date
        date = None
        d1 = card.select_one('[data-testid="review-date"], .c-review-block__date, .review_item_date')
        if d1: date = d1.get_text(" ", strip=True)
        # title
        title = None
        t_title = card.select_one('[data-testid="review-title"]')
        if t_title: title = t_title.get_text(" ", strip=True)
        # text (combine pos/neg parts)
        parts = []
        for tsel in ('[data-testid="review-negative-text"]',
                     '[data-testid="review-positive-text"]',
                     '.c-review__body', '.review_item_review_content'):
            for node in card.select(tsel):
                txt = node.get_text(" ", strip=True)
                if txt and txt not in parts:
                    parts.append(txt)
        text = " ".join(parts).strip() if parts else None

        if text or title:
            rows.append({
                "id": None,
                "author": author,
                "rating": rating,
                "title": title,
                "text": text,
                "date": date,
                "url": page_url,
                "source": "booking",
                "page": page_idx + 1,
            })
    return rows

def crawl_booking(start_url: str, max_pages: int, delay: float):
    sess = make_session()
    all_rows = []
    for p in range(max_pages):
        page_url = booking_build_page_url(start_url, p) if p > 0 else start_url
        print(f"[BK] Page {p+1}: {page_url}")
        r = sess.get(page_url, timeout=30)
        if r.status_code != 200:
            print(f"[BK] HTTP {r.status_code} — stopping.")
            break
        page_rows = parse_booking(r.text, page_url, p)
        if not page_rows:
            print("[BK] No reviews parsed on this page — stopping.")
            break
        all_rows.extend(page_rows)
        time.sleep(delay + random.uniform(0, 0.8))
    return all_rows

# -------------------------- Router --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Direct reviews page (TripAdvisor or Booking.com)")
    ap.add_argument("--query", help="Place name to search (auto-discovery)")
    ap.add_argument("--site", choices=["tripadvisor","booking"], help="Restrict discovery to one site")
    ap.add_argument("--city", help="City to bias discovery (optional)")
    ap.add_argument("--country", help="Country to bias discovery (optional)")
    ap.add_argument("--max-pages", type=int, default=10, help="Max pages to crawl")
    ap.add_argument("--delay", type=float, default=1.8, help="Delay seconds between pages (base; jitter added)")
    ap.add_argument("--out", type=str, help="Output CSV path (optional)")
    args = ap.parse_args()

    sess = make_session()

    # Resolve URL either directly or via discovery
    url = args.url
    if not url:
        if not args.query:
            print("Provide either --url or --query")
            sys.exit(2)
        if args.site == "tripadvisor":
            url = ta_discover(sess, args.query, args.city, args.country)
        elif args.site == "booking":
            url = booking_discover(sess, args.query, args.city, args.country)
        else:
            url = ta_discover(sess, args.query, args.city, args.country) or \
                  booking_discover(sess, args.query, args.city, args.country)
        if not url:
            print("Discovery failed: no suitable URL found.")
            sys.exit(2)

    domain = urlparse(url).netloc.lower()
    if "tripadvisor" in domain:
        rows = crawl_tripadvisor(url, args.max_pages, args.delay)
        default_out = "data/raw/reviews_tripadvisor.csv"
    elif "booking.com" in domain:
        rows = crawl_booking(url, args.max_pages, args.delay)
        default_out = "data/raw/reviews_booking.csv"
    else:
        print("Only TripAdvisor and Booking.com are supported.")
        sys.exit(2)

    out_path = Path(args.out) if args.out else Path(default_out)
    save_csv(rows, out_path)

if __name__ == "__main__":
    main()
