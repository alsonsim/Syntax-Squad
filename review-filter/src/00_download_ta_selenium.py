# src/00_download_ta_selenium.py
import time, re, json, argparse
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from webdriver_manager.chrome import ChromeDriverManager

# ----------------------- Parsing (HTML) -----------------------

def parse_tripadvisor_html(html, page_url, page_idx):
    soup = BeautifulSoup(html, "lxml")
    rows = []

    # Be generous with selectors; TA A/B-tests their markup
    cards = soup.select('[data-test-target="HR_CC_CARD"], div.review-container, div.YibKl')
    if not cards:
        cards = soup.select("div.review, div[data-reviewid]")

    for card in cards:
        rid = card.get("data-reviewid") or card.get("data-reviewId")

        author = None
        a1 = card.select_one('[data-test-target="reviewer-info"] span, a.ui_header_link')
        if a1:
            author = a1.get_text(strip=True)

        title = None
        t1 = card.select_one('[data-test-target="review-title"] span, a.review_title, .glasR4aX')
        if t1:
            title = t1.get_text(" ", strip=True)

        text = None
        t2 = card.select_one('[data-test-target="review-text"] span, q span, .QewHA, .pIRBV')
        if t2:
            text = t2.get_text(" ", strip=True)

        rating = None
        r1 = card.select_one('.ui_bubble_rating, [class*="bubble_"], [aria-label*="of 5 bubbles"]')
        if r1:
            aria = r1.get("aria-label", "")
            m = re.search(r"([0-9.]+)\s*out of 5", aria)
            if m:
                rating = float(m.group(1))
            else:
                m2 = re.search(r"bubble_(\d+)", " ".join(r1.get("class", [])))
                if m2:
                    rating = int(m2.group(1)) / 10.0

        date = None
        d1 = card.select_one('[data-test-target="review-date"] span, .ratingDate, span.teHYY._R.Me.S4.H3')
        if d1:
            date = d1.get_text(" ", strip=True)

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
                "page": page_idx + 1
            })
    return rows

# ----------------------- Parsing (JSON fallback) -----------------------

def _walk_reviews_from_obj(obj):
    """Yield dicts that look like review records from any nested JSON structure."""
    if isinstance(obj, dict):
        keys = set(obj.keys())
        if {"id","text"}.issubset(keys) or {"reviewId","reviewText"}.issubset(keys):
            yield obj
        for v in obj.values():
            yield from _walk_reviews_from_obj(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _walk_reviews_from_obj(it)

def parse_tripadvisor_json_state(driver, page_url, page_idx):
    """
    Pull reviews from window.__WEB_CONTEXT__ (or similar) when DOM cards aren't present.
    Returns list of rows with the same shape as parse_tripadvisor_html().
    """
    rows = []
    try:
        raw = driver.execute_script(
            "return JSON.stringify(window.__WEB_CONTEXT__ || window.__WEB_CONTEXT__PATCH || null)"
        )
        if not raw:
            return rows
        data = json.loads(raw)
    except Exception:
        return rows

    for rv in _walk_reviews_from_obj(data):
        rid   = rv.get("id") or rv.get("reviewId")
        text  = rv.get("text") or rv.get("reviewText") or rv.get("reviewTextRaw")
        title = rv.get("title")
        rating = None
        for rk in ("rating", "ratingValue", "bubbleRating", "ratingNumber"):
            if rk in rv and isinstance(rv[rk], (int,float)):
                rating = float(rv[rk]); break

        author = None
        for ak in ("userName", "author", "user", "username", "displayName"):
            if ak in rv:
                v = rv[ak]
                author = v if isinstance(v,str) else v.get("username") or v.get("displayName")
                break

        date = rv.get("publishedDate") or rv.get("creationDate") or rv.get("localizedDateOfExperience")

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
                "page": page_idx + 1
            })
    return rows

# ----------------------- CSV helper -----------------------

def save_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    keep = ["id","author","rating","title","text","date","url","source","page"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    df[keep].to_csv(out_path, index=False, encoding="utf-8")
    print(f"[done] Saved {len(df)} rows -> {out_path}")

# ----------------------- Consent + “poke” helpers -----------------------

def try_dismiss_consent(driver, total_timeout=8):
    selectors = [
        'button#onetrust-accept-btn-handler',
        'button[aria-label="Accept all"]',
        'button[aria-label="I agree"]',
        'button[aria-label="Accept"]',
    ]
    deadline = time.time() + total_timeout

    def click_first_match():
        for sel in selectors:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            if elems:
                try:
                    driver.execute_script("arguments[0].click();", elems[0])
                    return True
                except Exception:
                    pass
        return False

    if click_first_match():
        return

    # also try iframes
    frames = driver.find_elements(By.CSS_SELECTOR, "iframe")
    for fr in frames:
        try:
            driver.switch_to.frame(fr)
            if click_first_match():
                driver.switch_to.default_content()
                return
        except Exception:
            driver.switch_to.default_content()
        finally:
            driver.switch_to.default_content()

    while time.time() < deadline:
        if click_first_match():
            break
        time.sleep(0.2)
    driver.switch_to.default_content()

def ensure_reviews_ready(driver, max_scrolls=3):
    # Click a "Reviews" tab if present (no blocking wait)
    for sel in [
        'a[aria-controls*="REVIEWS"]',
        'a[data-tab-name="Reviews"]',
        'a[data-test-target="reviews-tab"]',
        'a[href*="#REVIEWS"]',
        'a[href*="-Reviews-"]',
    ]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            driver.execute_script("arguments[0].click();", el)
            break
        except Exception:
            pass
    # Nudge the page to trigger lazy loads
    for i in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * arguments[0]);", 0.35 + 0.2*i)
        time.sleep(0.6)

# ----------------------- Main crawl -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="TripAdvisor Attraction/Hotel/Restaurant review URL (.com)")
    ap.add_argument("--max-pages", type=int, default=10, help="Max pages to crawl")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    args = ap.parse_args()

    url = args.url.replace("tripadvisor.com.sg", "tripadvisor.com")

    options = uc.ChromeOptions()
    if args.headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1366,2200")
    options.add_argument("--lang=en-US")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-allow-origins=*")
    # Make headless look more like real Chrome
    options.add_argument("–-disable-features=IsolateOrigins,site-per-process")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0")

    driver = uc.Chrome(driver_executable_path=ChromeDriverManager().install(), options=options)
    wait = WebDriverWait(driver, 8)
    all_rows = []

    try:
        driver.get(url)
        try_dismiss_consent(driver, total_timeout=8)

        for p in range(args.max_pages):
            print(f"[TA] Page {p+1}")

            # Light “poke” (tab + scroll), but do NOT block on container waits
            ensure_reviews_ready(driver)

            page_rows = []
            # up to 4 quick attempts: HTML → JSON → scroll → retry
            for attempt in range(4):
                html = driver.page_source

                # 1) Try regular HTML parsing
                page_rows = parse_tripadvisor_html(html, driver.current_url, p)
                if page_rows:
                    break

                # 2) JSON state fallback
                page_rows = parse_tripadvisor_json_state(driver, driver.current_url, p)
                if page_rows:
                    break

                # 3) nudge & retry
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight * arguments[0]);", 0.85)
                time.sleep(0.8)

            if not page_rows:
                print("[info] no reviews parsed on this page; stopping.")
                break

            all_rows.extend(page_rows)

            # Try to click "Next page"
            next_clicked = False
            for sel in ['a[aria-label="Next page"]', 'a[aria-label="Next"]', 'a.ui_button.nav.next.primary']:
                elems = driver.find_elements(By.CSS_SELECTOR, sel)
                if elems:
                    try:
                        driver.execute_script("arguments[0].click();", elems[0])
                        next_clicked = True
                        break
                    except Exception:
                        continue
            if not next_clicked:
                print("[info] No next button; stopping.")
                break

            time.sleep(2 + p * 0.1)

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    save_csv(all_rows, Path(args.out))


if __name__ == "__main__":
    main()
