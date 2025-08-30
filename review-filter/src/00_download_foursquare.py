# src/00_download_foursquare.py
import os, sys, time, csv, argparse, requests
from pathlib import Path
from urllib.parse import urlencode

FSQ_API = "https://api.foursquare.com/v3"
NOMINATIM = "https://nominatim.openstreetmap.org/search"

def hdrs(key: str):
    return {
        "Accept": "application/json",
        "Authorization": key.strip(),   # Foursquare v3 Service API Key (fsq3_...)
    }

def geocode_nominatim(query: str, city_hint: str | None = None):
    """
    Free geocoder (no key). Returns (lat, lon) floats or None.
    """
    q = f"{query}, {city_hint}" if city_hint else query
    params = {"q": q, "format": "json", "limit": 1}
    r = requests.get(NOMINATIM, params=params, headers={"User-Agent": "review-filter/1.0"}, timeout=20)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        return None
    return float(arr[0]["lat"]), float(arr[0]["lon"])

def fsq_search_ll(api_key: str, ll: str, query: str, radius=1200, limit=5, categories: str | None = None):
    """
    Place Search with ll (lat,lng). Avoids 'near' which can 410 on some plans.
    Docs: /v3/places/search  (params: ll, query, radius, limit, categories, sort)
    """
    params = {"ll": ll, "query": query, "radius": radius, "limit": limit, "sort": "RELEVANCE"}
    if categories:
        params["categories"] = categories
    r = requests.get(f"{FSQ_API}/places/search", headers=hdrs(api_key), params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Foursquare search failed ({r.status_code}): {r.text}")
    return r.json().get("results", [])

def fsq_details(api_key: str, fsq_id: str, fields="fsq_id,name,location,website,hours,rating,stats,link"):
    r = requests.get(f"{FSQ_API}/places/{fsq_id}", headers=hdrs(api_key), params={"fields": fields}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Foursquare details failed ({r.status_code}): {r.text}")
    return r.json()

def fsq_tips_paged(api_key: str, fsq_id: str, per_page=50, max_total=200, sort="NEWEST"):
    """
    Fetch tips with pagination using 'cursor'. Returns list of tips dicts.
    Docs: /v3/places/{fsq_id}/tips  (supports limit & cursor)
    """
    tips = []
    cursor = None
    while True:
        params = {"limit": min(per_page, max_total - len(tips)), "sort": sort}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{FSQ_API}/places/{fsq_id}/tips", headers=hdrs(api_key), params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Foursquare tips failed ({r.status_code}): {r.text}")
        batch = r.json() or []
        tips.extend(batch)
        # pagination: cursor comes via 'Link' header, but many SDKs also echo a 'cursor' in JSON; check both
        link = r.headers.get("Link", "")
        next_cursor = None
        # parse Link: <...cursor=XYZ>; rel="next"
        if "rel=\"next\"" in link:
            # crude parse
            try:
                part = link.split(";")[0].strip().strip("<>").split("cursor=")[-1]
                next_cursor = part
            except Exception:
                next_cursor = None
        # some responses include 'cursor' in the last element/meta (not always). Stop when no growth or at cap.
        if not next_cursor or len(tips) >= max_total or not batch or len(batch) < params["limit"]:
            break
        cursor = next_cursor
        time.sleep(0.15)
    return tips[:max_total]

def normalize_rows(place: dict, tips: list, page_index: int = 1):
    rows = []
    fsq_id = place.get("fsq_id")
    url = place.get("link") or f"https://foursquare.com/v/{fsq_id}"
    for t in tips:
        tid = t.get("id")
        text = (t.get("text") or "").strip()
        created = t.get("created_at") or t.get("createdAt")
        author_obj = t.get("author") or {}
        author = author_obj.get("name") or " ".join(filter(None, [author_obj.get("firstName"), author_obj.get("lastName")])).strip() or None
        rows.append({
            "id": f"{fsq_id}:{tid}",
            "author": author,
            "rating": None,
            "title": None,
            "text": text,
            "date": created,
            "url": url,
            "source": "foursquare",
            "page": page_index
        })
    return rows

def save_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id","author","rating","title","text","date","url","source","page"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c) for c in cols})
    print(f"[done] Saved {len(rows)} rows -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help='e.g., "Gardens by the Bay"')
    ap.add_argument("--city-hint", default="Singapore", help="Used for geocoding bias only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-key", default=os.getenv("FSQ_API_KEY"))
    ap.add_argument("--max-places", type=int, default=3)
    ap.add_argument("--radius", type=int, default=1200, help="Search radius in meters")
    ap.add_argument("--tips-per-place", type=int, default=100, help="Max tips per place (paged)")
    args = ap.parse_args()

    key = (args.api_key or "").strip()
    if not key.startswith("fsq3_"):
        print("ERROR: Missing/invalid Foursquare Service API Key (fsq3_...). Use --api-key or set FSQ_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # Geocode once to get ll
    pos = geocode_nominatim(args.query, args.city-hint if False else args.city_hint)  # keep mypy calm
    if not pos:
        print("Could not geocode that query.", file=sys.stderr); save_csv([], Path(args.out)); return
    lat, lon = pos
    ll = f"{lat:.6f},{lon:.6f}"

    # Search with ll (avoid 'near' -> 410)
    results = fsq_search_ll(key, ll=ll, query=args.query, radius=args.radius, limit=max(1, args.max_places))
    if not results:
        print("No Foursquare places found near that location/query.")
        save_csv([], Path(args.out)); return

    all_rows = []
    for idx, pl in enumerate(results[:args.max_places], start=1):
        fsq_id = pl.get("fsq_id")
        if not fsq_id: continue
        try:
            details = fsq_details(key, fsq_id)
        except Exception:
            details = {"fsq_id": fsq_id, "link": None}
        tips = fsq_tips_paged(key, fsq_id, per_page=50, max_total=args.tips_per_place)
        all_rows.extend(normalize_rows(details, tips, page_index=idx))
        time.sleep(0.15)

    save_csv(all_rows, Path(args.out))

if __name__ == "__main__":
    main()
