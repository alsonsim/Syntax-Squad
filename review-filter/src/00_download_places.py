# src/00_download_places.py
import os, sys, time, argparse, requests, csv
from pathlib import Path

API = "https://places.googleapis.com/v1"
HEADERS = lambda key: {
    "X-Goog-Api-Key": key,
    # Field mask tells Google which fields you want back (Places API v1).
    # Details: https://developers.google.com/maps/documentation/places/web-service/place-details
    "X-Goog-FieldMask": (
        "places.id,places.name,places.displayName,places.googleMapsUri"
        ",places.rating,places.userRatingCount"
    )
}

DETAILS_FIELDS = (
    "id,name,displayName,googleMapsUri,rating,userRatingCount,"
    "reviews.text,reviews.rating,reviews.publishTime,reviews.authorAttribution"
)

def search_place_id(api_key: str, text_query: str) -> dict | None:
    r = requests.post(
        f"{API}/places:searchText",
        headers=HEADERS(api_key),
        json={"textQuery": text_query}
    )
    r.raise_for_status()
    data = r.json()
    places = data.get("places", [])
    return places[0] if places else None

def get_place_details(api_key: str, place_id: str, field_mask: str = DETAILS_FIELDS) -> dict:
    # For Details, the field mask is usually passed via header; allow ?fields for clarity.
    h = HEADERS(api_key).copy()
    h["X-Goog-FieldMask"] = field_mask
    r = requests.get(f"{API}/places/{place_id}", headers=h)
    r.raise_for_status()
    return r.json()

def normalize_reviews(place: dict, page_index: int = 1):
    out = []
    pid = place.get("id")
    maps_uri = place.get("googleMapsUri")
    display = (place.get("displayName") or {}).get("text")
    reviews = place.get("reviews", []) or []
    for rv in reviews:
        text = (rv.get("text") or {}).get("text") or ""
        rating = rv.get("rating")
        date = rv.get("publishTime")
        author_attr = rv.get("authorAttribution") or {}
        author = author_attr.get("displayName")
        # No stable review ID is exposed by Places API; synthesize one deterministically
        rid = f"{pid}:{author}:{date}"
        out.append({
            "id": rid,
            "author": author,
            "rating": rating,
            "title": None,            # Google reviews don't have a separate title
            "text": text,
            "date": date,
            "url": maps_uri,
            "source": "google_places",
            "page": page_index
        })
    return out

def save_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id","author","rating","title","text","date","url","source","page"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

def discover_related(api_key: str, seed_place: dict, radius_m=400):
    """Find a few related places near the seed (to collect more 5-review chunks)."""
    loc = seed_place.get("location", {})
    if not loc:
        return []
    lat, lng = loc.get("latitude"), loc.get("longitude")
    h = HEADERS(api_key).copy()
    h["X-Goog-FieldMask"] = "places.id,places.name,places.displayName,places.location,places.googleMapsUri"
    r = requests.get(
        f"{API}/places:searchNearby",
        headers=h,
        params={"location": f"{lat},{lng}", "radius": radius_m, "type": "tourist_attraction"}
    )
    r.raise_for_status()
    return r.json().get("places", []) or []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Text query for the seed place (e.g., 'Gardens by the Bay')")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--api-key", default=os.getenv("GOOGLE_MAPS_API_KEY"))
    ap.add_argument("--nearby", action="store_true", help="Also pull reviews from nearby places for more volume")
    ap.add_argument("--nearby-radius", type=int, default=400, help="Meters for nearby search")
    ap.add_argument("--nearby-max", type=int, default=6, help="Max number of nearby places to include")
    args = ap.parse_args()
    if not args.api_key:
        print("ERROR: Provide --api-key or set GOOGLE_MAPS_API_KEY.", file=sys.stderr)
        sys.exit(1)

    seed = search_place_id(args.api_key, args.query)
    if not seed:
        print("No place found for that query.")
        sys.exit(0)

    # Seed details + reviews
    details = get_place_details(args.api_key, seed["id"])
    rows = normalize_reviews(details, page_index=1)

    # Optionally, discover related places around the seed and grab their 5 reviews each
    if args.nearby:
        related = discover_related(args.api_key, details, radius_m=args.nearby_radius)
        # Dedup & limit
        seen = {seed["id"]}
        extras = [p for p in related if p.get("id") not in seen][:args.nearby_max]
        for i, pl in enumerate(extras, start=2):
            d = get_place_details(args.api_key, pl["id"])
            rows.extend(normalize_reviews(d, page_index=i))
            time.sleep(0.25)  # be polite

    save_csv(rows, Path(args.out))
    print(f"[done] Saved {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
