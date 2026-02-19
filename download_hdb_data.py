"""
HDB Resale Flat Prices - Auto-discover datasets by name keywords.
Fetches collection metadata, filters by keywords, then downloads matches.

Just edit KEYWORDS to target any datasets you want — no manual IDs needed.
"""

import requests
import os
import time

COLLECTION_ID = 189
COLLECTION_API = "https://api-production.data.gov.sg/v2/public/api"
DATASET_API = "https://api-open.data.gov.sg/v1/public/api/datasets"
OUTPUT_DIR = "hdb_data"

# ─────────────────────────────────────────────────────
# CONFIGURE: all keywords must appear in the dataset name (case-insensitive)
# Each tuple = one dataset to find
# ─────────────────────────────────────────────────────
KEYWORDS = [
    ("Registration Date", "Mar 2012", "Dec 2014"),   # matches "...Registration Date...From Mar 2012 to Dec 2014"
    ("Registration Date", "Jan 2015", "Dec 2016"),   # matches "...Registration Date...From Jan 2015 to Dec 2016"
]


def get_with_retry(url, max_retries=6, timeout=30):
    """GET with exponential backoff on 429 rate limit."""
    wait = 10
    for attempt in range(1, max_retries + 1):
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 429:
            print(f"    Rate limited. Waiting {wait}s (retry {attempt}/{max_retries})...")
            time.sleep(wait)
            wait *= 2
            continue
        resp.raise_for_status()
        return resp
    raise Exception(f"Still rate limited after {max_retries} retries.")


def discover_datasets():
    """Fetch collection metadata and match datasets by keywords."""
    print("Fetching collection metadata...")
    resp = requests.get(
        f"{COLLECTION_API}/collections/{COLLECTION_ID}/metadata", timeout=30
    )
    resp.raise_for_status()
    data = resp.json()

    child_ids = (
        data.get("data", {})
        .get("collectionMetadata", {})
        .get("childDatasets", [])
    )
    print(f"Found {len(child_ids)} datasets in collection. Fetching names...\n")

    # Build id -> name map
    id_to_name = {}
    for ds_id in child_ids:
        meta_resp = get_with_retry(f"{COLLECTION_API}/datasets/{ds_id}/metadata")
        name = meta_resp.json().get("data", {}).get("name", "")
        id_to_name[ds_id] = name
        print(f"  {ds_id} → {name}")
        time.sleep(1)

    # Match against keywords
    matched = {}
    print()
    for keywords in KEYWORDS:
        for ds_id, name in id_to_name.items():
            if all(kw.lower() in name.lower() for kw in keywords):
                safe = name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
                safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in safe).strip("_")
                filename = f"{safe}.csv"
                matched[ds_id] = (name, filename)
                print(f"  Matched: [{', '.join(keywords)}] → {name}")
                break
        else:
            print(f"  No match found for keywords: {keywords}")

    return matched


def download_dataset(dataset_id, name, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Step 1: Initiate
    print(f"  Initiating download...")
    resp = get_with_retry(f"{DATASET_API}/{dataset_id}/initiate-download")
    if resp.json().get("code") != 0:
        raise Exception(f"Initiate failed: {resp.json()}")
    time.sleep(5)

    # Step 2: Poll for URL
    print(f"  Polling for download URL...")
    dl_url = None
    for attempt in range(1, 15):
        resp = get_with_retry(f"{DATASET_API}/{dataset_id}/poll-download")
        dl_url = resp.json().get("data", {}).get("url")
        if dl_url:
            print(f"  URL ready.")
            break
        print(f"    Not ready yet (attempt {attempt}/14), waiting 5s...")
        time.sleep(5)

    if not dl_url:
        raise Exception("Timed out waiting for download URL.")

    # Step 3: Download CSV
    print(f"  Downloading...")
    resp = requests.get(dl_url, timeout=120, stream=True)
    resp.raise_for_status()

    total = 0
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)

    print(f"  Saved: {filepath} ({total / 1024:.1f} KB)")


def processing_download_hdb_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("HDB Resale Flat Prices — Auto-discover & Download")
    print("=" * 60)

    # Auto-discover matching datasets
    matched = discover_datasets()

    if not matched:
        print("\nNo datasets matched. Check your KEYWORDS and try again.")
        return

    print(f"\nDownloading {len(matched)} matched dataset(s)...\n")
    print("=" * 60)

    items = list(matched.items())
    for i, (ds_id, (name, filename)) in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}] {name}")
        try:
            download_dataset(ds_id, name, filename)
        except requests.exceptions.ConnectionError as e:
            print(f"  Connection error: {e}")
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP error: {e}")
        except Exception as e:
            print(f"  Error: {e}")

        if i < len(items):
            print(f"\n  Pausing 15s before next download...")
            time.sleep(15)

    print("\n" + "=" * 60)
    print(f"Done! Files saved in './{OUTPUT_DIR}/'")

    
if __name__ == "__main__":
    processing_download_hdb_data()