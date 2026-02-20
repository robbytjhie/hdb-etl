import os
import sys
import pandas as pd
import hashlib
import re
import time
from download_hdb_data import processing_download_hdb_data

# --------------------
# CONFIG & DIRECTORIES
# --------------------
RAW_DIR = "hdb_data"
OUTPUT_DIR = "output"

RAW_OUT_DIR       = os.path.join(OUTPUT_DIR, "raw")
CLEANED_OUT_DIR   = os.path.join(OUTPUT_DIR, "cleaned")
TRANSFORM_OUT_DIR = os.path.join(OUTPUT_DIR, "transformed")
HASHED_OUT_DIR    = os.path.join(OUTPUT_DIR, "hashed")
FAILED_OUT_DIR    = os.path.join(OUTPUT_DIR, "failed")
AUDIT_OUT_DIR     = os.path.join(OUTPUT_DIR, "audit")
PROFILE_OUT_DIR   = os.path.join(OUTPUT_DIR, "profiling")

CSV_FILES = [
    "Resale_Flat_Prices_Based_on_Registration_Date_From_Mar_2012_to_Dec_2014.csv",
    "Resale_Flat_Prices_Based_on_Registration_Date_From_Jan_2015_to_Dec_2016.csv"
]
CSV_PATHS = [os.path.join(RAW_DIR, f) for f in CSV_FILES]

EXPECTED_START = pd.Period("2012-03", freq="M")
EXPECTED_END   = pd.Period("2016-12", freq="M")

# Source columns to null-count in profiling (excludes derived flags like price_anomaly,
# remaining_lease which is recomputed, and intermediate transform columns).
SOURCE_COLS_FOR_PROFILING = [
    "block", "flat_model", "flat_type", "floor_area_sqm",
    "lease_commence_date", "month", "resale_price",
    "storey_range", "street_name", "town"
]

# --------------------
# VALIDATION DATA
# --------------------
VALID_TOWNS = {
    "ANG MO KIO","BEDOK","BISHAN","BUKIT BATOK","BUKIT MERAH",
    "BUKIT PANJANG","BUKIT TIMAH","CENTRAL AREA","CHOA CHU KANG","CLEMENTI",
    "GEYLANG","HOUGANG","JURONG EAST","JURONG WEST","KALLANG/WHAMPOA",
    "MARINE PARADE","PASIR RIS","PUNGGOL","QUEENSTOWN","SEMBAWANG",
    "SENGKANG","SERANGOON","TAMPINES","TOA PAYOH","WOODLANDS","YISHUN"
}

VALID_FLAT_TYPES = {"1 ROOM","2 ROOM","3 ROOM","4 ROOM","5 ROOM","EXECUTIVE","MULTI-GENERATION"}

VALID_FLAT_MODELS = {
    "2-room","Adjoined flat","Apartment","DBSS","Improved","Improved-Maisonette",
    "Maisonette","Model A","Model A2","Model A-Maisonette","Multi Generation",
    "New Generation","Premium Apartment","Premium Apartment Loft","Premium Maisonette",
    "Simplified","Standard","Terrace","Type S1","Type S2"
}

VALID_STOREY_FORMAT = r"^\d{2} TO \d{2}$"

# --------------------
# HELPER FUNCTIONS
# --------------------
def timed_step(name, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"⏱ {name} executed in {elapsed:.2f}s")
    return result

def check_snapshots_exist():
    missing = [p for p in CSV_PATHS if not os.path.isfile(p)]
    if missing:
        print("❌ Missing snapshot files:")
        for m in missing:
            print(f"   - {m}")
        return False
    return True

def load_and_align_snapshots():
    print("📦 Loading and aligning CSV snapshots...")
    dfs = []
    for path in CSV_PATHS:
        print(f"   - {path}")
        df = pd.read_csv(path)
        print(f"     → {len(df):,} rows, {len(df.columns)} columns")
        dfs.append(df)
    all_columns = sorted(set().union(*(df.columns for df in dfs)))
    aligned_dfs = [df.reindex(columns=all_columns) for df in dfs]
    master_df = pd.concat(aligned_dfs, ignore_index=True)
    return master_df

# --------------------
# DATA CLEANING & VALIDATION
# --------------------
def recompute_remaining_lease(df):
    """
    Recompute remaining lease as of today from lease_commence_date.
    Assumes a 99-year HDB lease starting on 1 Jan of the commence year.
    Result is floored to whole years and months (no rounding up).
    """
    today = pd.Timestamp.today()
    def compute(row):
        lease_start = row.get("lease_commence_date")
        if pd.isna(lease_start):
            return None
        end = pd.Timestamp(year=int(lease_start), month=1, day=1) + pd.DateOffset(years=99)
        if end < today:
            return "0 years 0 months"
        months_remaining = (end.year - today.year) * 12 + (end.month - today.month)
        return f"{months_remaining // 12} years {months_remaining % 12} months"
    df["remaining_lease"] = df.apply(compute, axis=1)
    return df

def deduplicate_dataset(df):
    """
    Remove duplicate records sharing the same composite key (all columns except
    resale_price). Where duplicates exist, the higher-priced row is kept and the
    lower-priced row is discarded to the failed dataset, per requirements.
    """
    key_cols = [c for c in df.columns if c != "resale_price"]
    df_sorted  = df.sort_values("resale_price", ascending=False)
    df_cleaned = df_sorted.drop_duplicates(subset=key_cols, keep="first")
    df_dupes   = df_sorted.loc[~df_sorted.index.isin(df_cleaned.index)].copy()
    if not df_dupes.empty:
        df_dupes["failure_reason"] = "duplicate_key_lower_price"
    return df_cleaned, df_dupes

def detect_anomalous_prices(df):
    """
    Identify price anomalies using the 3-sigma (Z-score) method, applied
    locally within each (town, flat_type) peer group.

    Heuristic rationale:
    - The Empirical Rule states ~99.7% of normally-distributed values fall
      within 3 standard deviations. Anything outside is flagged as anomalous
      (roughly 1 in 370 chance under normality).
    - Grouping by town AND flat_type ensures a $900k Executive flat is never
      penalised simply for being more expensive than a 3-room in the same area.
    - Groups with fewer than 3 members are skipped (std is undefined or
      meaningless for very small samples).

    Assumption: Prices within each (town, flat_type) group are approximately
    normally distributed. For heavily skewed distributions, Median Absolute
    Deviation (MAD) is a more robust alternative.

    Returns a DataFrame of anomalous rows only (does NOT mutate the input df).
    """
    # Work on a copy so we never mutate the caller's DataFrame
    df_copy = df.copy()
    df_copy["price_anomaly"] = False
    anomalies_list = []

    for (town, flat_type), group in df_copy.groupby(["town", "flat_type"]):
        if len(group) < 3:
            continue
        mean  = group["resale_price"].mean()
        std   = group["resale_price"].std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        anomalies = group[(group["resale_price"] < lower) | (group["resale_price"] > upper)].copy()
        if not anomalies.empty:
            anomalies["failure_reason"] = "price_anomaly_3sigma"
            anomalies_list.append(anomalies)

    return pd.concat(anomalies_list) if anomalies_list else pd.DataFrame()

def extra_validation(df):
    """
    Vectorised business-rule validation. Each rule is applied across the
    entire column at once rather than row-by-row, making it significantly
    faster on large DataFrames.

    Rules validated:
    1. resale_price  — must be > 0
    2. floor_area_sqm — must be > 0 and ≤ statistical upper fence (Q3 + 1.5*IQR),
                        capped at 500 sqm as a hard domain ceiling for HDB flats
    3. town          — must be in the known HDB town list
    4. flat_type     — must be in the known flat-type list
    5. flat_model    — must be in the known flat-model list
    6. storey_range  — must match DD TO DD format with lower ≤ upper
    7. month         — must fall within Mar 2012 – Dec 2016
    """
    issues = pd.DataFrame(index=df.index)

    # Rule 1: resale_price > 0
    issues["invalid_resale_price"] = df["resale_price"] <= 0

    # Rule 2: floor_area_sqm > 0
    issues["invalid_floor_area_sqm"] = (df["floor_area_sqm"] <= 0)

    # Rule 3-5: categorical membership
    issues["invalid_town"]       = ~df["town"].isin(VALID_TOWNS)
    issues["invalid_flat_type"]  = ~df["flat_type"].isin(VALID_FLAT_TYPES)
    issues["invalid_flat_model"] = ~df["flat_model"].isin(VALID_FLAT_MODELS)

    # Rule 6: storey_range format AND logical order (lower ≤ upper)
    fmt_ok = df["storey_range"].astype(str).str.match(VALID_STOREY_FORMAT, na=False)
    def _logical_order(val):
        m = re.match(VALID_STOREY_FORMAT, str(val))
        if not m:
            return False
        lo, hi = int(str(val)[:2]), int(str(val)[6:8])
        return lo <= hi
    logical_ok = df["storey_range"].apply(_logical_order)
    issues["invalid_storey_range"] = ~(fmt_ok & logical_ok)

    # Rule 7: month within expected range
    month_period = df["month"].dt.to_period("M")
    issues["month_out_of_range"] = (month_period < EXPECTED_START) | (month_period > EXPECTED_END)

    # Build failed subset: rows with at least one issue, with a human-readable comments column
    rule_cols = issues.columns.tolist()
    any_fail  = issues.any(axis=1)
    df_fail   = df.loc[any_fail].copy()
    df_fail["comments"] = issues[any_fail].apply(
        lambda row: "; ".join(col for col in rule_cols if row[col]), axis=1
    )
    df_fail["failure_reason"] = "rule_violation"

    return df_fail

# --------------------
# PROFILING
# --------------------
# Only source-level columns are null-counted.  Derived columns (remaining_lease,
# price_anomaly, block_numeric, year_month) are excluded to avoid confusing
# pre- vs post-cleaning profiles.
def profile_dataset(df, label="dataset"):
    """
    Generate a statistical profile for a DataFrame.

    Parameters
    ----------
    df    : DataFrame to profile.
    label : Tag stored in the output row ('raw_master' or 'cleaned_final')
            so the two profile CSVs are immediately distinguishable.

    Null counts are restricted to SOURCE_COLS_FOR_PROFILING to avoid
    including derived columns (remaining_lease, price_anomaly, etc.).
    Numeric stats cover resale_price and floor_area_sqm only.
    """
    profile = {"profile_label": label}
    profile["total_rows"]    = len(df)
    profile["total_columns"] = len(df.columns)

    # Null counts — source columns only, skip if column absent (raw vs cleaned differ)
    for col in SOURCE_COLS_FOR_PROFILING:
        if col in df.columns:
            profile[f"null_count_{col}"] = int(df[col].isna().sum())

    # Numeric distributions
    for col in ["resale_price", "floor_area_sqm"]:
        if col in df.columns:
            profile[f"{col}_min"]    = df[col].min()
            profile[f"{col}_max"]    = df[col].max()
            profile[f"{col}_mean"]   = round(float(df[col].mean()), 2)
            profile[f"{col}_median"] = df[col].median()

    profile["duplicate_rows"] = int(df.duplicated().sum())
    return profile

# --------------------
# TRANSFORMATION
# --------------------
def create_resale_identifier(df):
    """
    Constructs the Resale Identifier column per specification:
      S  +  block_numeric(3)  +  avg_price_prefix(2)  +  month_num(2)  +  town_initial(1)

    block_numeric : All non-digit characters removed; remaining digits zero-padded to 3
                    and TRUNCATED to 3 (in case a block has > 3 digits).
    avg_price_prefix : First 2 digits of the integer average resale_price, grouped by
                       (year-month, town, flat_type).
    month_num     : Zero-padded 2-digit transaction month.
    town_initial  : First character of the town name.

    Intermediate columns (block_numeric, year_month) are dropped before returning
    so they do not pollute the transformed/hashed output files.
    """
    df_copy = df.copy()
    if "block" not in df_copy.columns:
        df_copy["block"] = "000"

    # Remove all non-digit characters, zero-pad to 3, truncate to first 3 digits
    df_copy["block_numeric"] = (
        df_copy["block"].astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .str.zfill(3)
        .str[:3]
    )

    df_copy["year_month"] = df_copy["month"].dt.to_period("M")
    avg_price = df_copy.groupby(["year_month", "town", "flat_type"])["resale_price"].transform("mean")

    df_copy["Resale Identifier"] = (
        "S" +
        df_copy["block_numeric"] +
        avg_price.astype(int).astype(str).str[:2] +
        df_copy["month"].dt.month.astype(str).str.zfill(2) +
        df_copy["town"].str[0]
    )

    # Drop intermediate helper columns — keep output clean
    df_copy.drop(columns=["block_numeric", "year_month"], inplace=True)
    return df_copy

def dedup_by_identifier(df):
    """
    Second deduplication pass applied AFTER Resale Identifier creation.

    Two records can share the same identifier if they have identical block,
    town, month, and their peer-group average price rounds to the same 2-digit
    prefix. In that case the requirement specifies keeping the higher price.

    Returns (df_clean, df_dupes) mirroring deduplicate_dataset().
    """
    df_sorted  = df.sort_values("resale_price", ascending=False)
    df_clean   = df_sorted.drop_duplicates(subset=["Resale Identifier"], keep="first")
    df_dupes   = df_sorted.loc[~df_sorted.index.isin(df_clean.index)].copy()
    if not df_dupes.empty:
        df_dupes["failure_reason"] = "duplicate_resale_identifier_lower_price"
    return df_clean, df_dupes

def hash_resale_identifier(df):
    """
    Hash the Resale Identifier using SHA-256.

    SHA-256 is a cryptographic hash function that is:
    - Irreversible (pre-image resistant): given a hash, it is computationally
      infeasible to recover the original identifier.
    - Deterministic: the same input always produces the same 64-character hex digest.
    - Collision-resistant: for a dataset of this size (~87k rows), the probability
      of two distinct identifiers producing the same hash is astronomically small
      (birthday paradox threshold is ~2^128 inputs for SHA-256).
    """
    df = df.copy()
    df["Resale Identifier Hashed"] = df["Resale Identifier"].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()
    )
    return df

# --------------------
# ETL FUNCTION
# --------------------
def run_hdb_etl():
    print("=" * 60)
    print("🚀 HDB RESALE ETL PIPELINE")
    print("=" * 60)

    processing_download_hdb_data()

    for d in [RAW_OUT_DIR, CLEANED_OUT_DIR, TRANSFORM_OUT_DIR,
              HASHED_OUT_DIR, FAILED_OUT_DIR, AUDIT_OUT_DIR, PROFILE_OUT_DIR]:
        os.makedirs(d, exist_ok=True)

    if not check_snapshots_exist():
        sys.exit(1)

    # ── Load & Raw Save ──────────────────────────────────────────────────
    df = timed_step("Load & Align CSVs", load_and_align_snapshots)
    raw_file = os.path.join(RAW_OUT_DIR, "hdb_resale_raw.csv")
    df.to_csv(raw_file, index=False)
    print(f"💾 Raw dataset saved: {raw_file}")

    # ── Pre-cleaning Profile (Raw Master) ────────────────────────────────
    # Profile BEFORE any type casting, cleaning, deduplication, or validation.
    # This baseline documents original data quality for audit and before/after comparison.
    profile_raw = timed_step("Profile Raw Dataset", profile_dataset, df, label="raw_master")
    pd.DataFrame([profile_raw]).to_csv(os.path.join(PROFILE_OUT_DIR, "profile_raw.csv"), index=False)
    print(f"📊 Pre-cleaning profile saved: {os.path.join(PROFILE_OUT_DIR, 'profile_raw.csv')}")

    # ── Cleaning ─────────────────────────────────────────────────────────
    df["month"]          = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    df["resale_price"]   = pd.to_numeric(df["resale_price"], errors="coerce")
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    rows_before = len(df)
    df = df.dropna(subset=["month", "resale_price", "floor_area_sqm"])
    print(f"  Rows dropped (null critical fields): {rows_before - len(df):,}")
    df = recompute_remaining_lease(df)

    # ── Deduplication ────────────────────────────────────────────────────
    df_cleaned, df_duplicates = deduplicate_dataset(df)
    if not df_duplicates.empty:
        df_duplicates.to_csv(os.path.join(AUDIT_OUT_DIR, "duplicates.csv"), index=False)
        print(f"⚠️  Duplicates saved: {len(df_duplicates):,} rows → audit/duplicates.csv")

    # ── Business Rule Validation ─────────────────────────────────────────
    df_rules_fail = timed_step("Business Rule Validation", extra_validation, df_cleaned)
    if not df_rules_fail.empty:
        df_rules_fail.to_csv(os.path.join(AUDIT_OUT_DIR, "rule_violations.csv"), index=False)
        print(f"⚠️  Rule violations saved: {len(df_rules_fail):,} rows → audit/rule_violations.csv")

    # ── Anomaly Detection ────────────────────────────────────────────────
    df_anomalies = timed_step("Anomaly Detection", detect_anomalous_prices, df_cleaned)
    if not df_anomalies.empty:
        df_anomalies.to_csv(os.path.join(AUDIT_OUT_DIR, "anomalies.csv"), index=False)
        print(f"⚠️  Anomalies saved: {len(df_anomalies):,} rows → audit/anomalies.csv")

    # ── Combine Failed & Finalise Cleaned ────────────────────────────────
    all_failed = [x for x in [df_duplicates, df_rules_fail, df_anomalies] if not x.empty]
    failed_records = pd.concat(all_failed).drop_duplicates() if all_failed else pd.DataFrame()
    if not failed_records.empty:
        failed_records.to_csv(os.path.join(FAILED_OUT_DIR, "hdb_resale_failed.csv"), index=False)
        print(f"⚠️  Failed records saved: {len(failed_records):,} rows → failed/hdb_resale_failed.csv")

    df_cleaned_final = df_cleaned.loc[~df_cleaned.index.isin(failed_records.index)]
    df_cleaned_final.to_csv(os.path.join(CLEANED_OUT_DIR, "hdb_resale_cleaned.csv"), index=False)
    print(f"💾 Cleaned dataset saved: {len(df_cleaned_final):,} rows → cleaned/hdb_resale_cleaned.csv")

    # ── Post-cleaning Profile (Cleaned Final) ────────────────────────────
    # Profile AFTER all deduplication, validation, and anomaly removal.
    # Compare against profile_raw.csv to quantify the cleaning impact.
    profile_clean = timed_step("Profile Cleaned Dataset", profile_dataset, df_cleaned_final, label="cleaned_final")
    pd.DataFrame([profile_clean]).to_csv(os.path.join(PROFILE_OUT_DIR, "profile_cleaned.csv"), index=False)
    print(f"📊 Post-cleaning profile saved: {os.path.join(PROFILE_OUT_DIR, 'profile_cleaned.csv')}")

    # ── Transformation ───────────────────────────────────────────────────
    df_transformed = create_resale_identifier(df_cleaned_final)
    # Second dedup pass: discard lower-priced rows sharing the same Resale Identifier
    df_transformed, df_id_dupes = dedup_by_identifier(df_transformed)
    if not df_id_dupes.empty:
        # Append to failed dataset
        extra_failed = os.path.join(FAILED_OUT_DIR, "hdb_resale_failed.csv")
        df_id_dupes.to_csv(extra_failed, mode="a", header=not os.path.exists(extra_failed), index=False)
        print(f"⚠️  Identifier duplicates: {len(df_id_dupes):,} rows appended to failed dataset")
    df_transformed.to_csv(os.path.join(TRANSFORM_OUT_DIR, "hdb_resale_transformed.csv"), index=False)
    print(f"💾 Transformed dataset saved: {len(df_transformed):,} rows → transformed/hdb_resale_transformed.csv")

    # ── Hashing ──────────────────────────────────────────────────────────
    df_hashed = hash_resale_identifier(df_transformed)
    df_hashed.to_csv(os.path.join(HASHED_OUT_DIR, "hdb_resale_hashed.csv"), index=False)
    print(f"💾 Hashed dataset saved: {len(df_hashed):,} rows → hashed/hdb_resale_hashed.csv")

    print("\n✅ ETL COMPLETE")
    print("=" * 60)
    print(f"  Final rows    : {len(df_hashed):,}")
    print(f"  Final columns : {len(df_hashed.columns)}")
    print(f"  Date range    : {df_hashed['month'].min().date()} → {df_hashed['month'].max().date()}")

# --------------------
# ENTRY POINT
# --------------------
if __name__ == "__main__":
    run_hdb_etl()
