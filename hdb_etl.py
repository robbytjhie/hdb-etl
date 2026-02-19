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

RAW_OUT_DIR = os.path.join(OUTPUT_DIR, "raw")
CLEANED_OUT_DIR = os.path.join(OUTPUT_DIR, "cleaned")
TRANSFORM_OUT_DIR = os.path.join(OUTPUT_DIR, "transformed")
HASHED_OUT_DIR = os.path.join(OUTPUT_DIR, "hashed")
FAILED_OUT_DIR = os.path.join(OUTPUT_DIR, "failed")
AUDIT_OUT_DIR = os.path.join(OUTPUT_DIR, "audit")
PROFILE_OUT_DIR = os.path.join(OUTPUT_DIR, "profiling")

CSV_FILES = [
    "Resale_Flat_Prices_Based_on_Registration_Date_From_Mar_2012_to_Dec_2014.csv",
    "Resale_Flat_Prices_Based_on_Registration_Date_From_Jan_2015_to_Dec_2016.csv"
]
CSV_PATHS = [os.path.join(RAW_DIR, f) for f in CSV_FILES]

EXPECTED_START = pd.Period("2012-03", freq="M")
EXPECTED_END = pd.Period("2016-12", freq="M")

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
    end = time.time()
    print(f"⏱ {name} executed in {end - start:.2f}s")
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
        dfs.append(df)
    all_columns = sorted(set().union(*(df.columns for df in dfs)))
    aligned_dfs = [df.reindex(columns=all_columns) for df in dfs]
    master_df = pd.concat(aligned_dfs, ignore_index=True)
    return master_df

# --------------------
# DATA CLEANING & VALIDATION
# --------------------
def recompute_remaining_lease(df):
    today = pd.Timestamp.today()
    def compute(row):
        lease_start = row.get("lease_commence_date")
        if pd.isna(lease_start):
            return None
        end = pd.Timestamp(year=int(lease_start), month=1, day=1) + pd.DateOffset(years=99)
        if end < today:
            return "0 years 0 months"
        months_remaining = (end.year - today.year) * 12 + (end.month - today.month)
        return f"{months_remaining//12} years {months_remaining%12} months"
    df['remaining_lease'] = df.apply(compute, axis=1)
    return df

def deduplicate_dataset(df):
    key_cols = [c for c in df.columns if c != "resale_price"]
    df_sorted = df.sort_values("resale_price", ascending=False)
    df_cleaned = df_sorted.drop_duplicates(subset=key_cols, keep="first")
    df_duplicates = df_sorted.loc[~df_sorted.index.isin(df_cleaned.index)]
    return df_cleaned, df_duplicates

def detect_anomalous_prices(df):
    """
    Identifies price anomalies by comparing transactions within the
    same Town and Flat Type using a 3-sigma (Z-score) threshold.
    """
    df['price_anomaly'] = False
    anomalies_list = []

    # Group by both Town and Flat Type for localized context
    # Note: Using 'town' and 'flat_type' as assumed column names
    for (town, flat_type), group in df.groupby(['town', 'flat_type']):

        # Calculate local statistics for this specific town/flat combination
        mean = group['resale_price'].mean()
        std = group['resale_price'].std()

        # Define bounds (assuming 3 standard deviations)
        lower = mean - 3 * std
        upper = mean + 3 * std

        # Identify rows that fall outside these localized bounds
        anomalies = group[(group['resale_price'] < lower) | (group['resale_price'] > upper)]

        if not anomalies.empty:
            df.loc[anomalies.index, 'price_anomaly'] = True
            anomalies_list.append(anomalies)

    return pd.concat(anomalies_list) if anomalies_list else pd.DataFrame()

def extra_validation(df):
    rows = []
    for _, r in df.iterrows():
        issues = []
        if r.get('resale_price',0) <= 0:
            issues.append("invalid resale_price")
        if r.get('floor_area_sqm',0) <= 0 or r.get('floor_area_sqm',0) > 500:
            issues.append("invalid floor_area_sqm")
        if r.get('town') not in VALID_TOWNS:
            issues.append("invalid town")
        if r.get('flat_type') not in VALID_FLAT_TYPES:
            issues.append("invalid flat_type")
        if r.get('flat_model') not in VALID_FLAT_MODELS:
            issues.append("invalid flat_model")
        if not re.match(VALID_STOREY_FORMAT, str(r.get('storey_range'))):
            issues.append("invalid storey_range")
        if issues:
            row_copy = r.copy()
            row_copy['comments'] = "; ".join(issues)
            rows.append(row_copy)
    return pd.DataFrame(rows)

# --------------------
# PROFILING
# --------------------
def profile_dataset(df):
    profile = {}
    profile["total_rows"] = len(df)
    profile["total_columns"] = len(df.columns)
    profile.update({f"null_count_{col}": df[col].isna().sum() for col in df.columns})
    numeric_cols = ["resale_price","floor_area_sqm"]
    for col in numeric_cols:
        if col in df.columns:
            profile[f"{col}_min"] = df[col].min()
            profile[f"{col}_max"] = df[col].max()
            profile[f"{col}_mean"] = df[col].mean()
            profile[f"{col}_median"] = df[col].median()
    profile["duplicate_rows"] = df.duplicated().sum()
    return profile

# --------------------
# TRANSFORMATION
# --------------------
def create_resale_identifier(df):
    df_copy = df.copy()
    if 'block' not in df_copy.columns:
        df_copy['block'] = '000'
    df_copy['block_numeric'] = (
        df_copy['block']
        .astype(str)
        .str.extract(r'(\d+)')[0]  # take first column from extract result
        .fillna('000')
        .str.zfill(3)
    )
    df_copy['year_month'] = df_copy['month'].dt.to_period('M')
    avg_price = df_copy.groupby(['year_month','town','flat_type'])['resale_price'].transform('mean')
    df_copy['Resale Identifier'] = (
            "S" +
            df_copy['block_numeric'] +
            avg_price.astype(int).astype(str).str[:2] +
            df_copy['month'].dt.month.astype(str).str.zfill(2) +
            df_copy['town'].str[0]
    )
    return df_copy

def hash_resale_identifier(df):
    df['Resale Identifier Hashed'] = df['Resale Identifier'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    return df

# --------------------
# ETL FUNCTION
# --------------------
def run_hdb_etl():
    print("="*60)
    print("🚀 HDB RESALE ETL PIPELINE")
    print("="*60)

    processing_download_hdb_data()

    for d in [RAW_OUT_DIR, CLEANED_OUT_DIR, TRANSFORM_OUT_DIR, HASHED_OUT_DIR, FAILED_OUT_DIR, AUDIT_OUT_DIR, PROFILE_OUT_DIR]:
        os.makedirs(d, exist_ok=True)

    if not check_snapshots_exist():
        sys.exit(1)

    # --------------------
    # Load & Raw Save
    # --------------------
    df = timed_step("Load & Align CSVs", load_and_align_snapshots)
    raw_file = os.path.join(RAW_OUT_DIR, "hdb_resale_raw.csv")
    df.to_csv(raw_file, index=False)
    print(f"💾 Raw dataset saved: {raw_file}")

    # --------------------
    # Cleaning
    # --------------------
    df['month'] = pd.to_datetime(df['month'], format="%Y-%m", errors='coerce')
    df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
    df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
    df = df.dropna(subset=['month','resale_price','floor_area_sqm'])
    df = recompute_remaining_lease(df)

    # Deduplication
    df_cleaned, df_duplicates = deduplicate_dataset(df)
    df_duplicates_file = os.path.join(AUDIT_OUT_DIR, "duplicates.csv")
    if not df_duplicates.empty:
        df_duplicates.to_csv(df_duplicates_file, index=False)
        print(f"⚠️ Duplicate rows saved for audit: {df_duplicates_file}")

    # Extra validation
    df_rules_fail = extra_validation(df_cleaned)
    df_rules_file = os.path.join(AUDIT_OUT_DIR, "rule_violations.csv")
    if not df_rules_fail.empty:
        df_rules_fail.to_csv(df_rules_file, index=False)
        print(f"⚠️ Rule violation rows saved for audit: {df_rules_file}")

    # Anomalies
    df_anomalies = detect_anomalous_prices(df_cleaned)
    df_anomalies_file = os.path.join(AUDIT_OUT_DIR, "anomalies.csv")
    if not df_anomalies.empty:
        df_anomalies.to_csv(df_anomalies_file, index=False)
        print(f"⚠️ Anomalous rows saved for audit: {df_anomalies_file}")

    # Combine all failed
    failed_records = pd.concat([df_duplicates, df_rules_fail, df_anomalies]).drop_duplicates()
    if not failed_records.empty:
        failed_file = os.path.join(FAILED_OUT_DIR, "hdb_resale_failed.csv")
        failed_records.to_csv(failed_file, index=False)
        print(f"⚠️ Failed records saved: {failed_file}")

    # Remove failed from cleaned
    failed_index = failed_records.index
    df_cleaned_final = df_cleaned.loc[~df_cleaned.index.isin(failed_index)]
    cleaned_file = os.path.join(CLEANED_OUT_DIR, "hdb_resale_cleaned.csv")
    df_cleaned_final.to_csv(cleaned_file, index=False)
    print(f"💾 Cleaned dataset saved: {cleaned_file}")

    # Profiling
    profile = timed_step("Profile Cleaned Dataset", profile_dataset, df_cleaned_final)
    profile_file = os.path.join(PROFILE_OUT_DIR, "profile_cleaned.csv")
    pd.DataFrame([profile]).to_csv(profile_file, index=False)
    print(f"📊 Profiling report saved: {profile_file}")

    # Transformation
    df_transformed = create_resale_identifier(df_cleaned_final)
    transformed_file = os.path.join(TRANSFORM_OUT_DIR, "hdb_resale_transformed.csv")
    df_transformed.to_csv(transformed_file, index=False)
    print(f"💾 Transformed dataset saved: {transformed_file}")

    # Hashing
    df_hashed = hash_resale_identifier(df_transformed)
    hashed_file = os.path.join(HASHED_OUT_DIR, "hdb_resale_hashed.csv")
    df_hashed.to_csv(hashed_file, index=False)
    print(f"💾 Hashed dataset saved: {hashed_file}")

    print("\n✅ ETL COMPLETE")
    print("="*60)
    print(f"Rows: {len(df_hashed):,}, Columns: {len(df_hashed.columns)}")
    print(f"Date range: {df_hashed['month'].min().date()} → {df_hashed['month'].max().date()}")

# --------------------
# ENTRY POINT
# --------------------
if __name__ == "__main__":
    run_hdb_etl()
