# HDB Resale Flat Prices — ETL Pipeline

> **Data Source:** [data.gov.sg — Collection 189](https://data.gov.sg/collections/189/view)
> **Coverage:** Mar 2012 – Dec 2016 (Registration Date)

---

## What This Project Does

This repository contains a full Extract, Transform, Load (ETL) pipeline for HDB resale flat prices. It automatically downloads raw data from data.gov.sg, cleans and validates it, detects anomalies, and produces structured output files ready for analysis.

---

## Repository Structure

```
hdb-etl/
├── README.md                        ← You are here
├── requirements.txt                 ← Python dependencies
├── download_hdb_data.py             ← Module: downloads CSV files from data.gov.sg
├── HDB_ETL_Jupyter.ipynb            ← Main notebook: full ETL pipeline
└── output/                          ← Auto-created when notebook is run
    ├── raw/                         ← hdb_resale_raw.csv
    ├── cleaned/                     ← hdb_resale_cleaned.csv
    ├── transformed/                 ← hdb_resale_transformed.csv
    ├── hashed/                      ← hdb_resale_hashed.csv
    ├── failed/                      ← hdb_resale_failed.csv
    ├── audit/                       ← duplicates.csv, rule_violations.csv, anomalies.csv
    └── profiling/                   ← profile_cleaned.csv
```

> **Note:** The `hdb_data/` folder and `output/` folder are excluded from Git (see `.gitignore`).
> They are generated automatically when you run the notebook.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.8 or higher | `python --version` to check |
| pip | Any recent version | Comes with Python |
| Jupyter | Any recent version | `pip install notebook` |
| Internet access | Required | To download data from data.gov.sg |

---

## Quick Start (Step by Step)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/hdb-etl.git
cd hdb-etl
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Create environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — Mac / Linux
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Launch Jupyter Notebook

```bash
jupyter notebook
```

This opens your browser. Click on **`HDB_ETL_Jupyter.ipynb`** to open it.

### Step 5 — Run the Notebook

In Jupyter, go to the menu bar and click:

```
Kernel → Restart & Run All
```

Or run cells one by one using **Shift + Enter**.

> The notebook will automatically download the CSV files, run all ETL stages,
> and save all output files to the `output/` folder.

---

## Pipeline Stages

| Stage | Cell | What Happens |
|---|---|---|
| **0 — Download** | Cell 1 | Fetches 2 CSV files from data.gov.sg API |
| **1 — Imports** | Cells 2–4 | Loads libraries, sets up directories and validation rules |
| **2 — Load** | Cell 5 | Reads and merges both CSVs into one DataFrame |
| **3 — Clean** | Cells 6–7 | Type casting, null removal, lease recomputation |
| **4 — Deduplicate** | Cell 8 | Removes duplicate records, saves audit file |
| **5 — Validate** | Cell 9 | Applies 6 business rules, captures violations |
| **6 — Anomaly Detection** | Cell 10 | 3-sigma outlier detection per town and flat type |
| **7 — Profile** | Cell 11 | Statistical summary of cleaned dataset |
| **8 — Transform** | Cell 12 | Creates synthetic Resale Identifier field |
| **9 — Hash** | Cell 13 | SHA-256 hashes the Resale Identifier |
| **Summary** | Cell 14 | Final row count, date range, output file checklist |

---

## Output Files

| File | Location | Description |
|---|---|---|
| `hdb_resale_raw.csv` | `output/raw/` | Merged raw data before any cleaning |
| `hdb_resale_cleaned.csv` | `output/cleaned/` | Clean data with all failed records removed |
| `hdb_resale_transformed.csv` | `output/transformed/` | Cleaned data with Resale Identifier added |
| `hdb_resale_hashed.csv` | `output/hashed/` | Final output with SHA-256 hashed identifier |
| `hdb_resale_failed.csv` | `output/failed/` | All records that failed any check |
| `duplicates.csv` | `output/audit/` | Duplicate records removed during deduplication |
| `rule_violations.csv` | `output/audit/` | Records failing business rule validation |
| `anomalies.csv` | `output/audit/` | Price outliers detected by 3-sigma method |
| `profile_cleaned.csv` | `output/profiling/` | Statistical profile of cleaned dataset |

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: download_hdb_data.py` | File not in same folder as notebook | Ensure both files are in the same directory |
| Download cell shows no output | `if __name__` guard missing in `.py` file | Use the version from this repo — do not modify the bottom of `download_hdb_data.py` |
| `429 Too Many Requests` | data.gov.sg API rate limit | Wait 5 minutes and re-run the download cell only |
| `ConnectionError / getaddrinfo failed` | VPN or firewall blocking the API | Disconnect VPN and retry |
| Files missing after download | Filename mismatch | Check `hdb_data/` folder — filenames must match exactly what is in `CSV_FILES` in the notebook |
| Notebook runs but output is empty | Kernel not restarted before run | Go to `Kernel → Restart & Run All` |

---

## Dependencies

Listed in `requirements.txt`:

```
pandas
requests
notebook
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Notes

- **No data files are committed to this repository.** All CSVs are downloaded at runtime from data.gov.sg.
- **Internet access is required** for Stage 0 (download). All other stages work offline once files are downloaded.
- **Re-running the notebook** will overwrite all output files — this is safe and expected.
- Data is licensed under the [Singapore Open Data Licence](https://data.gov.sg/open-data-licence).

---

*HDB Resale ETL Pipeline — data.gov.sg Collection 189*
