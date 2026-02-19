# Git Repository Setup Guide
## HDB Resale ETL Pipeline

---

## Files to Commit to Git

These are the **only files** that go into the repository:

```
hdb-etl/
├── README.md                  ← Project documentation
├── requirements.txt           ← Python dependencies
├── .gitignore                 ← Tells Git what to exclude
├── download_hdb_data.py       ← Downloader module
└── HDB_ETL_Jupyter.ipynb      ← Main ETL notebook
```

> Do NOT commit `hdb_data/` or `output/` — these are in `.gitignore`
> and will be generated automatically when the notebook is run.

---

## Step-by-Step: First Time Setup

### Step 1 — Install Git
Download from https://git-scm.com and install.
Verify installation:
```bash
git --version
```

### Step 2 — Create a GitHub Account
Go to https://github.com and sign up (if you don't have one).

### Step 3 — Create a New Repository on GitHub
1. Click the **+** button (top right) → **New repository**
2. Fill in:
   - **Repository name:** `hdb-etl`
   - **Description:** `HDB Resale Flat Prices ETL Pipeline`
   - **Visibility:** Public or Private
   - ❌ Do NOT tick "Add a README" (we have our own)
3. Click **Create repository**
4. Copy the repository URL shown (e.g. `https://github.com/your-username/hdb-etl.git`)

### Step 4 — Set Up Your Local Project Folder

Open terminal / Command Prompt and navigate to your project folder:
```bash
cd path\to\your\project        # Windows
cd path/to/your/project        # Mac / Linux
```

Your folder should contain these files:
```
download_hdb_data.py
HDB_ETL_Jupyter.ipynb
README.md
requirements.txt
.gitignore
```

### Step 5 — Initialise Git Locally
```bash
git init
```

### Step 6 — Connect to GitHub
```bash
git remote add origin https://github.com/your-username/hdb-etl.git
```

### Step 7 — Stage All Files
```bash
git add README.md
git add requirements.txt
git add .gitignore
git add download_hdb_data.py
git add HDB_ETL_Jupyter.ipynb
```

Or add everything at once (`.gitignore` will automatically exclude data/output folders):
```bash
git add .
```

Verify what will be committed:
```bash
git status
```

You should see only the 5 project files listed — NOT `hdb_data/` or `output/`.

### Step 8 — Commit
```bash
git commit -m "Initial commit: HDB Resale ETL pipeline"
```

### Step 9 — Push to GitHub
```bash
git branch -M main
git push -u origin main
```

Enter your GitHub username and password/token when prompted.

> **Note:** GitHub no longer accepts passwords — you need a Personal Access Token.
> Generate one at: GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)

---

## Day-to-Day Workflow (Making Changes)

After making any changes to your files:

```bash
# 1. Check what changed
git status

# 2. Stage changes
git add .

# 3. Commit with a message describing what changed
git commit -m "Update anomaly detection threshold"

# 4. Push to GitHub
git push
```

---

## How Another Person Clones and Runs It

Share your GitHub URL with them. They run:

```bash
# 1. Clone the repo
git clone https://github.com/your-username/hdb-etl.git
cd hdb-etl

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook

# 5. Open HDB_ETL_Jupyter.ipynb
# 6. Click: Kernel → Restart & Run All
```

That's it — the notebook downloads the data and runs the full pipeline automatically.

---

## Quick Command Reference

| Task | Command |
|---|---|
| Check status | `git status` |
| Stage all changes | `git add .` |
| Commit | `git commit -m "your message"` |
| Push to GitHub | `git push` |
| Pull latest changes | `git pull` |
| View commit history | `git log --oneline` |
| Undo last commit (keep files) | `git reset --soft HEAD~1` |
