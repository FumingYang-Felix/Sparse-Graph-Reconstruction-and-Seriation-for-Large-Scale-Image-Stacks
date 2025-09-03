# GCD Seriation Toolkit — Graph‑Condensation‑Densification + CSV Clean + Chain Stitch

A compact, end‑to‑end toolkit for 1‑D **section seriation** built around three scripts:

1. **`Graph-Condensation-Densification.py`** — the main seriation engine (GCD).  
2. **`clean_csv.py`** — quick cleaner that filters pairwise results and adds a `score`.  
3. **`chain_stitch.py`** — builds chains from best/second‑best neighbors and links them into a global order.

This README explains prerequisites, how to run the pipeline step‑by‑step, expected inputs/outputs, practical tips, and benchmarks you can try.

---

## Table of Contents
- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data & File Conventions](#data--file-conventions)
- [Quick Start (3 Steps)](#quick-start-3-steps)
  - [Step 1 — Run GCD](#step-1--run-gcd)
  - [Step 2 — Clean the CSV](#step-2--clean-the-csv)
  - [Step 3 — Build & Link Chains](#step-3--build--link-chains)
- [Script Details](#script-details)
  - [`Graph-Condensation-Densification.py`](#graph-condensation-densificationpy)
  - [`clean_csv.py`](#cleancsvpy)
  - [`chain_stitch.py`](#chain_stitchpy)
- [Benchmarks](#benchmarks)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Citing & License](#citing--license)
- [Contact](#contact)

---

## Overview

**GCD (Graph‑Condensation‑Densification)** gives a fast, sub‑quadratic pipeline to recover a linear order of sections using image‑based pairwise alignment as an oracle. The flow is:

1) Build a **short‑edge Borůvka tree** with strict biological plausibility checks.  
2) **Condense** and **densify** connectivity locally (K‑window).  
3) Produce a robust linear order (BFS double‑sweep; optional **spectral** refinement).  
4) Verify the final order with a last **K‑densification check**.

Then we post‑process pairwise results:

- `clean_csv.py` filters invalid rows and adds a quality **score = ssim × num_inliers`.  
- `chain_stitch.py` uses **best/second‑best** neighbors to create non‑overlapping chains and link all super‑chains into a global order.

---

## Repository Layout

```
.
├── Graph-Condensation-Densification.py   # main GCD algorithm
├── clean_csv.py                           # CSV cleaner / scorer
├── chain_stitch.py                        # chain builder & linker
└── README.md
```

> Your repo may also contain an `assets/` or `results/` folder created at runtime.

---

## Requirements

- **Python** ≥ 3.9 (3.10+ recommended)
- Python packages:
  - `numpy`, `pandas`, `opencv-python`, `scikit-image`, `networkx`, `matplotlib`, `scipy`
  - (standard library modules are already used: `argparse`, `glob`, `pickle`, etc.)

Install everything with:

```bash
pip install -U numpy pandas opencv-python scikit-image networkx matplotlib scipy
```

> On macOS with Apple Silicon, prefer a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`.

---

## Data & File Conventions

### Section image naming

The GCD script auto-discovers sections from a folder that matches patterns like:

- `w7_png_4k/section_<num>_r01_c01.png` (e.g., `section_123_r01_c01.png`)
- `S_<num>` (optionally with suffixes), e.g., `S_01094_inverted_cropped`

Regular expressions in the scripts already recognize both formats. If your naming differs, update `NAME_PAT` in `chain_stitch.py` and the discovery logic in `Graph-Condensation-Densification.py`.

### Pairwise CSV columns

Some steps consume a CSV with **pairwise alignment** fields such as:

- `fixed`, `moving`, `ssim`, `scale`, `num_inliers` (others are preserved).  
- After cleaning, a **`score`** column is present: `score = ssim * num_inliers` (if `num_inliers` exists).

### Output directories

The GCD script creates run‑specific output folders (e.g., `comprehensive_all_<N>_sections_build_order_cached_fastflann_spectral/`) and stores:

- `final_order.csv` — the final linear order  
- `comprehensive_pairwise_results.csv` — all computed pairs and stats  
- `spanning_tree_edges.csv` — Borůvka tree edges (if enabled)  
- `build_order_statistics.txt` — detailed per‑phase stats  
- optional visualizations (PNG)

---

## Quick Start (3 Steps)

### Step 1 — Run GCD

Recommended, fast & reproducible run:

```bash
python Graph-Condensation-Densification.py   --build-and-order   --cache-sift   --fast-flann   --verbose
```

Alternative examples:

```bash
# Baseline with custom parameters
python Graph-Condensation-Densification.py --build-and-order --samples-per-phase 7 --window-k 7

# Auto-tune SSIM threshold to target ~10 neighbors/node
python Graph-Condensation-Densification.py --build-and-order --auto-tune-threshold --verbose

# Two-phase (relaxed → strict) with proximity mask
python Graph-Condensation-Densification.py --build-and-order --two-phase --proximity-window 15

# Precompute SIFT for maximal speed on repeated runs
python Graph-Condensation-Densification.py --build-and-order --precompute-sift --verbose
```

> **Tip:** On reruns, `--cache-sift` (and optionally `--precompute-sift`) gives dramatic speed‑ups.

---

### Step 2 — Clean the CSV

Assuming your raw pairwise CSV is `results/sequencing/pairs.csv` (or a CSV produced by the GCD step):

```bash
python clean_csv.py results/sequencing/pairs.csv
# or choose an explicit output path:
python clean_csv.py results/sequencing/pairs.csv -o results/sequencing/cleaned_csv/pairs_cleaned.csv
```

What the cleaner does:

1. Drops rows with `ssim == -1`.  
2. Keeps only `0.9 ≤ scale ≤ 1.1`.  
3. Adds `score = ssim * num_inliers` (when `num_inliers` exists).

By default the cleaned file is written to:
```
<project_root>/results/sequencing/cleaned_csv/<input_basename>_cleaned.csv
```

---

### Step 3 — Build & Link Chains

Use the **cleaned** CSV (must have `fixed,moving,score` headers):

```bash
python chain_stitch.py   --csv results/sequencing/cleaned_csv/pairs_cleaned.csv   --output results/sequencing/best_pair_chains_graph.txt
```

The script prints a five‑part report and saves the same content to the output text file:

1. **BEST pair** per section  
2. Chain grouping (undirected graph)  
3. **SECOND‑BEST** pair per section  
4. Stitch merges (using second‑best hints, endpoint‑only concatenations)  
5. Global linking by highest remaining endpoint scores

If all super‑chains cannot be connected, it prints the residual final chains.

---

## Script Details

### `Graph-Condensation-Densification.py`

**Algorithm sketch**  
- **Phase A** — Short‑edge **Borůvka** tree (O(N log N)).  
- **Phase B** — Farthest‑point condensation (3 rounds).  
- **Phase C** — Double‑sweep BFS for a rough order.  
- **Phase D** — **K‑window** densification (local edges).  
- **Phase E** — Optional **spectral** ordering with the Fiedler vector.  
- **Phase F** — Final K‑densification verification.

**Biological plausibility checks (always on):**
- Rotation within ±90°.  
- Scale within ~0.95–1.05 (configurable).  
- Inlier ratio > threshold (default 8%).

**Good defaults**
```bash
python Graph-Condensation-Densification.py --build-and-order --cache-sift --fast-flann --verbose
```

**Performance notes**
- SIFT caching: ~orders‑of‑magnitude fewer feature detections on reruns.  
- Fast FLANN: 2–3× faster matching with minimal accuracy impact.  
- Overall complexity: **O(N (log N + K))**, far below exhaustive O(N²).

Outputs include `final_order.csv`, `comprehensive_pairwise_results.csv`, `build_order_statistics.txt`, and optionally plots.

---

### `clean_csv.py`

**Purpose**: sanitize raw pairwise CSV and add a composite `score`.

**Behavior**
- Remove `ssim == -1`.  
- Keep `scale` near 1: `0.9–1.1`.  
- If `num_inliers` exists → `score = ssim × num_inliers`.
- Preserves other columns unchanged; prints summary stats.

**Usage**
```bash
python clean_csv.py path/to/pairs.csv
python clean_csv.py path/to/pairs.csv -o path/to/cleaned.csv
```

**Output (default)**: `results/sequencing/cleaned_csv/<basename>_cleaned.csv`.

---

### `chain_stitch.py`

**Purpose**: derive chains from best/second‑best pairs and link them into a global order.

**Input CSV headers (required)**: `fixed,moving,score`  
If you have both `A→B` and `B→A`, the loader keeps the **max score per unordered pair**.

**Pipeline**
1. BEST neighbor per section.  
2. Undirected chain grouping (non‑overlapping; each section appears once).  
3. SECOND‑BEST neighbor per section.  
4. Endpoint‑only chain stitching guided by second‑best pairs.  
5. Global linking using the highest endpoint scores from the CSV.

**Usage**
```bash
python chain_stitch.py --csv new_pairwise_filtered.csv --output best_pair_chains_graph.txt
```

---

## Benchmarks

Two ready‑to‑run benchmark bundles are prepared:

1. **H01 dataset (33 nm sections)** — subsets of **100 / 500 / 1000** sections.  
2. **HI‑MC dataset (250 nm sections)** — subset of **100** sections.

**How to run**

- Place images so the GCD script can discover them (e.g., `w7_png_4k/section_<num>_r01_c01.png`).  
- Run the **Quick Start** steps above on each subset directory.  
- Compare `final_order.csv`, chain reports, and per‑phase statistics.

> If your benchmark folders have a different structure, either symlink them or adjust the discovery path/pattern in `Graph-Condensation-Densification.py`.

---

## Performance Tips

- **Cache features**: `--cache-sift` for every serious run; `--precompute-sift` before batch experiments.  
- **Speed up matching**: `--fast-flann`.  
- **Downstream speed**: clean your CSV first; keep `score` numeric.  
- **When order is stable**: you can **disable spectral** phase to save time.  
- **Two‑phase** (`--two-phase`) reduces long‑range tests by focusing on plausible local connections later.

---

## Troubleshooting

- **`KeyError: 'fixed'/'moving'/'score'` in `chain_stitch.py`**  
  CSV headers must *exactly* be `fixed,moving,score`. Use `clean_csv.py` to produce a conforming CSV.

- **`ValueError` parsing scores**  
  Ensure the `score` column is numeric (no strings/NA).

- **Names not detected / missing**  
  If your IDs don’t match `NAME_PAT`, update the regex at the top of `chain_stitch.py`.

- **Outliers that won’t place**  
  Common causes are incorrect ROI placement or rotation/scale far outside biological limits. Fix ROIs and re‑export the pairs.

- **Large CSVs are slow**  
  Pre‑deduplicate unordered pairs (keep the max score). The loader also keeps only the maximum per unordered pair.

---

## FAQ

**Q: Do I need both `section_<n>_r01_c01` and `S_<n>` styles?**  
A: No. Either is fine; both are recognized. If you use something else, update the regex.

**Q: Can I reproduce results?**  
A: Yes. The main script sets seeds where appropriate; results are deterministic given the same inputs and flags.

**Q: Where does the final order live?**  
A: See the run output folder (created by GCD). The file is `final_order.csv`.

---

## Citing & License

If you use this toolkit in academic work, please cite this repository (author: **Fuming Yang**) and the GCD seriation pipeline.

**License:** MIT (unless you choose otherwise). Add a `LICENSE` file to override.

---

## Contact

Maintainer: **Fuming Yang** (`FumingYang-Felix`)  
Issues and feature requests are welcome via GitHub Issues.

_Last updated: 2025-09-03_
