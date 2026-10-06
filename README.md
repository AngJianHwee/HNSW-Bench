# HNSW Benchmark on SIFT1M

> **Evaluating HNSW Approximate Nearest Neighbor Search at Scale**

This project benchmarks **HNSW** (Hierarchical Navigable Small World) on the standard **SIFT1M** dataset (1M vectors, 128 dimensions) using **Faiss**.

## 🎯 Project Goals

- Evaluate **Recall@10 vs QPS** trade-offs for HNSW
- Measure **index built time** and **search time** between different configurations
- Provide reproducible benchmarks with parameter sweeps via `config.yaml`

---

## Plot Results

<img width="989" height="1039" alt="image" src="https://github.com/user-attachments/assets/d34ba3d8-c9fe-495a-b1c4-7d67e1b15221" />

<img width="1002" height="881" alt="image" src="https://github.com/user-attachments/assets/e66fae89-23c3-4bd3-a8b4-255f6569a7db" />

<img width="962" height="1665" alt="image" src="https://github.com/user-attachments/assets/c86b66b1-8efb-4c28-b5ba-7cb8ea3bfc7c" />


---

## 📊 Dataset

| Component | File | Size |
|-----------|------|------|
| Base vectors | `sift_base.fvecs` | 1,000,000 × 128 (float32) |
| Query vectors | `sift_query.fvecs` | 10,000 × 128 (float32) |
| Training vectors | `sift_learn.fvecs` | 100,000 × 128 (float32) |
| Ground truth | `sift_groundtruth.ivecs` | 10,000 × 100 (int32) |

**Source:** [TEXMEX Corpus](http://corpus-texmex.irisa.fr/) — SIFT1M

---

## ⚙️ Experimental Setup

### Grid Search Parameters (HNSW)

| Parameter | Values | Description |
|-----------|--------|-------------|
| **M** | `[16, 32, 64]` | Max connections per node (higher = better recall, more memory) |
| **ef_construction** | `[64, 128, 256, 512]` | Search width during index build (higher = better quality, slower build) |
| **ef_search** | `[16, 64, 128, 256]` | Search width at query time (higher = better recall, slower queries) |

### Fixed Parameters

| Parameter | Value |
|-----------|-------|
| **k (top-k)** | 10 |
| **Dataset** | SIFT1M (128-dim) |

### Metrics Collected

- **Recall@10** — Fraction of true top-10 neighbors recovered
- **QPS** — Queries per second (queries / total search time)
- **Build Time** — Index construction time (seconds)
- **Index Memory** — Memory footprint of the index structure (via Faiss serialization)

---

## 📈 Results

### Generated Visualizations

| Plot | Description |
|------|-------------|
| `recall_vs_qps.png` | **QPS vs Recall@10** — Subplots per M, color by M, marker by ef_construction. Shows Pareto frontier. |
| `memory_3d_bar.png` | **3D Bar Plot** — Recall@10 by ef_construction (x) and ef_search (y), grouped by M (color). Height = Recall. |
| `recall_vs_efsearch.png` | **Recall vs Index Memory** — One row per ef_construction. Lines connect same ef_search; markers differ by M. |

### Output Files

| File | Description |
|------|-------------|
| `results_hnsw.csv` | Raw HNSW grid search results |
| `results_combined.csv` | Combined results (currently HNSW only) |
| `recall_vs_qps.png` | Plot 1: QPS vs Recall@10 |
| `memory_3d_bar.png` | Plot 2: 3D Recall surface |
| `recall_vs_efsearch.png` | Plot 3: Recall vs Memory by ef_search |

---

## 🏁 Key Conclusions

*To be filled after running experiments. Expected findings:*

- **HNSW** achieves high QPS at high recall (≥90%) with moderate memory usage
- **ef_search** controls the recall-QPS trade-off at query time
- **M** and **ef_construction** control index quality and build time
- Higher **M** increases memory but improves recall ceiling
- Higher **ef_construction** improves graph quality at build-time cost

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Execution (`config.yaml`)

Edit `config.yaml` to select which phases to run:

```yaml
run:
  phase1_download: true      # Download SIFT1M dataset
  phase2_groundtruth: true   # Build Flat index & compute ground truth
  phase3_hnsw: true          # HNSW parameter sweep
  phase4_visualize: true     # Generate plots & combined CSV
```

Set any phase to `false` to skip it.

### 3. Run Pipeline

```bash
# Run all enabled phases from config.yaml
python run_all.py
```

### 4. Data Download Notes

The downloader tries multiple mirrors automatically:
1. **Hugging Face (qbo-odp/sift1m)** - Most reliable, serves uncompressed `.fvecs` directly
2. **GitHub (FAISS repo)** - Raw files from FAISS benchmarks
3. **IRISA FTP** - Official mirror
4. **IRISA HTTP** - Original source

Files from Hugging Face are served uncompressed (no `.gz` extension), while others are gzipped. The downloader auto-detects and extracts if needed.

If all mirrors fail, you can manually download from:
- https://huggingface.co/datasets/qbo-odp/sift1m/tree/main
- https://github.com/facebookresearch/faiss/tree/main/benchs/sift1M
- Place files in `data/` (`.gz` files will be auto-extracted on next run)

### 5. View Results

- **CSV tables:** `results_hnsw.csv`, `results_combined.csv`
- **Plots:** `recall_vs_qps.png`, `memory_3d_bar.png`, `recall_vs_efsearch.png`

---

## 🔧 Configuration (`config.yaml`)

All parameters are configurable via `config.yaml`.

---

## 🔬 Phase Details

### Phase 1: Download Data (`download_data.py`)
- Downloads SIFT1M from multiple mirrors with fallback
- Auto-detects and extracts `.gz` files
- Skips already-downloaded files
- Outputs to `data/` directory

### Phase 2: Ground Truth (`phase2_groundtruth.py`)
- Builds `faiss.IndexFlatL2` (exact brute-force)
- Computes ground truth for all queries at top-k
- Saves ground truth indices (`.ivecs`) and distances (`.fvecs`)
- Provides `recall_at_k_vectorized()` for fast evaluation
- Validates query count matches between ground truth and predictions

### Phase 3: HNSW Sweep (`phase3_hnsw.py`)
- Grid search over M × ef_construction × ef_search
- Builds index once per (M, ef_construction) pair
- Sweeps ef_search on the same index (efficient)
- Measures: build time, search time, QPS, Recall@10, index memory
- Index memory measured via `faiss.serialize_index()`
- Handles query/groundtruth count mismatches gracefully
- Saves results to `results_hnsw.csv`

### Phase 4: Visualization (`phase4_visualize.py`)
- Loads HNSW results CSV
- Generates 3 publication-quality plots:
  1. **QPS vs Recall@10** (subplots per M)
  2. **3D Recall Surface** (ef_construction × ef_search × M)
  3. **Recall vs Memory** (rows per ef_construction)
- Exports combined CSV
- Uses serif fonts, consistent styling matching academic standards

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `faiss-cpu` | ≥1.7.4 | Vector search (HNSW, Flat) |
| `numpy` | ≥1.21.0 | Array operations |
| `psutil` | ≥5.9.0 | Memory measurement |
| `matplotlib` | ≥3.5.0 | Static plots |
| `plotly` | ≥5.10.0 | Interactive plots (optional) |
| `pandas` | ≥1.4.0 | DataFrames, CSV I/O |
| `requests` | ≥2.28.0 | HTTP downloads |
| `tqdm` | ≥4.64.0 | Progress bars |
| `pyyaml` | ≥6.0 | Config parsing |

**For GPU:** Replace `faiss-cpu` with `faiss-gpu` in `requirements.txt`

---

## 📚 References

1. **HNSW:** Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs* (2018)
2. **Faiss Library:** Johnson et al., *Billion-scale similarity search with GPUs* (2019)
3. **SIFT1M:** Jégou et al., *Evaluating Compact Descriptors* (2011)

---

## 📄 License

MIT License — Feel free to use for research or production benchmarking.
