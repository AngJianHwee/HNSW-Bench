# HNSW vs IVF-PQ Benchmark on SIFT1M

> **Comparing Approximate Nearest Neighbor Search Algorithms at Scale**

This project benchmarks two popular ANN algorithms — **HNSW** (Hierarchical Navigable Small World) and **IVF-PQ** (Inverted File with Product Quantization) — on the standard **SIFT1M** dataset (1M vectors, 128 dimensions).

## 🎯 Project Goals

- Evaluate **Recall@10 vs QPS** trade-offs for both algorithms
- Measure **memory efficiency** compared to exact Flat (brute-force) search
- Provide reproducible benchmarks with parameter sweeps
- Visualize Pareto frontiers for algorithm selection guidance

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

### Fixed Parameters

| Algorithm | Parameters |
|-----------|------------|
| **HNSW** | `M=16`, `ef_construction=64` |
| **IVF-PQ** | `nlist=1024`, `m=8`, `nbits=8` |

### Swept Parameters

| Algorithm | Parameter | Values |
|-----------|-----------|--------|
| **HNSW** | `ef_search` | `[8, 16, 32, 64, 128]` |
| **IVF-PQ** | `nprobe` | `[1, 4, 8, 16, 32]` |

### Metrics Collected

- **Recall@10** — Fraction of true top-10 neighbors recovered
- **QPS** — Queries per second (10,000 queries / total search time)
- **Peak Memory** — RSS memory during search (MB)
- **Build Time** — Index construction + training time (seconds)
- **Index Memory** — Memory footprint of the index structure

---

## 📈 Results

### Recall@10 vs QPS Trade-off

![Recall vs QPS](plot_recall_vs_qps.png)

*Figure 1: Pareto frontier comparison. Higher and to the right is better.*

### Memory Usage Comparison

![Memory Comparison](plot_memory.png)

*Figure 2: Index memory footprint. Flat index uses ~512 MB for 1M × 128 float32 vectors.*

### Build Time Comparison

![Build Time](plot_build_time.png)

*Figure 3: Index construction time.*

---

## 📋 Results Summary

### HNSW (M=16, ef_construction=64)

| ef_search | Recall@10 | QPS | Peak Memory (MB) |
|-----------|-----------|-----|------------------|
| 8         | —         | —   | —                |
| 16        | —         | —   | —                |
| 32        | —         | —   | —                |
| 64        | —         | —   | —                |
| 128       | —         | —   | —                |

### IVF-PQ (nlist=1024, m=8, nbits=8)

| nprobe | Recall@10 | QPS | Peak Memory (MB) |
|--------|-----------|-----|------------------|
| 1      | —         | —   | —                |
| 4      | —         | —   | —                |
| 8      | —         | —   | —                |
| 16     | —         | —   | —                |
| 32     | —         | —   | —                |

> **Note:** Run the benchmark to populate actual results. See [Quick Start](#-quick-start).

---

## 🏁 Key Conclusions

*To be filled after running experiments. Expected findings:*

- **HNSW** typically achieves higher QPS at high recall (≥90%) but uses more memory
- **IVF-PQ** offers better memory compression (8× smaller than float32) with competitive speed at moderate recall
- **Pareto frontier**: HNSW dominates at high recall; IVF-PQ wins on memory-constrained scenarios

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Execution (config.yaml)

Edit `config.yaml` to select which phases to run:

```yaml
run:
  phase1_download: true      # Download SIFT1M dataset
  phase2_groundtruth: true   # Build Flat index & compute ground truth
  phase3_hnsw: true          # HNSW parameter sweep
  phase3_ivfpq: true         # IVF-PQ parameter sweep
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

### 4. View Results

- **CSV tables:** `results_hnsw.csv`, `results_ivfpq.csv`, `results_combined.csv`
- **Plots:** `plot_recall_vs_qps.png`, `plot_memory.png`, `plot_build_time.png`

---

## 📁 Project Structure

```
HNSW-IVF-PQ/
├── config.yaml               # Configuration file (NEW)
├── config_loader.py          # Config loader utility (NEW)
├── requirements.txt          # Python dependencies
├── download_data.py          # Phase 1: Download SIFT1M
├── data_utils.py             # .fvecs/.ivecs I/O utilities
├── phase2_groundtruth.py     # Phase 2: Flat index + ground truth
├── phase3_hnsw.py            # Phase 3: HNSW parameter sweep
├── phase3_ivfpq.py           # Phase 3: IVF-PQ parameter sweep
├── phase4_visualize.py       # Phase 4: Plots & combined CSV
├── run_all.py                # Orchestrator script
├── README.md                 # This file
└── data/                     # Dataset (created after download)
    ├── sift_base.fvecs
    ├── sift_query.fvecs
    ├── sift_learn.fvecs
    └── sift_groundtruth.ivecs
```

---

## 🔧 Customization

### Modify Parameters via config.yaml (Recommended)

All parameters are now configurable via `config.yaml` — no code changes needed:

```yaml
# HNSW parameters
hnsw:
  M: 16
  ef_construction: 64
  ef_search_values: [8, 16, 32, 64, 128]
  k: 10

# IVF-PQ parameters
ivfpq:
  nlist: 1024
  m: 8
  nbits: 8
  nprobe_values: [1, 4, 8, 16, 32]
  k: 10

# Data paths
data:
  base_path: "data/sift_base.fvecs"
  query_path: "data/sift_query.fvecs"
  learn_path: "data/sift_learn.fvecs"
  gt_path: "data/groundtruth_flat.ivecs"
```

### Override Parameters via Command Line

You can also override specific parameters by modifying the config programmatically:

```python
from config_loader import get_config

cfg = get_config()
# Access values
print(cfg.hnsw.M)           # 16
print(cfg.hnsw.ef_search_values)  # [8, 16, 32, 64, 128]
print(cfg.ivfpq.nlist)      # 1024
```

### Use GPU (if available)

Replace `faiss-cpu` with `faiss-gpu` in `requirements.txt` and use:
```python
# For HNSW
index = faiss.IndexHNSWFlat(d, M)
# For IVF-PQ - move to GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
```

---

## 📚 References

1. **HNSW:** Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs* (2018)
2. **IVF-PQ:** Jégou et al., *Product Quantization for Nearest Neighbor Search* (2011)
3. **Faiss Library:** Johnson et al., *Billion-scale similarity search with GPUs* (2019)
4. **SIFT1M:** Jégou et al., *Evaluating Compact Descriptors* (2011)

---

## 📄 License

MIT License — Feel free to use for research or production benchmarking.

---

*Generated with ❤️ for ANN algorithm evaluation*