# Benchmark datasets

GENOME's benchmarks run against third-party datasets that are **not redistributed
in this repository** because they carry their own licenses. Download them yourself
(each is free) and place them here. The `benchmarks/data/` directory is gitignored.

## LoCoMo

- Source: https://github.com/snap-research/locomo
- License: **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)** — Snap Inc.
- Non-commercial use only (evaluation and research are permitted; do not use the
  dataset itself for a commercial purpose or redistribute it under different terms).
- Place the file at: `benchmarks/data/locomo10.json`

```bash
mkdir -p benchmarks/data
curl -L -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

## LongMemEval

- Source: https://github.com/xiaowu0162/LongMemEval (dataset on Hugging Face)
- License: see the upstream repository / dataset card for terms.
- Place the small variant at: `benchmarks/data/lme/longmemeval_s`

## TempBelief (generated locally, no download)

TempBelief is GENOME's own synthetic bi-temporal dataset — generate it on demand:

```bash
python genome/evals/tempbelief.py 12 > benchmarks/data/tempbelief.json
```

---

Attribution note: LoCoMo is © its authors (Maharana et al., "Evaluating Very
Long-Term Conversational Memory of LLM Agents"), used here under CC BY-NC 4.0 for
non-commercial evaluation. GENOME's own source is licensed separately (Apache-2.0);
the dataset licenses are independent of GENOME's license and are not altered by it.
