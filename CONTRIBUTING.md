# Contributing to GENOME

Thanks for your interest. GENOME is Apache-2.0 and contributions are welcome.

## Ground rules

- **Benchmarks must stay honest.** Any claim in the README or `benchmarks/RESULTS.md`
  must trace to a runnable script and captured output. PRs that add or change claims
  without reproducible evidence will not be merged. Nulls are publishable results here.
- **The default write path stays LLM-free and local.** Features that add an LLM, API
  call, or network dependency to `Memory.add()`'s default path will be declined; make
  them opt-in instead (see `llm_call=` and the belief-state layer for the pattern).
- Match the existing code style; keep solutions simple.

## Workflow

1. Fork, branch from `main`.
2. `pip install -e ".[dev]"`
3. Make your change, with tests.
4. `ruff check genome tests` and `pytest tests/` must both pass (CI enforces this).
5. Open a PR with a clear description of what changed and why.

## Running the benchmarks

Benchmark datasets are not bundled (licenses) — see `benchmarks/data/README.md`.
The no-key, no-dataset entry points are `benchmarks/local_writepath.py` and
`benchmarks/tco_project.py`.

## Questions

Open an issue, or email info@northtek.io.
