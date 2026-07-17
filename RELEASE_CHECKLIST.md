# Release checklist — genome-memory 1.0.1

> Staged locally the night of 2026-07-16. **Nothing was pushed, built, or published.**
> No outbound calls were made. This is a review-and-publish runbook for the morning.

## What's in 1.0.1

- `genome-verify` / `python -m genome.verify` — one-command local self-proof (no key).
- `benchmarks/head_to_head.py` — reproduce GENOME-vs-Mem0 accuracy parity with your own
  key (same responder/judge/embedder/top-k; paired McNemar). Offline `--smoke` included.
- README: "Prove it yourself" + "Run it as an HTTP API" sections.
- Security: proxy-header denial on the keyless opt-in; `GENOME_REQUIRE_SCOPE=1` for
  multi-tenant isolation; docker-compose Postgres loopback-only + required secrets.
- Version bumped to 1.0.1 in `pyproject.toml` and `genome/__init__.py`; CHANGELOG updated.

## Verified tonight (local, offline — evidence captured in the session)

- [x] `python -m genome.verify` → all PASS, exit 0 (air-gapped write, 0 network, 7.1 ms/msg).
- [x] `benchmarks/head_to_head.py --smoke` → pipeline runs end-to-end, exit 0.
- [x] Head-to-head real-dataset parse + no-key fail-fast (no spend, no outbound).
- [x] README HTTP curl examples run live (201 add / 200 search / 503 on proxied keyless).
- [x] Full suite green locally (516 + new verify/head-to-head tests); ruff clean.
- [x] Two adversarial security re-audits: bypass_closed=true, no must-fix.

## MUST DO before publishing (needs network + a key — not done tonight)

- [ ] **Run the real head-to-head once** to confirm it produces a parity number end to
      end (the only path not exercisable offline):
      ```
      export OPENAI_API_KEY=sk-...
      # LoCoMo (CC BY-NC 4.0) already present at benchmarks/data/locomo10.json;
      # if not: curl -L -o benchmarks/data/locomo10.json \
      #   https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
      pip install mem0ai
      python benchmarks/head_to_head.py --n 2 --q 10        # small, cheap sanity run
      ```
      Expect: both systems print a headline accuracy and a McNemar `VERDICT:` line. A
      small run is noisy; confirm it *runs and pairs*, not the exact number.

## Publish steps (morning)

1. [ ] Review the staged, **unpushed** commits: `git log --oneline origin/main..HEAD`.
2. [ ] `python -m ruff check genome tests` → clean.
3. [ ] `.venv/Scripts/python.exe -m pytest -q` → green (3 Postgres skips expected).
4. [ ] `git push origin main` (held overnight).
5. [ ] Confirm CI green on the pushed HEAD (`gh run watch`).
6. [ ] Build: `python -m build` (produces `dist/genome_memory-1.0.1*`).
7. [ ] `twine check dist/genome_memory-1.0.1*`.
8. [ ] `twine upload dist/genome_memory-1.0.1*` — uses the existing PyPI token
       (**not rotated**, per your instruction). Confirm you intend to publish.
9. [ ] Tag: `git tag -a v1.0.1 -m "genome-memory 1.0.1" && git push origin v1.0.1`.
       (Verify the tag lands on the clean HEAD, not a stale one — see the prior tag
       incident; `git merge-base --is-ancestor v1.0.1 origin/main`.)

## Post-publish smoke (fresh venv)

10. [ ] `pip install genome-memory==1.0.1` in a clean venv, then:
        - `python -m genome.verify` → all PASS.
        - `genome-verify`, `genome-mcp` console scripts exist on PATH.
        - `python -c "import genome; print(genome.__version__)"` → `1.0.1`.

## Notes

- Old `Frostbyte-Devs/genome` deletion still pending (needs
  `gh auth refresh -h github.com -s delete_repo`, your action).
- PyPI token intentionally not rotated.
