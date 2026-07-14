"""Competitor / reference baselines for the LOCOMO benchmark.

Runs non-GENOME systems through the SAME protocol as genome.evals.locomo:
identical answer prompt, identical retrieval top-k, identical responder +
judge models, identical scoring. That controlled-variable design is what
makes the published comparison defensible -- J scores from different
harnesses/judges are not comparable (see the Zep-vs-Mem0 methodology
dispute).

Systems:
- full-context: the entire conversation transcript in the responder's
  context window. No memory system at all. The "does a memory layer even
  help" reference line -- any credible LoCoMo publication includes it.
- mem0: Mem0 OSS (pip install mem0ai), configured with the same LLM and
  embedder as the GENOME configs. Ingestion mirrors Mem0's own published
  harness (github.com/mem0ai/memory-benchmarks): one add() call per turn,
  speaker_a as the "user" role and speaker_b as "assistant", photo captions
  folded into the text, session timestamps prepended to message content.

The naive dense-RAG baseline is `genome-baseline` in genome.evals.locomo
(raw turns + cosine top-k, every GENOME architectural lever off).

Run:
    python -m genome.evals.baselines --systems full-context,mem0 \
        --output-dir results/locomo
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from genome.evals.locomo import (
    ANSWER_PROMPT,
    LocomoConversation,
    LocomoQuestion,
    LocomoResult,
    PerQuestionResult,
    _judge_one,
    _sanitize_locomo_text,
    load_locomo,
)
from genome.memory.extraction import LLMCallFn


def _format_turn_text(turn) -> str:
    if turn.session_datetime:
        return f"[{turn.session_datetime}] {turn.speaker}: {turn.text}"
    return f"{turn.speaker}: {turn.text}"


class FullContextBaseline:
    """Entire transcript in context. No retrieval, no memory system."""

    name = "baseline-full-context"

    def __init__(self, responder: LLMCallFn, top_k: int = 30) -> None:
        self._responder = responder
        self._transcript = ""

    def ingest(self, conversation: LocomoConversation) -> None:
        self._transcript = "\n".join(
            f"- {_sanitize_locomo_text(_format_turn_text(t))}"
            for t in conversation.turns
        )

    def answer(self, question: LocomoQuestion) -> tuple[str, list[str], float]:
        t0 = time.perf_counter()
        prompt = ANSWER_PROMPT.format(
            context=self._transcript or "(empty conversation)",
            question=_sanitize_locomo_text(question.question),
        )
        predicted = self._responder(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return predicted.strip(), [], latency_ms

    def close(self) -> None:
        self._transcript = ""


class Mem0Baseline:
    """Mem0 OSS through the shared protocol.

    Ingestion mirrors mem0ai/memory-benchmarks (their own LoCoMo harness):
    CHUNK_SIZE=1 -> one add() per turn; role user/assistant mapped from
    speaker_a/speaker_b; blip captions folded in; per-conversation user_id.
    Retrieval + answering use the SAME top-k and answer prompt as every
    other system in the comparison.
    """

    name = "baseline-mem0"

    def __init__(
        self,
        responder: LLMCallFn,
        top_k: int = 30,
        llm_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        llm_provider: str = "openai",
    ) -> None:
        try:
            from mem0 import Memory as _Mem0Memory
        except ImportError as e:
            raise ImportError(
                "The mem0 baseline requires the mem0ai package: "
                "pip install mem0ai  (and OPENAI_API_KEY set)"
            ) from e
        self._responder = responder
        self._top_k = top_k
        # Mem0's INTERNAL fact-extraction LLM must match the shared responder
        # family so the baseline runs on the same LLM budget as every other
        # system (symmetry). The EMBEDDER stays on OpenAI regardless -- it must
        # match GENOME's embedder, and Anthropic offers no embeddings API.
        self._mem = _Mem0Memory.from_config(
            {
                "llm": {
                    "provider": llm_provider,
                    "config": {"model": llm_model, "temperature": 0.0},
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": embed_model},
                },
            }
        )
        self._user_id = "locomo_mem0"
        self._version = self._detect_version()

    @staticmethod
    def _detect_version() -> str:
        try:
            from importlib.metadata import version
            return version("mem0ai")
        except Exception:  # noqa: BLE001
            return "unknown"

    def ingest(self, conversation: LocomoConversation) -> None:
        self._user_id = f"locomo_{conversation.conversation_id}"
        # Fresh scope per conversation: delete any leftovers from a
        # previous (crashed) run so memories never leak across convs.
        try:
            self._mem.delete_all(user_id=self._user_id)
        except Exception:  # noqa: BLE001 -- nothing to delete on first run
            pass
        # Map the dataset's declared speaker_a to the "user" role, matching
        # Mem0's own LOCOMO harness. Falling back to speakers[0] (alphabetical)
        # flips roles on 3/10 conversations and confounds this baseline, whose
        # fact-extraction is role-sensitive.
        speaker_a = conversation.speaker_a or (
            conversation.speakers[0] if conversation.speakers else ""
        )
        for turn in conversation.turns:
            role = "user" if turn.speaker == speaker_a else "assistant"
            content = _format_turn_text(turn)
            self._mem.add(
                [{"role": role, "content": content}],
                user_id=self._user_id,
                metadata={"dia_id": turn.dia_id, "session": turn.session},
            )

    def answer(self, question: LocomoQuestion) -> tuple[str, list[str], float]:
        t0 = time.perf_counter()
        # mem0 2.0.11 search API: top_k (not `limit`) and scope via
        # filters={'user_id': ...} (top-level user_id was removed).
        res = self._mem.search(
            question.question,
            top_k=self._top_k,
            filters={"user_id": self._user_id},
        )
        hits = res.get("results", res) if isinstance(res, dict) else res
        contents = [
            str(h.get("memory", h.get("text", ""))) for h in hits if isinstance(h, dict)
        ]
        context = "\n".join(f"- {_sanitize_locomo_text(c)}" for c in contents)
        prompt = ANSWER_PROMPT.format(
            context=context or "(no relevant memories)",
            question=_sanitize_locomo_text(question.question),
        )
        predicted = self._responder(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return predicted.strip(), contents, latency_ms

    def close(self) -> None:
        try:
            self._mem.delete_all(user_id=self._user_id)
        except Exception:  # noqa: BLE001
            pass


def run_baseline_eval(
    conversations: list[LocomoConversation],
    system,
    judge: LLMCallFn,
    *,
    judge_mode: str = "mem0",
    workers: int = 1,
    progress=None,
) -> list[LocomoResult]:
    """Run one baseline system over conversations; same result shape as
    run_locomo_eval so checkpoints/summaries merge into one table."""
    say = progress or (lambda msg: None)
    results: list[LocomoResult] = []
    for conv in conversations:
        say(f"  [{system.name}] conversation: {conv.conversation_id} "
            f"({len(conv.turns)} turns, {len(conv.questions)} q)")
        t0 = time.time()
        system.ingest(conv)
        say(f"    ingested in {time.time() - t0:.0f}s")

        def _eval_one(q: LocomoQuestion) -> PerQuestionResult:
            predicted, contents, latency_ms = system.answer(q)
            verdict = _judge_one(judge, q, predicted, judge_mode)
            return PerQuestionResult(
                question_id=q.question_id,
                question=q.question,
                gold=q.answer,
                predicted=predicted,
                category=q.category,
                judge_label=verdict.label,
                judge_score=verdict.score,
                judge_reason=verdict.reason,
                # dia_id-based hit rate is a genome-internal diagnostic;
                # external systems don't preserve turn provenance uniformly.
                retrieval_hit_rate=0.0,
                retrieved_ids=[],
                retrieved_contents=[c[:200] for c in contents],
                latency_ms=latency_ms,
            )

        if workers > 1 and len(conv.questions) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                per_q = list(pool.map(_eval_one, conv.questions))
        else:
            per_q = [_eval_one(q) for q in conv.questions]

        scores = [p.judge_score for p in per_q]
        by_cat: dict[str, list[float]] = {}
        count_cat: dict[str, int] = {}
        for p in per_q:
            by_cat.setdefault(p.category, []).append(p.judge_score)
            count_cat[p.category] = count_cat.get(p.category, 0) + 1
        results.append(
            LocomoResult(
                config_name=system.name,
                conversation_id=conv.conversation_id,
                n_questions=len(per_q),
                mean_score=statistics.mean(scores) if scores else 0.0,
                per_category_score={
                    c: statistics.mean(v) for c, v in by_cat.items()
                },
                per_category_count=count_cat,
                mean_retrieval_hit_rate=0.0,
                mean_latency_ms=(
                    statistics.mean(p.latency_ms for p in per_q) if per_q else 0.0
                ),
                per_question=per_q,
            )
        )
        say(f"    score={results[-1].mean_score:.3f}")
    return results


def _main() -> int:
    import argparse
    import os

    from genome.evals.locomo import (
        _checkpoint_path,
        _load_checkpoint,
        _make_metered_llm,
        _TokenMeter,
        load_all_checkpoints,
        print_summary_table,
        save_summary,
    )

    parser = argparse.ArgumentParser(description="Run LOCOMO baselines")
    default_dataset = Path("benchmarks/data/locomo10.json")
    parser.add_argument(
        "--dataset", type=str,
        default=str(default_dataset) if default_dataset.exists() else None,
    )
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/locomo"))
    parser.add_argument(
        "--systems", type=str, default="full-context",
        help="Comma-separated: full-context, mem0",
    )
    parser.add_argument("--llm", choices=["anthropic", "openai", "echo"],
                        default="openai")
    parser.add_argument("--responder-model", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-mode", choices=["mem0", "binary", "graded"],
                        default="mem0")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    _defaults = {"openai": "gpt-4o-mini",
                 "anthropic": "claude-haiku-4-5-20251001", "echo": "echo"}
    responder_model = args.responder_model or _defaults[args.llm]
    judge_model = args.judge_model or responder_model

    print(f"Loading LOCOMO from {args.dataset or 'HuggingFace'}...")
    conversations = load_locomo(args.dataset)
    if args.limit_conversations:
        conversations = conversations[: args.limit_conversations]
    if args.limit_questions:
        for c in conversations:
            c.questions = c.questions[: args.limit_questions]
    print(f"  loaded {len(conversations)} conversations, "
          f"{sum(len(c.questions) for c in conversations)} questions")

    responder_meter = _TokenMeter()
    judge_meter = _TokenMeter()
    if args.llm == "echo":
        def responder(prompt: str) -> str:
            responder_meter.record(0, 0)
            return "I don't know."
        def judge(prompt: str) -> str:
            judge_meter.record(0, 0)
            return "INCORRECT\nEcho LLM has no knowledge."
    elif args.llm == "anthropic":
        from anthropic import Anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Set ANTHROPIC_API_KEY env var.")
            return 1
        client = Anthropic()
        responder = _make_metered_llm(
            "anthropic", client, responder_model, 512, responder_meter)
        judge = _make_metered_llm(
            "anthropic", client, judge_model, 256, judge_meter)
    else:
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY env var.")
            return 1
        client = OpenAI()
        responder = _make_metered_llm(
            "openai", client, responder_model, 512, responder_meter)
        judge = _make_metered_llm(
            "openai", client, judge_model, 256, judge_meter)

    system_names = [s.strip() for s in args.systems.split(",") if s.strip()]
    factories = {
        "full-context": lambda: FullContextBaseline(responder, top_k=args.top_k),
        "mem0": lambda: Mem0Baseline(
            responder, top_k=args.top_k, llm_model=responder_model,
            llm_provider=("anthropic" if args.llm == "anthropic" else "openai"),
        ),
    }
    unknown = [s for s in system_names if s not in factories]
    if unknown:
        print(f"Unknown systems {unknown}. Available: {sorted(factories)}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)

    for name in system_names:
        system = factories[name]()
        try:
            for conv in conversations:
                ckpt = _checkpoint_path(
                    args.output_dir, system.name, conv.conversation_id)
                if not args.no_resume and _load_checkpoint(ckpt) is not None:
                    print(f"[resume] {system.name} / {conv.conversation_id}")
                    continue
                run = run_baseline_eval(
                    [conv], system, judge,
                    judge_mode=args.judge_mode,
                    workers=args.workers,
                    progress=lambda m: print(m, flush=True),
                )
                ckpt.write_text(
                    json.dumps(asdict(run[0]), indent=2, default=str),
                    encoding="utf-8",
                )
                print(f"  [checkpoint] {ckpt.name} "
                      f"(responder {responder_meter.calls:,} calls, "
                      f"judge {judge_meter.calls:,} calls)")
        finally:
            system.close()

    # Unified table over EVERYTHING checkpointed in this output dir
    # (genome configs + baselines).
    all_results = load_all_checkpoints(args.output_dir)
    summary = save_summary(all_results, args.output_dir / "summary.json")
    print_summary_table(summary)
    print(
        f"\nBaseline spend this run: responder {responder_meter.calls:,} calls "
        f"({responder_meter.input_tokens:,} in / "
        f"{responder_meter.output_tokens:,} out), "
        f"judge {judge_meter.calls:,} calls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
