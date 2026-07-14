#!/usr/bin/env bash
# Mem0 keeps dying ~1 conversation per launch on this box (process/network
# exhaustion). It checkpoints per conversation, so just keep resuming until all
# 10 are done. Bounded to avoid an infinite loop.
set +e
cd ~/projects/active/genome
source ~/scratch/locomo.env
DIR=results/locomo_claude_v2
for i in $(seq 1 20); do
  n=$(ls "$DIR"/checkpoints/baseline-mem0__*.json 2>/dev/null | wc -l)
  echo "[autoresume $i] mem0 checkpoints so far: $n/10"
  if [ "$n" -ge 10 ]; then echo "[autoresume] mem0 COMPLETE"; break; fi
  .venv/Scripts/python.exe -u -m genome.evals.baselines \
    --systems mem0 --llm anthropic \
    --responder-model claude-haiku-4-5-20251001 \
    --judge-model claude-haiku-4-5-20251001 \
    --judge-mode mem0 --workers 4 \
    --output-dir "$DIR" >> "$DIR/run_mem0_auto.log" 2>&1
  echo "[autoresume $i] mem0 exited; pausing before retry"
  sleep 45
done
echo "[autoresume] done at $(ls "$DIR"/checkpoints/baseline-mem0__*.json 2>/dev/null | wc -l)/10"
