"""Turn bench_mlx JSONL runs into markdown tables for the README.

Usage: aggregate.py RUNS_DIR
Groups rows by (target, draft, bits, block, temperature) and prints one table per target
with mean tok/s, speedup over that target's draft-less rows at the same temperature,
mean accepted tokens per verify step, and peak memory.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

runs = Path(sys.argv[1])
rows = [json.loads(l) for f in sorted(runs.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()]

groups = defaultdict(list)
for r in rows:
    groups[(r["model"], r["draft"], r["draft_bits"], r["block_size"], r["temperature"])].append(r)

def short(model):
    return model.split("/")[-1]

def draft_name(d, bits):
    if d is None:
        return "none (plain mlx-lm)"
    return f"{d.split('/')[-1]} ({bits or 'bf16'})"

targets = sorted({k[0] for k in groups})
for target in targets:
    print(f"\n### {short(target)}\n")
    print("| draft | block | temp | tok/s | speedup | accept/step | peak GB |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    keys = sorted((k for k in groups if k[0] == target), key=lambda k: (k[4], k[1] or "", k[2] or 0, k[3] or 0))
    for k in keys:
        g = groups[k]
        base = groups.get((target, None, None, None, k[4]))
        tps = statistics.mean(r["gen_tps"] for r in g)
        base_tps = statistics.mean(r["gen_tps"] for r in base) if base else None
        acc = [r["mean_accept"] for r in g if r["mean_accept"] is not None]
        speed = f"{tps / base_tps:.2f}x" if base_tps and k[1] else "1.00x" if base_tps else "-"
        print(f"| {draft_name(k[1], k[2])} | {k[3] or '-'} | {k[4]:g} | {tps:.1f} | {speed} | {statistics.mean(acc):.2f} | {max(r['peak_gb'] for r in g):.1f} |" if acc else
              f"| {draft_name(k[1], k[2])} | - | {k[4]:g} | {tps:.1f} | {speed} | - | {max(r['peak_gb'] for r in g):.1f} |")

print("\n### Per-prompt tok/s, stock 4-bit, greedy\n")
names = ["factual", "code", "essay", "math", "json", "zh"]
cols = [(None, None, None), ("z-lab/Muse-Glimmer-30B-DFlash2", 4, 5), ("meta-models/Muse-Glimmer-30B-assistant", 4, 5),
        ("z-lab/Muse-Glimmer-30B-DFlash2", 4, 8), ("meta-models/Muse-Glimmer-30B-assistant", 4, 8)]
print("| prompt | " + " | ".join("baseline" if c[0] is None else f"{'DFlash2' if 'z-lab' in c[0] else 'Meta'} b{c[2]}" for c in cols) + " |")
print("|---|" + "---:|" * len(cols))
stock = "mlx-community/Muse-Glimmer-30B-4bit"
for i, name in enumerate(names):
    cells = []
    for d, bits, blk in cols:
        g = [r for r in groups.get((stock, d, bits, blk, 0.0), []) if r["prompt_index"] == i]
        cells.append(f"{g[0]['gen_tps']:.1f}" + (f" ({g[0]['mean_accept']:.1f})" if g and g[0]["mean_accept"] else "") if g else "-")
    print(f"| {name} | " + " | ".join(cells) + " |")
