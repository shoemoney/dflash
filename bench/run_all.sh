#!/bin/sh
# Units C-E, sequential so tok/s numbers are not skewed by a second model on the GPU.
cd ~/dflash-port/dflash || exit 1
PY=../.venv/bin/python
R=~/dflash-port/runs
STOCK=mlx-community/Muse-Glimmer-30B-4bit
ABL=shoemoney/Muse-Glimmer-30B-Abliterated-MLX-q4
META=meta-models/Muse-Glimmer-30B-assistant
DF2=z-lab/Muse-Glimmer-30B-DFlash2
COMMON="--max-tokens 256 --reasoning low"
run() { name=$1; shift; echo "== $name $(date +%H:%M:%S)"; $PY -m dflash.bench_mlx "$@" $COMMON --out $R/$name.jsonl > $R/$name.log 2>&1 || echo "FAILED $name"; tail -6 $R/$name.log; }
run C-meta-4bit        --model $STOCK --draft $META --draft-bits 4 --block-sizes 5,8,16 --temperature 0 --baseline $R/baseline-4bit.jsonl
run C-meta-bf16        --model $STOCK --draft $META               --block-sizes 5,8    --temperature 0 --baseline $R/baseline-4bit.jsonl
run D-baseline-abl     --model $ABL                                                     --temperature 0
run D-dflash2-abl      --model $ABL   --draft $DF2  --draft-bits 4 --block-sizes 5,8    --temperature 0 --baseline $R/D-baseline-abl.jsonl
run D-meta-abl         --model $ABL   --draft $META --draft-bits 4 --block-sizes 5,8    --temperature 0 --baseline $R/D-baseline-abl.jsonl
run E-meta-sampled     --model $STOCK --draft $META --draft-bits 4 --block-sizes 5,8    --temperature 1 --top-p 0.95 --top-k 64
run E-baseline-sampled --model $STOCK                                                   --temperature 1 --top-p 0.95 --top-k 64
echo "CDE-DONE $(date +%H:%M:%S)"
