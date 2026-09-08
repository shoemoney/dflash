"""Does the target's own argmax flip between single-token decode and a batched pass?

For each (prompt, position) pair, replay the baseline's tokens up to `position`
two ways: one token per step (as plain decode does) and as one batched forward
(as the DFlash verify does). Print the top-2 logit margin and both argmaxes.
"""
import json
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from dflash.benchmark import apply_chat_template

model_id, baseline_path, prompts_path = sys.argv[1:4]
pairs = [tuple(int(x) for x in p.split(":")) for p in sys.argv[4:]]

model, tokenizer = load(model_id)
prompts = json.load(open(prompts_path))
rows = {json.loads(l)["prompt_index"]: json.loads(l) for l in open(baseline_path)}


def logits_at(ids, batched):
    cache = make_prompt_cache(model)
    x = mx.array(ids)[None]
    if batched:
        out = model(x, cache=cache)[:, -1]
    else:
        for i in range(x.shape[1]):
            out = model(x[:, i : i + 1], cache=cache)[:, -1]
    mx.eval(out)
    return out[0].astype(mx.float32)


for p, pos in pairs:
    text = apply_chat_template(tokenizer, [{"role": "user", "content": prompts[p]}], "low")
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    ids = prompt_ids + rows[p]["tokens"][:pos]
    for mode in ("single", "batched"):
        lg = logits_at(ids, mode == "batched")
        top = mx.argsort(-lg)[:2].tolist()
        margin = (lg[top[0]] - lg[top[1]]).item()
        print(f"p{p} pos{pos} {mode:8s} argmax={top[0]} second={top[1]} margin={margin:.4f}")
