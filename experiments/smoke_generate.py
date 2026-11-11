"""
smoke_generate.py

Isolates the LLM generation path from the experiment. Job 50364 loaded the
model onto cuda:0 in 5s and then produced no output for three hours, which
means the stall is inside generation, not in model loading, the environment,
or the metrics. This script tests generation alone, smallest case first, so
the failure shows up in seconds instead of hours.

Escalating sequence -- whichever line fails to print is the culprit:
  1. one trivial prompt, batch of 1            (does generate() work at all?)
  2. one real env prompt, batch of 1           (is it prompt length/content?)
  3. real env prompts, batch of 4              (is it batching/padding?)
  4. real env prompts, batch of 32             (is it the production batch size?)

Run interactively on a compute node, NOT as a long batch job:
    srun --partition=cn4_mangala --qos=mangala --account=cminds_anandi \
         --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:20:00 --pty \
         bash -c 'source ~/miniconda3/bin/activate && conda activate ./cactus_env && \
                  PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
                  HF_HOME=/janaki/common/huggingface \
                  python experiments/smoke_generate.py'
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.nl_conditional_env import NLConditionalEnv

MODEL = os.environ.get("CACTUS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")


def main():
    import transformers
    print(f"transformers {transformers.__version__}  torch {torch.__version__}", flush=True)
    print(f"cuda available: {torch.cuda.is_available()}  "
          f"devices: {torch.cuda.device_count()}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto", local_files_only=True)
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s, device={model.device}, "
          f"dtype={next(model.parameters()).dtype}", flush=True)
    print(f"mem allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    def gen(prompts, label, max_new_tokens=12):
        print(f"\n--- {label}: {len(prompts)} prompt(s) ---", flush=True)
        t = time.time()
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        enc.pop("token_type_ids", None)
        print(f"  tokenized: {tuple(enc['input_ids'].shape)} in {time.time() - t:.1f}s", flush=True)
        t = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        dt = time.time() - t
        gen_ids = out[:, enc.input_ids.shape[1]:]
        texts_out = tok.batch_decode(gen_ids, skip_special_tokens=True)
        print(f"  generated in {dt:.1f}s  ({dt / len(prompts):.2f}s/prompt)", flush=True)
        for o in texts_out[:3]:
            print(f"    -> {o.strip()!r}", flush=True)
        return texts_out

    # 1. does generate() work at all?
    gen(["Say the word: hello"], "1. trivial prompt, batch 1")

    # 2. is it the real prompt's length or content?
    env = NLConditionalEnv(alpha=1.0, seed=0)
    obs, info = env.reset(options={"scenario": 0})
    ntok = len(tok(obs)["input_ids"])
    print(f"\n(real env prompt is {ntok} tokens)", flush=True)
    gen([obs], "2. real env prompt, batch 1")

    # 3. does batching/padding break it?
    prompts = []
    for i in range(4):
        e = NLConditionalEnv(alpha=1.0, seed=i)
        o, _ = e.reset(options={"scenario": i})
        prompts.append(o)
    gen(prompts, "3. real env prompts, batch 4")

    # 4. the production batch size
    prompts32 = []
    for i in range(32):
        e = NLConditionalEnv(alpha=1.0, seed=i)
        o, _ = e.reset(options={"scenario": i % 7})
        prompts32.append(o)
    gen(prompts32, "4. real env prompts, batch 32")

    print("\nALL STAGES PASSED -- generation path is healthy.", flush=True)
    print("If the full experiment still stalls, the problem is in the episode "
          "loop, not generation.", flush=True)


if __name__ == "__main__":
    main()
