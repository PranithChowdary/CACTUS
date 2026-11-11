"""
merge_topup.py

Folds a recovered config back into a model's result file.

experiments/llm_probe_constrained.py starts from an empty list and rewrites its
--out file, so a top-up run cannot share the filename of the run it is topping
up. It writes results/llm_probe_bank61_topup_<tag>.json; this merges that into
results/llm_probe_bank61_<tag>.json, keyed on (scoring, alpha), refusing to
silently overwrite a config that is already present.

    python scripts/merge_topup.py
"""
import glob, json, os, sys

def main():
    tops = sorted(glob.glob("results/llm_probe_bank61_topup_*.json"))
    if not tops:
        print("no results/llm_probe_bank61_topup_*.json found -- nothing to merge")
        return 0
    for tf in tops:
        tag = os.path.basename(tf).replace("llm_probe_bank61_topup_", "").replace(".json", "")
        base = f"results/llm_probe_bank61_{tag}.json"
        if not os.path.exists(base):
            print(f"  {tag}: no base file {base}; leaving the top-up as-is")
            continue
        b = json.load(open(base)); t = json.load(open(tf))
        have = {(r["scoring"], r["alpha"]) for r in b}
        added = []
        for r in t:
            k = (r["scoring"], r["alpha"])
            if k in have:
                print(f"  {tag}: {k} already present -- NOT overwriting")
                continue
            b.append(r); added.append(k)
        b.sort(key=lambda r: (r["scoring"], r["alpha"]))
        json.dump(b, open(base, "w"), indent=2)
        print(f"  {tag}: added {added or 'nothing'} -> {len(b)} configs in {base}")
        # Retire the source so it stops matching results/llm_probe_bank61_*.json,
        # where compare_models.py would otherwise see a partial run and warn.
        # Renamed, not deleted: the merge is the only copy otherwise.
        os.rename(tf, tf + ".merged")
        print(f"  {tag}: retired {os.path.basename(tf)} -> .merged")
    return 0

if __name__ == "__main__":
    sys.exit(main())
