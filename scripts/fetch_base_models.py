#!/usr/bin/env python
"""
Download the base checkpoints for run_base_vs_instruct.sh.

RUN THIS ON THE LOGIN NODE. Compute nodes on cn4_mangala have no outbound
internet, so every network step has to happen here first.

    python scripts/fetch_base_models.py
    python scripts/fetch_base_models.py --check      # probe access, no download

Three environment facts this script exists to encapsulate, each of which has
already cost a debugging round:

1. The inherited HF_HOME is /janaki/common/huggingface, which is READ-ONLY to
   this account. Any download through it dies with EACCES while "ignoring" the
   failure to write a ref, and `hf auth login` silently fails to persist the
   token -- the OAuth succeeds, prints "Token is valid.", then raises
   PermissionError on stored_tokens, leaving you unauthenticated. So HF_HOME is
   overridden below, BEFORE huggingface_hub is imported, because the library
   reads its path constants at import time.

2. Base checkpoints therefore cannot live in the shared cache and are written
   to --dest as plain directories, loaded by path. The instruct checkpoints are
   already in the shared cache and are NOT touched by this script.

3. huggingface_hub.model_info() is NOT an access check. Metadata on a gated
   repo is public, so model_info() returns successfully for a repo you are not
   authorized to download, which looks exactly like success. Access is probed
   by resolving an actual FILE.
"""
import argparse
import os
import sys

# --- must precede the huggingface_hub import; see docstring note 1 -----------
_DEFAULT_HF_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
if "--keep-hf-home" not in sys.argv:
    os.environ["HF_HOME"] = os.environ.get("CACTUS_HF_HOME", _DEFAULT_HF_HOME)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import json

from huggingface_hub import hf_hub_download, snapshot_download, whoami
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

DEST_DEFAULT = "/users/staff/pranith/models"

# repo id -> local directory name. The directory names must match BASE_DIR
# entries in scripts/run_base_vs_instruct.sh.
MODELS = [
    ("Qwen/Qwen2.5-7B", "Qwen2.5-7B"),
    ("meta-llama/Llama-3.1-8B", "Llama-3.1-8B"),
]

# Llama repos ship a second full copy of the weights under original/ as .pth
# consolidated files -- ~16 GB of duplicate download that transformers never
# loads. Skip them.
IGNORE = ["original/*", "*.pth", "*.gguf", "*.msgpack", "*.h5"]


def verify(path):
    """Return (ok, message). Checks the shard index against what is on disk."""
    if not os.path.isdir(path):
        return False, "directory does not exist"

    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return False, "no config.json"
    try:
        with open(cfg) as fh:
            json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"config.json is not readable JSON ({exc})"

    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
    else:
        single = os.path.join(path, "model.safetensors")
        if not os.path.isfile(single):
            return False, "no shard index and no model.safetensors"
        shards = ["model.safetensors"]

    missing = [s for s in shards if not os.path.isfile(os.path.join(path, s))]
    if missing:
        return False, f"{len(missing)}/{len(shards)} shards missing: {missing[:3]}"

    empty = [s for s in shards if os.path.getsize(os.path.join(path, s)) == 0]
    if empty:
        return False, f"{len(empty)} zero-byte shards: {empty[:3]}"

    tok = [f for f in ("tokenizer.json", "tokenizer.model", "vocab.json")
           if os.path.isfile(os.path.join(path, f))]
    if not tok:
        return False, "no tokenizer file"

    gb = sum(os.path.getsize(os.path.join(path, s)) for s in shards) / 1e9
    return True, f"{len(shards)} shards, {gb:.1f} GB, tokenizer={tok[0]}"


def probe(repo):
    """Resolve one real file. Returns (ok, reason). See docstring note 3."""
    try:
        hf_hub_download(repo, "config.json")
        return True, "authorized"
    except GatedRepoError as exc:
        code = "403" if "403" in str(exc) else "401" if "401" in str(exc) else "?"
        if code == "401":
            return False, ("NOT AUTHENTICATED -- the token was not found or not "
                           "sent. Run: hf auth login  (with HF_HOME writable)")
        return False, ("NOT APPROVED -- authenticated, but not on the authorized "
                       f"list. Accept the licence at https://huggingface.co/{repo}")
    except RepositoryNotFoundError:
        return False, "repo id does not exist (check the spelling)"
    except Exception as exc:                                # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEST_DEFAULT,
                    help=f"where to write the checkpoints (default {DEST_DEFAULT})")
    ap.add_argument("--check", action="store_true",
                    help="probe access and verify what is present; download nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if verification already passes")
    ap.add_argument("--keep-hf-home", action="store_true",
                    help="do NOT override HF_HOME (will fail if it is read-only)")
    args = ap.parse_args()

    print(f"HF_HOME = {os.environ.get('HF_HOME')}")
    try:
        print(f"logged in as: {whoami()['name']}")
    except Exception:                                       # noqa: BLE001
        print("logged in as: NOBODY -- run `hf auth login` first.")
        print("  If that fails with PermissionError on stored_tokens, your")
        print("  HF_HOME is the read-only shared cache; export a writable one.")
        return 1

    os.makedirs(args.dest, exist_ok=True)
    results = []

    for repo, name in MODELS:
        path = os.path.join(args.dest, name)
        print(f"\n{'=' * 70}\n{repo}\n  -> {path}\n{'=' * 70}")

        ok, msg = verify(path)
        if ok and not args.force:
            print(f"  already complete: {msg}")
            results.append((name, True, "present"))
            continue
        if os.path.isdir(path):
            print(f"  incomplete ({msg}) -- resuming")

        allowed, reason = probe(repo)
        if not allowed:
            print(f"  SKIP: {reason}")
            results.append((name, False, reason.split(" -- ")[0]))
            continue

        try:
            snapshot_download(repo, local_dir=path, ignore_patterns=IGNORE,
                              max_workers=4)
        except Exception as exc:                             # noqa: BLE001
            print(f"  DOWNLOAD FAILED: {type(exc).__name__}: {exc}")
            results.append((name, False, "download failed"))
            continue

        ok, msg = verify(path)
        print(f"  {'verified' if ok else 'CORRUPT'}: {msg}")
        results.append((name, ok, msg if ok else "failed verification"))

    print(f"\n{'=' * 70}\nsummary\n{'=' * 70}")
    for name, ok, msg in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<16} {msg}")

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_ok}/{len(results)} base checkpoints ready.")
    if n_ok == 0:
        print("Nothing to run. Resolve the access problems above first.")
        return 1
    if n_ok < len(results):
        print("run_base_vs_instruct.sh will skip the missing model and run the")
        print("rest -- its preflight reports a missing checkpoint as such.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
