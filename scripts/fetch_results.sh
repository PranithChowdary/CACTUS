#!/bin/bash
#
# Pull results and job logs back from kalpana. RUN THIS ON YOUR MAC, not on the
# cluster and not via sbatch.
#
#   bash scripts/fetch_results.sh            # results + logs
#   bash scripts/fetch_results.sh -n         # dry run: show what would transfer
#   bash scripts/fetch_results.sh results    # results only
#   bash scripts/fetch_results.sh logs       # logs only
#   bash scripts/fetch_results.sh checkpoints # trained model checkpoints too
#
# Why narrow paths instead of rsyncing the whole repo:
# pulling the whole tree has twice restored files that were deliberately
# deleted -- gemini_simulation.py (the fabricated-data generator) and
# paper/generate_expanded_paper.py, which overwrites the rewritten main.tex if
# anyone runs it. The cluster copy is stale for code; it is authoritative only
# for things the cluster PRODUCES. So: only pull what the jobs generate.

set -uo pipefail

REMOTE="pranith@kalpana.minds.iitb.ac.in"
RPATH="/users/staff/pranith/CACTUS"
LPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY=""
WHAT="all"
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY="-n" ;;
        results|logs|checkpoints|all) WHAT="$arg" ;;
        *) echo "unknown argument: $arg"; exit 1 ;;
    esac
done

[ -n "$DRY" ] && echo ">>> DRY RUN -- nothing will be written" && echo

fetch () {   # fetch <remote-subpath> <local-subpath>
    echo "--- $1"
    mkdir -p "$LPATH/$2"
    rsync -avzu $DRY "$REMOTE:$RPATH/$1" "$LPATH/$2"
}

if [ "$WHAT" = "all" ] || [ "$WHAT" = "results" ]; then
    fetch "results/" "results/"
fi

if [ "$WHAT" = "all" ] || [ "$WHAT" = "logs" ]; then
    # Job stdout/stderr only -- these are the provenance for every number.
    # Note SLURM writes to the directory you submitted FROM, so logs may be in
    # scripts/logs/ if you submitted from inside scripts/.
    echo "--- logs/*.out, logs/*.err"
    mkdir -p "$LPATH/logs"
    rsync -avzu $DRY --include='*.out' --include='*.err' --exclude='*' \
        "$REMOTE:$RPATH/logs/" "$LPATH/logs/"
    rsync -avzu $DRY --include='*.out' --include='*.err' --exclude='*' \
        "$REMOTE:$RPATH/scripts/logs/" "$LPATH/logs/" 2>/dev/null || true
fi

if [ "$WHAT" = "checkpoints" ]; then
    fetch "logs/checkpoints/" "logs/checkpoints/"
fi

echo
if [ -n "$DRY" ]; then
    echo "Dry run complete. Re-run without -n to transfer."
else
    echo "Done. Local results:"
    ls -lat "$LPATH/results/" 2>/dev/null | head -12
fi
