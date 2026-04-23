#!/bin/bash
# -----------------------------------------------------------------------------
# Submit all 7 ablation variants on Isaac campus-gpu QOS (6-submit limit).
#
# Step 1: submit the 6-array (indices 0..5 — the 6 most important variants)
# Step 2: poll the queue every 5 minutes; submit the 7th (dem alone) the
#         moment a slot frees up.
#
# Run this as a background process so it can poll overnight:
#     nohup bash slurm/submit_ablation_all.sh > submit_ablation.log 2>&1 &
#
# Or run foreground to watch it live:
#     bash slurm/submit_ablation_all.sh
#
# IMPORTANT: this script uses `squeue -r` to count array tasks individually
# (each array task counts against the 6-submit QOS limit, even if squeue
#  displays them collapsed as e.g. "5630526_[0-5]").
# -----------------------------------------------------------------------------

# Note: no `-e` — a failed sbatch (QOS limit) should trigger retry, not exit.
set -uo pipefail

cd "$(dirname "$0")/.."   # go to project root

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Guard: don't double-submit the array if one is already queued.
EXISTING=$(squeue -u "$USER" -h -r -n ablation -o "%A" 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ]; then
    log "Found $EXISTING existing 'ablation' jobs in queue — skipping array submission"
    log "Will still poll and submit the 7th variant when a slot frees"
else
    log "Submitting 6-task ablation array..."
    ARRAY_JOB=$(sbatch --parsable slurm/train_ablation_array.sbatch)
    rc=$?
    if [ $rc -ne 0 ]; then
        log "sbatch for array failed with exit $rc — aborting"
        exit $rc
    fi
    log "Array submitted: job id $ARRAY_JOB"
fi

log "Polling for a free slot (need ≤5 submitted to add the 7th variant)..."
ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    # -r expands array tasks into individual rows; each counts against MaxSubmit=6
    SUBMITTED=$(squeue -u "$USER" -h -r -o "%A" 2>/dev/null | wc -l)
    log "Poll #$ATTEMPT — currently submitted: $SUBMITTED / 6"

    if [ "$SUBMITTED" -lt 6 ]; then
        log "Slot available — submitting 7th variant (dem alone)"
        if sbatch slurm/train_ablation_single.sbatch; then
            log "Success — all 7 ablation variants now in the pipeline."
            exit 0
        else
            log "sbatch failed (QOS limit may have raced). Will retry in 5min."
        fi
    fi
    sleep 300    # 5 minutes
done
