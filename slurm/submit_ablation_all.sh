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
# Or run foreground if you'd rather watch it live:
#     bash slurm/submit_ablation_all.sh
# -----------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."   # go to project root

echo "[$(date)] Submitting ablation array (6 of 7 variants)..."
ARRAY_JOB=$(sbatch --parsable slurm/train_ablation_array.sbatch)
echo "[$(date)] Array jobid: $ARRAY_JOB"

echo "[$(date)] Polling queue for a free slot (max 6 submitted; need <=5 to add one)..."
while true; do
    SUBMITTED=$(squeue -u "$USER" -h -o "%A" | wc -l)
    echo "[$(date)] Currently submitted: $SUBMITTED"
    if [ "$SUBMITTED" -lt 6 ]; then
        echo "[$(date)] Slot available — submitting the 7th variant (dem alone)"
        sbatch slurm/train_ablation_single.sbatch
        echo "[$(date)] Done. All 7 ablation variants now in the pipeline."
        exit 0
    fi
    sleep 300    # 5 minutes
done
