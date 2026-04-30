#!/usr/bin/env bash
# Bash script to detect spindles via NCI GADI - aaron.lam@sydney.edu.au (24 Oct 2025)
# submit_all_annotator.sh

set -euo pipefail

############################
# USER CONFIG (EDIT ME!) 
############################

ROOT="/scratch/ca97/MCIdata"
MASTER_LIST="/scratch/ca97/al_code/subjects.txt"
PBS="/scratch/ca97/al_code/AL_annotator_batch.pbs"
PY_SCRIPT="/scratch/ca97/al_code/hdEEG_annotator_GADI.py"

JOB_PREFIX="AL_annotator"
SLEEP_BETWEEN=10
LOGDIR="logs"
LISTSDIR="lists"

mkdir -p "$LOGDIR" "$LISTSDIR"

i=0
while IFS= read -r SUBJ || [[ -n "${SUBJ:-}" ]]; do
  SUBJ="$(echo -n "$SUBJ" | tr -d '\r' | xargs || true)"
  [[ -z "${SUBJ}" || "${SUBJ:0:1}" == "#" ]] && continue

  i=$((i+1))
  one_list="${LISTSDIR}/${SUBJ}.txt"
  printf '%s\n' "$SUBJ" > "$one_list"

  export ROOT
  export LIST="$one_list"
  export PY_SCRIPT

  echo "[$(date)] ($i) submitting ${SUBJ} ..."
    qsub -N "${JOB_PREFIX}_${SUBJ}" -V "$PBS"
    sleep "$SLEEP_BETWEEN"
done < "$MASTER_LIST"

echo "[$(date)] Submitted $i jobs total."
