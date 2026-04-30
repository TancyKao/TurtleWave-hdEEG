#!/usr/bin/env bash

# Bash script to detect spindles via NCI GADI - aaron.lam@sydney.edu.au (24 Oct 2025)
# submit_all_spl.sh
set -euo pipefail

############################
# USER CONFIG (EDIT ME!) 
############################
ROOT="/scratch/ca97/MCIdata"                               # parent dir of subject folders
MASTER_LIST="/scratch/ca97/al_code/subjects.txt"           
PBS="/scratch/ca97/al_code/AL_spindle_batch.pbs"         
PY_SCRIPT="/scratch/ca97/al_code/hdEEG_spindle_detector_GADI.py"


METHOD="Moelle2011" 
STAGES="NREM2,NREM3"
FREQ="9.0,12.0"
DURATION="0.5,3.0"


JOB_PREFIX="AL_spindle"
SLEEP_BETWEEN=10        
LOGDIR="logs"
LISTSDIR="lists"

i=0
while IFS= read -r SUBJ || [[ -n "${SUBJ:-}" ]]; do
  # trim whitespace / CR
  SUBJ="$(echo -n "$SUBJ" | tr -d '\r' | xargs || true)"
  [[ -z "${SUBJ}" || "${SUBJ:0:1}" == "#" ]] && continue

  i=$((i+1))
  one_list="${LISTSDIR}/${SUBJ}.txt"
  printf '%s\n' "$SUBJ" > "$one_list"

  export ROOT
  export LIST="$one_list"
  export METHOD
  export STAGES
  export FREQ
  export DURATION
  export PY_SCRIPT

  echo "[$(date)] ($i) submitting ${SUBJ} ..."
    qsub -N "${JOB_PREFIX}_${SUBJ}" -V "$PBS"
    sleep "$SLEEP_BETWEEN"
done < "$MASTER_LIST"

echo "[$(date)] Submission loop finished. Total submitted: $i"
