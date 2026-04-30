#!/usr/bin/env bash

# Bash script to detect slow waves via NCI GADI - aaron.lam@sydney.edu.au (24 Oct 2025)
# submit_all_sw.sh
set -euo pipefail

############################
# USER CONFIG (EDIT ME!) 
############################
ROOT="/scratch/ca97/MCIdata"                             
MASTER_LIST="/scratch/ca97/al_code/subjects.txt"          
PBS="/scratch/ca97/al_code/AL_sw_batch.pbs"             
PY_SCRIPT="/scratch/ca97/al_code/hdEEG_sw_detector_GADI.py"     

METHOD="Staresina2015"                                      # Massimini2004, AASM/Massimini2004, Ngo2015, Staresina2015
STAGES="NREM2,NREM3"
FREQ="0.1,4.0"
TROUGH_DURATION="0.8,2.0"
NEG_PEAK_THRESH="-20.0"
P2P_THRESH="40.0"
POLAR="normal"                                              # normal or opposite

JOB_PREFIX="AL_sw"
SLEEP_BETWEEN=10                                          
LOGDIR="logs"
LISTSDIR="lists"


mkdir -p "$LOGDIR" "$LISTSDIR"

if [[ ! -f "$MASTER_LIST" ]]; then
  echo "ERROR: master subjects list not found: $MASTER_LIST" >&2
  exit 2
fi
if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found in PATH." >&2
  exit 3
fi
if [[ ! -f "$PBS" ]]; then
  echo "ERROR: PBS file not found: $PBS" >&2
  exit 4
fi

dos2unix -q "$MASTER_LIST" 2>/dev/null || true

echo "Submitting SW jobs using:"
echo "  ROOT=$ROOT"
echo "  MASTER_LIST=$MASTER_LIST"
echo "  PBS=$PBS"
echo "  PY_SCRIPT=$PY_SCRIPT"
echo "  METHOD=$METHOD"
echo "  STAGES=$STAGES"
echo "  FREQ=$FREQ"
echo "  TROUGH_DURATION=$TROUGH_DURATION"
echo "  NEG_PEAK_THRESH=$NEG_PEAK_THRESH"
echo "  P2P_THRESH=$P2P_THRESH"
echo "  POLAR=$POLAR"
echo

i=0
while IFS= read -r SUBJ || [[ -n "${SUBJ:-}" ]]; do
  # trim whitespace / CR
  SUBJ="$(echo -n "$SUBJ" | tr -d '\r' | xargs || true)"
  # skip blanks/comments
  [[ -z "${SUBJ}" || "${SUBJ:0:1}" == "#" ]] && continue

  i=$((i+1))
  one_list="${LISTSDIR}/${SUBJ}.txt"
  printf '%s\n' "$SUBJ" > "$one_list"

  # Export env so qsub -V can carry them into the job
  export ROOT
  export LIST="$one_list"
  export PY_SCRIPT

  export METHOD
  export STAGES
  export FREQ
  export TROUGH_DURATION
  export NEG_PEAK_THRESH
  export P2P_THRESH
  export POLAR

  jobname="${JOB_PREFIX}_${SUBJ}"
  echo "[$(date)] ($i) submitting ${SUBJ} as ${jobname} ..."
    qsub -N "$jobname" -V "$PBS"
    sleep "$SLEEP_BETWEEN"
  
done < "$MASTER_LIST"

echo "[$(date)] Submission loop finished. Total queued: $i"
