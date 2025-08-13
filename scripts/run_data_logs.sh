#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
# --- CONFIG ---
BASE_DIR="/scratch/$USER/parallel-vertex-clustering"
EXE="${EXE:-$BASE_DIR/build/merge-vertices}"
LOGDIR="${LOGDIR:-$BASE_DIR/logs}"
OUTDIR="${OUTDIR:-$BASE_DIR/outputs}"
DATADIR="${DATADIR:-$BASE_DIR/data}"
# datasets -> ply
declare -A PLY=(
  [bunny]="$DATADIR/bunny.ply"
  [lucy]="$DATADIR/lucy.ply"
)

# eps per dataset (strings that match file names & your extract regex)
declare -A EPS
EPS[lucy]="8.000e-4 7.000e-3 7.200e-2 5.009e-1"
EPS[bunny]="2.800e-4 4.280e-4 9.965e-4 1.209e-3"
# alg list: "Name:Id"
ALGS=("S-Weld:0" "P-Weld:1" "P-Weld-Async:2")

THREADS=${THREADS:-"1 2 4 8 16 32 64"}

mkdir -p "$LOGDIR" "$OUTDIR"

for ds in bunny lucy; do
  ply="${PLY[$ds]}"
  if [[ ! -f "$ply" ]]; then
    echo "[warn] missing PLY for $ds at $ply" >&2
    continue
  fi
  for alg in "${ALGS[@]}"; do
    algName="${alg%%:*}"
    algId="${alg##*:}"
    for eps in ${EPS[$ds]}; do
      for t in $THREADS; do
        log="$LOGDIR/eps${eps}alg${algName}n0t${t}data${ds}.log"
        out="$OUTDIR/${ds}_${algName}_eps${eps}_${t}cores.ply"
        if [[ -s "$log" ]]; then
          echo "[skip] $log exists"
          continue
        fi
        echo "[Run] $ds $algName eps=$eps t=$t"
        srun -c "$t" "$EXE" "$eps" "$algId" "$ply" "$t" "$out" > "$log" 2>&1
      done
    done
  done
done

echo "Done. Compatible logs in $LOGDIR"
