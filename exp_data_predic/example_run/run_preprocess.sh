#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$RUN_DIR/data_raw"
PROC_DIR="$RUN_DIR/data_proc/fids_raw"
NMRPIPE_DIR="$RUN_DIR/nmrpipe"

SCRIPT1="$NMRPIPE_DIR/fid.com"
SCRIPT2="$NMRPIPE_DIR/process.com"

mkdir -p "$PROC_DIR"

# Example: raw folders are data_raw/480 ... data_raw/494
for i in $(seq 480 494); do
  SRC="$RAW_DIR/$i"
  DST="$PROC_DIR/$i"

  if [[ -d "$SRC" ]]; then
    echo "==> Processing $i"
    mkdir -p "$DST"
    rsync -a "$SRC/" "$DST/"

    cp "$SCRIPT1" "$SCRIPT2" "$DST/"

    (
      cd "$DST"
      tcsh "./fid.com"
      tcsh "./process.com"
    )
  else
    echo "[SKIP] Missing dir: $SRC"
  fi
done

echo "Done. Processed files are in: $PROC_DIR/<id>/fid_phased.fid"

