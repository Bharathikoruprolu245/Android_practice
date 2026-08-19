#!/bin/bash
# Downloads every PDB structure listed in nrdld_labels.csv into ./structures/
# Run this on your WSL box (needs real internet access to files.rcsb.org).
#
# Usage:
#   chmod +x download_nrdld_structures.sh
#   ./download_nrdld_structures.sh
#
# Then train with:
#   python train_druggability_model.py \
#       --labels-csv nrdld_labels.csv \
#       --structures-dir ./structures \
#       --model-out druggability_model.joblib

set -e

STRUCTURES_DIR="./structures"
LABELS_CSV="nrdld_labels.csv"
MAX_RETRIES=4
SLEEP_BETWEEN=1.5   # seconds between requests -- raised from 0.3s, which was
                     # too aggressive and likely tripped RCSB's rate limiting
                     # partway through the list (successful downloads early
                     # on, then a wall of failures, is the classic symptom)

mkdir -p "$STRUCTURES_DIR"

fetch_one() {
    local pdb_id="$1"
    local out_file="$2"
    local attempt=1
    local backoff=2

    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        if wget -q --user-agent="Mozilla/5.0 (compatible; biomedix-ai-research/1.0)" \
                "https://files.rcsb.org/download/${pdb_id}.pdb" -O "$out_file" \
                && [ -s "$out_file" ]; then
            return 0
        fi
        echo "  [retry $attempt/$MAX_RETRIES] $pdb_id failed — waiting ${backoff}s"
        rm -f "$out_file"
        sleep "$backoff"
        backoff=$((backoff * 2))
        attempt=$((attempt + 1))
    done
    return 1
}

rm -f failed_downloads.txt

tail -n +2 "$LABELS_CSV" | cut -d',' -f1 | while read -r pdb_id; do
    out_file="$STRUCTURES_DIR/${pdb_id}.pdb"
    if [ -f "$out_file" ]; then
        echo "[skip] $pdb_id.pdb already exists"
        continue
    fi
    echo "[fetch] $pdb_id"
    if fetch_one "$pdb_id" "$out_file"; then
        :
    else
        echo "[FAILED] $pdb_id — giving up after $MAX_RETRIES attempts" >> failed_downloads.txt
    fi
    sleep "$SLEEP_BETWEEN"
done

echo ""
echo "Done. $(ls "$STRUCTURES_DIR" 2>/dev/null | wc -l) structure files in $STRUCTURES_DIR/"
if [ -f failed_downloads.txt ]; then
    echo "$(wc -l < failed_downloads.txt) PDB(s) still failed after retries -- see failed_downloads.txt"
    echo "Re-running this script later will only retry those (existing files are skipped)."
fi