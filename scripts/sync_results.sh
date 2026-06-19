#!/usr/bin/env bash
# Pull experiment results from both GPU servers into ./results/.
# Subdirectory layout mirrors the remote path so runs from different servers
# are kept separate:
#   results/gpu14/<agent>/<seed>/...
#   results/gpu15/<agent>/<seed>/...
#
# Usage: bash scripts/sync_results.sh
set -euo pipefail

mkdir -p results/gpu14 results/gpu15

for HOST in 10.26.110.14 10.26.110.15; do
    LABEL="gpu${HOST##*.}"   # gpu14 or gpu15
    echo "=== Syncing results from pedroamaral@$HOST → ./results/$LABEL/ ==="
    rsync -av \
          pedroamaral@$HOST:~/netslice-drl/results/ \
          "./results/$LABEL/"
    echo ""
done

echo "=== All results synced. ==="
echo "  results/gpu14/  ← 10.26.110.14"
echo "  results/gpu15/  ← 10.26.110.15"
