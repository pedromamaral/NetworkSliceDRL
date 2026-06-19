#!/usr/bin/env bash
# Phase 0 acceptance-criteria check
# Usage: bash scripts/check_gpu_servers.sh
set -euo pipefail

HOSTS=("pedroamaral@10.26.110.14" "pedroamaral@10.26.110.15")
PASS=0
FAIL=0

for HOST in "${HOSTS[@]}"; do
  echo "==> Checking $HOST ..."
  if ssh -o ConnectTimeout=10 -o BatchMode=yes "$HOST" nvidia-smi 2>&1; then
    echo "[PASS] $HOST – nvidia-smi OK"
    ((PASS++))
  else
    echo "[FAIL] $HOST – could not connect or nvidia-smi failed"
    ((FAIL++))
  fi
  echo ""
done

echo "============================"
echo "Results: $PASS passed, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
