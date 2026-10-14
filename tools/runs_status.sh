#!/bin/bash
# One-call status of active runs (rows of .mem/runs.md with status "running"), <=20 lines.
# Batches here run through a driver (scripts/run_wp1.sh) with auto-named containers, so per host we
# report: containers up, jobs done/failed in done.txt, last driver log line, free disk.
# Usage: bash tools/runs_status.sh
cd "$(dirname "$0")/.." || exit 1
alias_of() { case "$1" in gpu15) echo GPUserver1;; gpu14) echo GPUserver2;; *) echo "$1";; esac; }

hosts=$(awk -F'|' '!/^#/ && /running/ {gsub(/ /,"",$2); print $2}' .mem/runs.md | sort -u)
[ -z "$hosts" ] && { echo "no running rows in .mem/runs.md"; exit 0; }

for h in $hosts; do
  rows=$(awk -F'|' -v h="$h" '!/^#/ && /running/ {gsub(/ /,"",$2); if ($2==h) {gsub(/^ +| +$/,"",$1); print $1}}' .mem/runs.md | tr '\n' ' ')
  echo "== $h ($(alias_of $h)) runs: $rows"
  ssh -o ConnectTimeout=8 -o BatchMode=yes "$(alias_of $h)" 'cd ~/netslice-drl 2>/dev/null || exit 3
    n=$(docker ps -q --filter ancestor=netslice-drl:latest | wc -l)
    drv=$(ls -t results/wp1_logs/driver_*.log 2>/dev/null | head -1)
    ok=$(grep -c " rc=0 " results/wp1_logs/done.txt 2>/dev/null)
    bad=$(grep -v " rc=0 " results/wp1_logs/done.txt 2>/dev/null | wc -l)
    echo "  containers up: $n | done.txt rc=0: $ok | rc!=0: $bad | disk free: $(df -h . | awk "NR==2{print \$4}")"
    [ -n "$drv" ] && echo "  $drv: $(tail -n 1 "$drv" | cut -c1-110)"
    [ -n "$drv" ] && grep -q "COMPLETE" "$drv" && echo "  FLAG: driver reports COMPLETE"
    [ "$n" -eq 0 ] && echo "  FLAG: no netslice containers running (finished or crashed?)"
    [ "$bad" -gt 0 ] && grep -v " rc=0 " results/wp1_logs/done.txt | tail -n 2 | sed "s/^/  FAILED: /"
    true' 2>&1 | head -n 8 || echo "  FLAG: ssh failed"
done | head -n 20
