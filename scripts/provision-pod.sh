#!/usr/bin/env bash
# Provision a PrimeIntellect GPU pod for TensorForge work.
# Uses H100 (default) or A100 as fallback. Attaches persistent disk if available.
#
# Usage:
#   ./provision-pod.sh [h100|a100]      # GPU type (default: h100)
#
# Output:
#   Writes POD_ID to stdout (last line)
#   Returns 0 on success, non-zero on failure
#
# Env:
#   PRIME_API_KEY (required) — PrimeIntellect API key
#   PRIME_DISK_ID (optional) — persistent disk to attach (default: tensorforge-cache)

set -euo pipefail

GPU_TYPE="${1:-h100}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DISK_ID="${PRIME_DISK_ID:-tensorforge-cache}"

if [[ -z "${PRIME_API_KEY:-}" ]]; then
  echo "ERROR: PRIME_API_KEY not set. Source .env or export it." >&2
  exit 1
fi

case "$GPU_TYPE" in
  h100) GPU_TYPE_FILTER="H100_80GB" ;;
  a100) GPU_TYPE_FILTER="A100_80GB" ;;
  *)
    echo "ERROR: unknown GPU type '$GPU_TYPE' (use h100 or a100)" >&2
    exit 1
    ;;
esac

echo ">>> Querying availability for $GPU_TYPE_FILTER..."
AVAIL_JSON="$(prime availability list --gpu-type "$GPU_TYPE_FILTER" --output json --plain)"

# Pick the cheapest available offer
PICK_ID="$(echo "$AVAIL_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
offers = [o for o in data.get('gpu_resources', []) if o.get('stock_status') == 'Available']
if not offers:
    sys.exit(2)
offers.sort(key=lambda o: o.get('price_value', 1e9))
print(offers[0]['id'])
")" || {
  echo "ERROR: no available $GPU_TYPE_FILTER pods" >&2
  exit 3
}

echo ">>> Selected offer: $PICK_ID"

# Try to attach disk (if it exists)
DISK_ARG=""
if prime disks list --output json 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    disks = data.get('disks', [])
    if any(d.get('id') == '$DISK_ID' for d in disks):
        print('FOUND')
    else:
        print('NOT_FOUND')
except Exception:
    print('NOT_FOUND')
" | grep -q "FOUND"; then
  DISK_ARG="--disks $DISK_ID"
  echo ">>> Will attach disk: $DISK_ID"
else
  echo ">>> No persistent disk named '$DISK_ID' (continuing without)"
fi

echo ">>> Provisioning pod..."
POD_JSON="$(prime pods create --gpu-type "$GPU_TYPE_FILTER" --gpu-count 1 $DISK_ARG --output json 2>&1)" || {
  echo "ERROR: pod creation failed" >&2
  echo "$POD_JSON" >&2
  exit 4
}

POD_ID="$(echo "$POD_JSON" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    pod = data.get('pod') or data
    print(pod.get('id') or pod.get('pod_id') or '')
except Exception as e:
    sys.stderr.write(f'parse error: {e}\n')
    sys.exit(5)
")"

if [[ -z "$POD_ID" ]]; then
  echo "ERROR: could not parse pod id from response" >&2
  echo "$POD_JSON" >&2
  exit 6
fi

echo ">>> Pod provisioned: $POD_ID"
echo ">>> Waiting for pod to be RUNNING..."

# Poll for readiness (max 3 min)
for i in $(seq 1 36); do
  STATE="$(prime pods list --output json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
for p in data.get('pods', []):
    if p.get('id') == '$POD_ID':
        print(p.get('status', 'UNKNOWN'))
        sys.exit(0)
print('NOT_FOUND')
")"
  if [[ "$STATE" == "RUNNING" ]]; then
    echo ">>> Pod is RUNNING (after ${i}*5s)"
    break
  fi
  if [[ "$STATE" == "NOT_FOUND" || "$STATE" == "ERROR" || "$STATE" == "STOPPED" ]]; then
    echo "ERROR: pod in unexpected state $STATE" >&2
    exit 7
  fi
  sleep 5
done

# Verify SSH works
echo ">>> Testing SSH access..."
if ! prime pods ssh "$POD_ID" -- 'echo "SSH_OK" && nvidia-smi -L | head -1' 2>&1 | grep -q "SSH_OK"; then
  echo "ERROR: SSH failed to pod $POD_ID" >&2
  exit 8
fi

# Save pod ID for later
echo "$POD_ID" > "$REPO_ROOT/.omo/active_pod_id"
echo "$GPU_TYPE" > "$REPO_ROOT/.omo/active_pod_type"

echo ">>> Pod $POD_ID is ready"
echo "$POD_ID"