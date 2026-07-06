#!/usr/bin/env bash
# Terminate a PrimeIntellect pod and clean up local pod tracking.
#
# Usage:
#   ./terminate-pod.sh [pod-id]
#
# If no pod-id is given, uses the active pod from .omo/active_pod_id.
# Env: PRIME_API_KEY

set -euo pipefail

POD_ID="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$POD_ID" && -f "$REPO_ROOT/.omo/active_pod_id" ]]; then
  POD_ID="$(cat "$REPO_ROOT/.omo/active_pod_id")"
fi

if [[ -z "$POD_ID" ]]; then
  echo "ERROR: no pod id provided and no active pod recorded" >&2
  exit 1
fi

if [[ -z "${PRIME_API_KEY:-}" ]]; then
  echo "ERROR: PRIME_API_KEY not set" >&2
  exit 2
fi

echo ">>> Terminating pod $POD_ID..."
prime pods terminate "$POD_ID" --output json 2>&1 | head -10 || {
  echo "WARN: terminate may have failed; pod may already be gone" >&2
}

rm -f "$REPO_ROOT/.omo/active_pod_id" "$REPO_ROOT/.omo/active_pod_type"
echo ">>> Done"