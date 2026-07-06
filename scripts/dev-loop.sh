#!/usr/bin/env bash
# Sync local source to GPU pod and run a bazel command.
# Use this to keep code in sync between local edits and the remote pod.
#
# Usage:
#   ./dev-loop.sh <pod-id> <bazel-target> [extra-bazel-args...]
#
# Examples:
#   ./dev-loop.sh $POD_ID //hello:hello --config=sm80_sm90
#   ./dev-loop.sh $POD_ID test //tests:cpu_smoke --config=cpu
#
# Env:
#   PRIME_API_KEY (required)
#   PRIME_DISK_ID (optional, default: tensorforge-cache)

set -euo pipefail

POD_ID="${1:?usage: dev-loop.sh <pod-id> <bazel-target> [args...]}"
shift
BAZEL_CMD="${1:?usage: dev-loop.sh <pod-id> <bazel-target> [args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="/data/tensorforge"

echo ">>> Syncing local code to pod $POD_ID:$WORK_DIR ..."
prime pods ssh "$POD_ID" -- "mkdir -p $WORK_DIR" 2>&1 | grep -v "Warning" || true

# Use rsync if available, else scp+tarball
if command -v rsync >/dev/null; then
  prime pods scp "$POD_ID:$WORK_DIR/" "$REPO_ROOT/" 2>&1 | tail -1 || true
  rsync -a --delete --exclude='.git' --exclude='bazel-*' --exclude='.omo' --exclude='public_key.pem' "$REPO_ROOT/" "root@$(prime pods ssh "$POD_ID" -- 'echo $PRIME_POD_HOST' 2>/dev/null || echo "<pod>"):$WORK_DIR/" || {
    echo "WARN: rsync over SSH not directly supported; falling back to tarball"
    prime pods ssh "$POD_ID" -- "rm -rf $WORK_DIR && mkdir -p $WORK_DIR"
    tar -C "$REPO_ROOT" --exclude='.git' --exclude='bazel-*' --exclude='.omo' --exclude='public_key.pem' -czf - . | \
      prime pods ssh "$POD_ID" -- "tar -xzf - -C $WORK_DIR"
  }
else
  prime pods ssh "$POD_ID" -- "rm -rf $WORK_DIR && mkdir -p $WORK_DIR"
  tar -C "$REPO_ROOT" --exclude='.git' --exclude='bazel-*' --exclude='.omo' --exclude='public_key.pem' -czf - . | \
    prime pods ssh "$POD_ID" -- "tar -xzf - -C $WORK_DIR"
fi

echo ">>> Running on pod: bazel $BAZEL_CMD $*"
prime pods ssh "$POD_ID" -- "cd $WORK_DIR && bazel $BAZEL_CMD $*"