# TensorForge CI Workflows

Two GitHub Actions workflows provide the CI signal for TensorForge:

| Workflow | File | Trigger | Runner | Cost |
| -------- | ---- | ------- | ------ | ---- |
| CPU Smoke | [`cpu-smoke.yml`](./cpu-smoke.yml) | every push to `main`, every PR into `main` | GitHub-hosted `ubuntu-latest` | free (public repo) / included minutes (private) |
| GPU Tests (PrimeIntellect) | [`gpu-tests.yml`](./gpu-tests.yml) | nightly cron `0 6 * * *` (UTC), plus `workflow_dispatch` | ephemeral H100 (→ A100 fallback) pod on PrimeIntellect | ~$0.50–$2 per run; see [`../docs/ci.md`](../docs/ci.md) |

## CPU Smoke (`cpu-smoke.yml`)

- **When**: every push to `main` and every PR targeting `main`.
- **What**: installs Bazelisk v1.20.0, runs `bazelisk build //hello:hello --config=cpu` plus a `bazelisk test //...` with `--test_tag_filters=cpu`. The test step is marked `continue-on-error: true` while the test suite is being written.
- **Why**: catches the cheap regressions first — bazelisk/MODULE.bazel/`.bazelrc` breakage, broken C++ toolchain on a clean machine, missing headers in transitive deps. Anything that doesn't need a GPU fails here in ~30s instead of costing a pod spin-up.
- **Cost**: included in GitHub-hosted runner minutes; effectively free for public repos.

## GPU Tests (`gpu-tests.yml`)

- **When**:
  - Nightly at 06:00 UTC (`cron: '0 6 * * *'`) so a breakage surfaces during work hours in NA/EU.
  - Manual via the **Run workflow** button on the Actions tab — useful when iterating on a CUDA-specific change.
- **What**:
  1. Installs the `prime` CLI.
  2. Queries `prime availability list` for the cheapest **Available** H100 80GB offer. If none, falls back to the cheapest Available A100 80GB. Aborts if neither is in stock (a stock-out is a real CI failure).
  3. `prime pods create --id <availability-id>` and parses the 32-char hex pod id out of the output.
  4. Polls `prime pods status` until the pod is `ACTIVE` (typically 60–120s, capped at 5 minutes).
  5. SSHs to the pod, installs `cuda-toolkit-12-2` + `g++-12`, drops bazelisk at `/usr/local/bin/bazelisk`, and creates `/data/tensorforge` (the image's `/data` is read-only for the `ubuntu` user out of the box).
  6. `rsync` of the repo (excluding `.git`, `bazel-*`, `.omo`, `public_key.pem`, `private_key.pem`, `.env`).
  7. SSHs in and runs `bazelisk test //... --config=sm80_sm90 --test_tag_filters=gpu`.
  8. Uploads `bazel-testlogs/` and `bazel-bin/` as an artifact regardless of pass/fail.
  9. **`prime pods terminate`** runs `if: always()` — orphaned pods keep billing until manually killed.
- **Cost**: ~5–10 minutes wall-clock on H100 PCIe. See [`../docs/ci.md`](../docs/ci.md) for the full pricing breakdown; nightly is roughly $30–$60/month depending on H100 spot pricing and run duration.

## Required GitHub configuration

- `PRIME_API_KEY` secret in the repo's Actions secrets. The GPU workflow
  references it as `${{ secrets.PRIME_API_KEY }}` and exports it as an
  environment variable inside the job. The secret is **never** echoed or
  logged.
- The PrimeIntellect account must have a payment method on file and the
  H100/A100 SKU enabled; otherwise nightly runs will fail at pod-provision
  time with an `ERROR: no GPU pods available` style exit.

## Concurrency

Both workflows set `concurrency.group` to `${{ github.workflow }}-${{ github.ref }}` with `cancel-in-progress: true`. This prevents two simultaneous pod spins for the same branch (which would double the GPU bill) and cancels stale PR pushes when a newer commit lands.

## Local parity

Before pushing, you can replicate the CI environment locally:

```bash
# CPU smoke (matches cpu-smoke.yml)
bazelisk build //hello:hello --config=cpu
bazelisk test  //...           --config=cpu --test_tag_filters=cpu

# GPU tests (matches gpu-tests.yml inside the pod)
prime pods create --id <availability-id>
prime pods status <pod_id>     # wait for ACTIVE
ssh ubuntu@<pod_ip>            # then the same apt + rsync + bazelisk commands
prime pods terminate <pod_id>
```

See [`../docs/ci.md`](../docs/ci.md) for the cost model and budget guardrails.