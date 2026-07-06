# TensorForge CI Cost Model

Two workflows, two very different cost profiles. The CPU smoke job is
free; the GPU tests job is the only line item on the PrimeIntellect bill.

## `cpu-smoke.yml` — GitHub-hosted runner

- **Runner**: `ubuntu-latest` (currently `ubuntu-24.04`) provided by
  GitHub Actions.
- **Wall-clock**: 1–2 minutes for a clean build, ~30s on warm cache.
- **Cost**:
  - **Public repo**: $0. GitHub waives runner minutes for public
    repositories.
  - **Private repo**: charged against the org's monthly included
    minutes; `ubuntu-latest` costs 1× multiplier. A 90-second job is
    0.025 minutes, well under any free tier.

The cache step (`actions/cache@v4` keyed on `MODULE.bazel` +
`.bazelrc` + `.bazelversion`) keeps the warm path under 30 seconds —
Bazel only has to re-compile the targets that actually changed.

## `gpu-tests.yml` — PrimeIntellect ephemeral pod

The workflow provisions a single pod, runs the test suite, and
terminates it. There are no idle runners; you only pay for the wall
time the pod exists.

### Per-run cost

PrimeIntellect pricing is per-GPU-second and varies by region, GPU
type, and provider (Lambda, CoreWeave, etc.). The order of magnitude:

| GPU | Typical on-demand price | 5 min cost | 10 min cost |
| --- | ----------------------- | ---------- | ----------- |
| H100 PCIe 80GB | ~$2.35/hr | ~$0.20 | ~$0.40 |
| H100 SXM 80GB  | ~$3.40/hr | ~$$0.28 | ~$0.57 |
| A100 80GB      | ~$1.50/hr | ~$0.13 | ~$0.25 |

For the workflow to stay under **$2 per nightly run**, the pod needs
to wrap up in roughly:

```
$2 / $2.35/hr * 60 min/hr ≈ 51 minutes  (H100 PCIe)
$2 / $3.40/hr * 60 min/hr ≈ 35 minutes  (H100 SXM)
```

In practice the build + test cycle is 5–10 minutes (Bazel warms its
cache via `bazelisk test //...` from scratch), so the realistic
per-run number is **$0.20–$0.60 on H100 PCIe**.

### Monthly budget

- **Nightly only**, 30 days/month, $0.50 average: **~$15/month**.
- **Nightly + 4 manual triggers/week** (~$2 each): **~$48/month**.
- **Conservative cap** (15 runs/month × $2 worst case): **~$30/month**.

Recommendation: set a billing alert on the PrimeIntellect account at
**$60/month**. If the bill climbs, the first place to look is
overlapping nightly + manual runs; the workflow's `concurrency` group
should normally prevent this, but a developer running `workflow_dispatch`
just as the cron fires can still slip through.

### Where the cost comes from

- **Pod provisioning**: ~60–120s. PrimeIntellect bills from pod start
  to terminate, so this is the "tax" on every run.
- **CUDA toolkit + g++-12 install**: ~30–60s. Apt pulls a few hundred MB
  on first run; cached on subsequent runs if you bake a custom image
  (out of scope here).
- **Bazel cold build**: ~3–5 minutes for `//hello:hello` with
  rules_cuda + the hermetic CUDA 12.6.3 redistributable. Dominates
  total wall time on the first run.
- **Test execution**: seconds to a couple of minutes depending on the
  suite size. Currently negligible — the test suite is still being
  written.

### Hard limits enforced in CI

- `concurrency.cancel-in-progress: true` — never two simultaneous pods
  for the same branch.
- `timeout-minutes: 90` — job-level cap; if a pod hangs past 90 minutes
  the workflow aborts, terminating the pod as a side effect.
- `prime pods terminate` runs `if: always()`. **Never delete this
  step.** An orphaned pod continues to bill at the per-second rate until
  manually killed; even one runaway overnight can blow the monthly
  budget.

## Cost-control checklist

- [ ] `PRIME_API_KEY` is stored as a repo secret, never echoed in
      logs.
- [ ] PrimeIntellect account has a billing alert configured.
- [ ] Manual triggers are limited to maintainers (branch protection
      rule on `main` + required reviewers).
- [ ] The `concurrency` block in `gpu-tests.yml` is preserved — do
      not remove `cancel-in-progress: true`.
- [ ] The `terminate` step keeps `if: always()` — do not scope it
      to `if: success()`.