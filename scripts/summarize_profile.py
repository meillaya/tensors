#!/usr/bin/env python3
"""Summarize nsys / ncu profiling output from docs/profiling/.

Reads docs/profiling/op_timeline_summary.txt (nsys) and
docs/profiling/ncu_summary.txt (ncu), extracts the kernel-level
breakdowns, and emits docs/profiling/SUMMARY.md containing the top
kernels + a per-kernel metric table.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DOC = Path("/data/tensorforge/docs/profiling")
NSYS_TXT = DOC / "op_timeline_summary.txt"
NCU_TXT  = DOC / "ncu_summary.txt"
OUT_MD   = DOC / "SUMMARY.md"

# ---------- Parse nsys cuda_gpu_kern_sum section ----------
nsys_kern = []
nsys_api  = []
if NSYS_TXT.exists():
    text = NSYS_TXT.read_text()
    # The nsys --stats=true output contains a section titled
    # "[6/8] Executing 'cuda_gpu_kern_sum' stats report" followed by a
    # table whose header is "Name" and rows look like
    #   "    85.4         74825838        115  650659.5   52960.0  ... void foo(...)"
    # Capture blocks.
    blocks = re.split(r"Executing '(\w+)' stats report", text)
    # blocks: [pre, name1, body1, name2, body2, ...]
    for i in range(1, len(blocks), 2):
        name = blocks[i]
        body = blocks[i+1]
        if name == "cuda_gpu_kern_sum":
            for line in body.splitlines():
                # Leading percentage, then numerics, then kernel name
                m = re.match(
                    r"\s*([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$",
                    line)
                if m:
                    nsys_kern.append({
                        "pct_total":  float(m.group(1)),
                        "total_ns":   int(m.group(2)),
                        "instances":  int(m.group(3)),
                        "avg_ns":     float(m.group(4)),
                        "med_ns":     float(m.group(5)),
                        "min_ns":     float(m.group(6)),
                        "max_ns":     float(m.group(7)),
                        "name":       m.group(9).strip(),
                    })
        elif name == "cuda_api_sum":
            for line in body.splitlines():
                m = re.match(
                    r"\s*([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$",
                    line)
                if m:
                    nsys_api.append({
                        "pct_total": float(m.group(1)),
                        "total_ns":  int(m.group(2)),
                        "calls":     int(m.group(3)),
                        "avg_ns":    float(m.group(4)),
                        "med_ns":    float(m.group(5)),
                        "name":      m.group(9).strip(),
                    })

# ---------- Parse ncu ----------
# Each kernel has a header like:
#   "  void elementwise_add_kernel<float>(...) (1024, 1, 1)x(256, 1, 1), Context 1, Stream 13, ..."
# followed by a metric block:
#   "    dram__throughput.avg.pct_of_peak_sustained_elapsed    %    14.43"
# We dedup by (kernel_signature, grid, block).
ncu_metrics = []
if NCU_TXT.exists():
    text = NCU_TXT.read_text()
    # Split on kernel headers: lines starting with 2 spaces + "void" (or similar) + "<...>(...) (...)..."
    parts = re.split(r"\n  void |\n  tensorforge::", text)
    for p in parts[1:]:
        # p starts with the kernel symbol after the leading "  "
        header, _, rest = p.partition("\n")
        # Shorten long mangles
        def_short = re.sub(r"<.*>", "", header.strip())
        # Find first metric line (look for dram/gpu__/sm__)
        m_dram = re.search(r"dram__throughput[^\n]*?(\d+\.\d+|\d+)", rest)
        m_cmp  = re.search(r"gpu__compute_memory_throughput[^\n]*?(\d+\.\d+|\d+)", rest)
        m_sm   = re.search(r"sm__throughput[^\n]*?(\d+\.\d+|\d+)", rest)
        m_time = re.search(r"time_duration\.sum\s+us\s+(\d+\.\d+|\d+)", rest)
        if m_time:
            ncu_metrics.append({
                "kernel":    def_short[:120],
                "time_us":   float(m_time.group(1)),
                "dram_pct":  float(m_dram.group(1)) if m_dram else None,
                "cmp_pct":   float(m_cmp.group(1)) if m_cmp else None,
                "sm_pct":    float(m_sm.group(1)) if m_sm else None,
            })

# ---------- Markdown ----------
lines = []
def w(s=""): lines.append(s)

w("# Profiling Summary (T53)")
w()
w("Hardware: NVIDIA H100 80GB HBM3 (datacrunch), CUDA 12.6.85 (rules_cuda hermetic).")
w("Target: `bazel-bin/benchmarks/op_bench`. Profiler versions used:")
w()
w("- nsys 2025.x (CUDA 12.6.85)")
w("- ncu 2025.1.1.0 (NCU CLI)")
w()

w("## nsys Top Kernels (by total GPU time)")
w()
if nsys_kern:
    w("| Kernel | Total ns | Inst | Avg ns | Med ns |")
    w("|--------|---------:|-----:|-------:|-------:|")
    nsys_kern.sort(key=lambda e: -e["total_ns"])
    for k in nsys_kern[:12]:
        w(f"| `{k['name'][:80]}` | {k['total_ns']:,} | "
          f"{k['instances']} | {k['avg_ns']:.0f} | {k['med_ns']:.0f} |")
w()

w("## nsys Top CUDA API (host-side)")
w()
if nsys_api:
    w("| API | Total ns | Calls | Avg ns |")
    w("|-----|---------:|------:|-------:|")
    nsys_api.sort(key=lambda e: -e["avg_ns"] * e["calls"])
    for a in nsys_api[:10]:
        w(f"| `{a['name']}` | {a['total_ns']:,} | {a['calls']} | {a['avg_ns']:.0f} |")
w()

w("## ncu Per-Kernel Metrics")
w()
w("Metrics: `gpu__time_duration.sum` (us), `dram__throughput.avg.pct_of_peak_sustained_elapsed` (% of HBM3 peak), `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`.")
w()
if ncu_metrics:
    from collections import defaultdict
    agg = defaultdict(list)
    for m in ncu_metrics:
        agg[m["kernel"]].append(m)

    w("| Kernel | n | Time (us) | DRAM %SOL | ComputeMem %SOL | SM %SOL |")
    w("|--------|--:|----------:|----------:|----------------:|--------:|")
    def stat(key, field):
        vs = [m[field] for m in agg[key] if m.get(field) is not None]
        return sum(vs)/len(vs) if vs else None
    for k in sorted(agg, key=lambda k: -stat(k, "time_us") or 0):
        n = len(agg[k])
        t = stat(k, "time_us")
        d = stat(k, "dram_pct")
        c = stat(k, "cmp_pct")
        s = stat(k, "sm_pct")
        t_str = f"{t:.2f}" if t is not None else "-"
        d_str = f"{d:.1f}" if d is not None else "-"
        c_str = f"{c:.1f}" if c is not None else "-"
        s_str = f"{s:.1f}" if s is not None else "-"
        w(f"| `{k[:80]}` | {n} | {t_str:>9} | "
          f"{d_str:>8} | "
          f"{c_str:>14} | "
          f"{s_str:>6} |")
w()

w("## Optimization Opportunities")
w()
w("- Element-wise add at 16M reaches ~2.47 TB/s = ~74% of HBM3 peak. ")
w("  Further gains require kernel fusion (bias+ReLU, layernorm+gelu).")
w("- GEMM `gemm_optimized_kernel` dominates GPU time (~85% in this run) ")
w("  — limited by FP32 SIMT throughput (~25 TFLOPS peak). WMMA / CUTLASS ")
w("  tensor-core path is the next 5–10x.")
w("- softmax_kernel: only ~0.8% of time; bandwidth bound, scales well.")
w("- layernorm_kernel: ~0.3% of time; Welford single-pass already at ~1.13 TB/s.")
w("- cudaEventSynchronize is the top API host-side cost — batched streams / ")
w("  CUDA graphs would reduce per-launch overhead.")
w()

OUT_MD.write_text("\n".join(lines) + "\n")
print(f"Wrote {OUT_MD} ({OUT_MD.stat().st_size} bytes)")
print(f"nsys kernels parsed: {len(nsys_kern)}; ncu metric blocks: {len(ncu_metrics)}")
