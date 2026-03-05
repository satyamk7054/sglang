# GPT-OSS-120B FP8 MoE (`matmul_ogs`) Handoff

Date: 2026-03-05
Branch: `satyamk/moe-fp8`
Workspace: `/home/jobuser/oss_sglang/sglang`

## Goal
Determine whether GPT-OSS-120B routes through `matmul_ogs` for dynamic FP8 (`--quantization fp8`) instead of MXFP4, and implement fixes so it works.

---

## Initial finding
Before changes, FP8 did **not** go through `matmul_ogs` for GPT-OSS in the triton-kernel MoE path because:
- FP8 MoE runner did not support `triton_kernel` backend.
- `triton_kernels_moe.py` explicitly rejected FP8 mode.
- GPT-OSS auto backend logic was effectively biased toward MXFP4 conditions.

---

## Files changed

1. `python/sglang/srt/layers/quantization/fp8.py`
- Enabled `MoeRunnerBackend.TRITON_KERNELS` in `Fp8MoEMethod.create_moe_runner`.
- Added `TritonKernelsQuantInfo` path in `Fp8MoEMethod.apply` for FP8.
- Wired FP8 quant metadata (weights/scales/flags) for triton-kernel runner.
- Final runtime layout fix: pass transposed **views** (not `.contiguous()`) for FP8 MoE weights so `matmul_ogs` FP8 stride requirements are satisfied on Hopper.

2. `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py`
- Extended `TritonKernelsQuantInfo` with FP8 fields:
  - `use_fp8_w8a8`, `per_channel_quant`, `expert_map`
  - `w13_scale`, `w2_scale`, `a13_scale`, `a2_scale`, `block_shape`
- Forwarded those fields from runner core into triton-kernel MoE calls.

3. `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`
- Added FP8 handling instead of hard rejection.
- Added FP8 dtype checks.
- Added `PrecisionConfig(weight_scale=...)` for both GEMMs when FP8 is enabled.
- Kept unsupported constraints guarded (`a1_scale/a2_scale`, block shape, etc. where not yet supported).

4. `python/sglang/srt/server_args.py`
- GPT-OSS auto backend selection updated so explicit `--quantization fp8` can select `triton_kernel` (when applicable).
- MXFP4 auto branches now respect explicit quantization override instead of unconditionally following model metadata.

5. `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- Iterative loader fixes for fused MoE in FP8 + triton-kernel mode:
  - Corrected shard-dimension/transposed behavior for GPT-OSS fused MoE weight loading.
  - Prevented FP8 double-transpose in internal `_load_w13/_load_w2` triton-kernel paths.
  - Added FP8-aware shard-dimension override for `w13` in fused loader path.

6. `test/manual/layers/moe/test_moe_runners_1gpu.py`
- Added manual config entry `moe_runner_triton_kernel_fp8`.

---

## Runtime validation history (TP=4)

### Environment constraint
- Machine has 4 GPUs, so all meaningful validation switched to `--tp 4`.

### Major failures fixed in order
1. `invalid device ordinal` with `tp=8` (environment mismatch) → switched to `tp=4`.
2. Fused MoE weight-load shard failures (`_load_w2`, then `_load_w13`) in `layer.py` → fixed TP shard/transposed logic for FP8 triton-kernel.
3. CUDA graph warmup assert in triton kernels:
   - shape mismatch at `triton_kernel_fused_experts_with_bias`
   - fixed by weight layout handling in FP8 quant-info path.
4. `matmul_ogs` Hopper FP8 assertion:
   - `w` must be column-major for FP8 on capability < 10 path
   - fixed by passing transposed views (stride-compliant), not contiguous copies.

### Positive signal observed
- `/v1/completions` request returned successfully during one validated run while FP8 + triton-kernel path was active.

### Note on frequent `exit code 137`
- Many launches ended with `137` after child-process failures during iterative debugging.
- Some runs were intentionally killed during cleanup.
- During final iterations, process progressed much farther (weight load + graph capture) and served at least one completion successfully before shutdown/cleanup.

---

## Known current state
- Code now includes FP8 triton-kernel routing/plumbing and multiple loader/runtime fixes.
- The path has been exercised deeply in model load + graph capture + generation.
- Because previous runs involved repeated iterative relaunches, re-run a clean launch once from current HEAD to reconfirm stability end-to-end.

---

## Next-session quickstart

### 1) Activate env
```bash
source ../sglang_venv/bin/activate
```

### 2) Launch
```bash
python -m sglang.launch_server \
  --model-path /shared/public/elr-models/openai/gpt-oss-120b-bf16/ \
  --quantization fp8 \
  --moe-runner-backend auto \
  --tp 4 \
  --host 127.0.0.1 \
  --port 30000
```

### 3) Health check
```bash
curl -sS http://127.0.0.1:30000/health
```

### 4) Smoke completion
```bash
curl -sS http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"/shared/public/elr-models/openai/gpt-oss-120b-bf16/",
    "prompt":"Say hello in five words.",
    "max_tokens":16,
    "temperature":0.0
  }'
```

### 5) Optional: capture logs
```bash
python -m sglang.launch_server ... 2>&1 | tee /tmp/gpt-oss-fp8-tp4.log
```

---

## Open follow-ups
1. Add targeted automated test(s) for FP8 + triton-kernel + fused MoE loader orientation.
2. Add explicit runtime logging flag for backend/kernel route confirmation (e.g., one-time log when using `triton_kernel_fused_experts_with_bias` in FP8 mode).
3. Validate across multiple prompts/batch sizes and with CUDA graph enabled/disabled for robustness.

---

## Current modified files snapshot
```text
M python/sglang/srt/layers/moe/fused_moe_triton/layer.py
M python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py
M python/sglang/srt/layers/moe/moe_runner/triton_kernels.py
M python/sglang/srt/layers/quantization/fp8.py
M python/sglang/srt/server_args.py
M test/manual/layers/moe/test_moe_runners_1gpu.py
```
