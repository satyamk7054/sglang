# MoE Quantization Benchmark: FP8 vs MXFP4 for GPT-OSS on H200

## Overview

This benchmark helps you determine whether to use **FP8** or **MXFP4** quantization for GPT-OSS MoE layers on H200 GPUs.

### Key Question
**Does H200's native FP8 support make FP8 faster than MXFP4 despite FP4 being more memory-efficient?**

### Background

- **MXFP4**: 4-bit floating point format (2x more memory efficient than FP8)
  - Requires software conversion/emulation on H200 (no native HW support)
  - Used in default GPT-OSS quantized models (`openai/gpt-oss-120b`)

- **FP8**: 8-bit floating point format
  - Native Tensor Core support on H200 (Hopper architecture)
  - 2x memory overhead vs MXFP4
  - May offer better throughput due to hardware acceleration

### What Gets Benchmarked

The benchmark tests the **matmul_ogs** kernel (from `triton_kernels`) which is the core MoE computation:

1. **Gate/Up projection** (W1): `[batch, hidden] x [experts, 2*intermediate, hidden]` with fused SwiGLU
2. **Down projection** (W2): `[batch, intermediate] x [experts, hidden, intermediate]`

This is the hot path in GPT-OSS inference.

## Quick Start

### 1. Simple Benchmark (Recommended)

Uses existing SGLang infrastructure:

```bash
# GPT-OSS-120B with TP=4 (typical H200 setup)
python benchmark/kernels/simple_moe_quant_bench.py --batch-size 1024 --tp-size 4

# Custom configuration
python benchmark/kernels/simple_moe_quant_bench.py \
    --batch-size 2048 \
    --hidden 7168 \
    --intermediate 24576 \
    --experts 128 \
    --topk 8 \
    --tp-size 4
```

**What it does:**
- Creates mock expert weights in BF16
- Quantizes to FP8 (using SGLang's FP8 kernel)
- Quantizes to MXFP4 (using flashinfer if available)
- Runs `fused_moe` kernel for both formats
- Compares throughput and memory

### 2. Detailed Benchmark (Advanced)

Direct `matmul_ogs` kernel testing:

```bash
# Requires triton_kernels package
python benchmark/kernels/moe_quant_comparison.py --model-config gpt-oss-120b --tp-size 4
```

**What it does:**
- Directly calls `matmul_ogs` from `triton_kernels`
- More precise control over quantization parameters
- Tests routing, gather/scatter operations
- Requires `triton_kernels` package to be installed

## Expected Results on H200

### Hypothesis

✅ **FP8 should be faster** due to:
1. Native Tensor Core support for FP8 operations
2. No software conversion overhead
3. Better memory bandwidth utilization with native formats

❌ **MXFP4 drawbacks** on H200:
1. Requires runtime conversion from MXFP4 → FP8/BF16
2. No direct Tensor Core support for FP4 operations
3. Software emulation adds latency

### Trade-off

| Format | Performance | Memory | Use When |
|--------|------------|---------|----------|
| **FP8** | Faster (expected) | 2x of MXFP4 | Inference speed critical, sufficient VRAM |
| **MXFP4** | Slower (expected) | 2x more efficient | Memory-constrained, batch size limited by VRAM |

### Sample Output Interpretation

```
COMPARISON
================================================================================
Format     Time (ms)    Throughput      Memory (MB)
--------------------------------------------------------------------------------
FP8        2.345        436.78          9872.50
MXFP4      3.567        287.23          4936.25

ANALYSIS FOR H200
================================================================================
✓ FP8 is 1.52x FASTER than MXFP4
  Reason: H200 has native FP8 Tensor Core support
  Trade-off: FP8 uses 2.00x MORE memory (9872.5 MB vs 4936.2 MB)

RECOMMENDATION:
  → Use FP8 for better performance on H200
  → Native hardware support outweighs memory overhead
```

## GPT-OSS Model Configurations

### GPT-OSS-120B
```python
hidden_size = 7168
intermediate_size = 24576  # per expert before TP
num_experts = 128
num_experts_per_tok = 8
```

With TP=4:
- `intermediate_size = 24576 / 4 = 6144` per rank
- W1 shape: `[128, 2*6144, 7168]` = `[128, 12288, 7168]`
- W2 shape: `[128, 7168, 6144]`

### GPT-OSS-20B
```python
hidden_size = 2048
intermediate_size = 10240
num_experts = 64
num_experts_per_tok = 8
```

## Understanding the Code

### Key Components

1. **Quantization Functions**
   - `quantize_fp8_e4m3()`: Converts BF16 → FP8 with per-channel scales
   - `quantize_mxfp4()`: Converts BF16 → MXFP4 with block-wise scales

2. **Kernel Call**
   - Uses `matmul_ogs` from `triton_kernels` package
   - Handles expert routing, gather/scatter operations
   - Fused SwiGLU activation for gate/up projection

3. **Metrics**
   - **Latency**: Time per MoE forward pass (ms)
   - **Throughput**: Tokens processed per second
   - **Memory**: Weight + scale storage (MB)

### Existing Kernel Benchmarks

SGLang already has related benchmarks:

```bash
# Tune fused MoE kernel for specific model
cd benchmark/kernels/fused_moe_triton
python tuning_fused_moe_triton.py \
    --model /path/to/gpt-oss-120b \
    --tp-size 4 \
    --dtype auto  # or fp8_w8a8

# Benchmark block-wise quantization kernels
cd benchmark/kernels/quantization
python tuning_block_wise_kernel.py --tp-size 4
```

See:
- [`benchmark/kernels/fused_moe_triton/README.md`](../fused_moe_triton/README.md)
- [`benchmark/kernels/quantization/README.md`](../quantization/README.md)

## Interpreting Results

### When FP8 Wins (Expected)
```
FP8: 2.3ms, MXFP4: 3.5ms → FP8 is 1.5x faster
```
**Action**: Use FP8 quantization for GPT-OSS on H200
- Modify server args or model config to use `w8a8_fp8` instead of `mxfp4`

### When MXFP4 Wins (Unexpected)
```
FP8: 3.2ms, MXFP4: 2.8ms → MXFP4 is 1.14x faster
```
**Action**: Investigate why
- Check if FP8 quantization is properly enabled
- Verify Tensor Core utilization
- May indicate kernel tuning issues

### Memory vs Speed Trade-off
If results show:
- FP8: 1.3x faster but 2x more memory
- Current batch size limited by VRAM

**Consider**: MXFP4 might enable larger batch sizes, potentially better overall throughput

## Integration with SGLang

### Current Behavior

GPT-OSS models default to different backends:
- **Blackwell (B200)**: `flashinfer_mxfp4` (optimized MXFP4 kernel)
- **Hopper (H100/H200)**: `triton_kernel` with MXFP4
- **AMD (MI300X)**: `aiter` with MXFP4

See [`server_args.py`](../../python/sglang/srt/server_args.py#L1366-L1382)

### Switching to FP8 (if faster)

**Option 1**: Use FP8 quantized model
```bash
# Not yet available for GPT-OSS, would need to quantize yourself
python -m sglang.launch_server \
    --model your-fp8-quantized-gpt-oss \
    --quantization w8a8_fp8
```

**Option 2**: Force MoE backend
```bash
python -m sglang.launch_server \
    --model openai/gpt-oss-120b \
    --moe-runner-backend triton_kernel \
    --quantization w8a8_fp8  # if using FP8 weights
```

## Limitations

1. **MXFP4 Emulation**: The simple benchmark may not have full MXFP4 support and might fall back to BF16
2. **Synthetic Routing**: Uses random expert selection, not real workload patterns
3. **Single-GPU**: Doesn't test EP (expert parallelism) scenarios
4. **No E2E**: Tests kernel only, not full model inference

## Next Steps

1. **Run benchmark** on your H200 node
2. **Compare results** with expectations
3. **If FP8 is faster**: Consider quantizing GPT-OSS weights to FP8 format
4. **If MXFP4 is faster**: Investigate kernel tuning, verify H200 FP8 support

## References

- [Fused MoE Tuning Guide](../fused_moe_triton/README.md)
- [W8A8 Block Quantization](../quantization/README.md)
- [GPT-OSS Benchmarks](../../gpt_oss/README.md)
- [triton_kernels matmul_ogs](https://github.com/foundation-model-stack/triton-kernels)

## Questions?

If you see unexpected results or need help:
1. Check GPU model: `nvidia-smi` should show "H200"
2. Verify FP8 support: Check compute capability ≥ 8.9
3. Profile kernels: Use `nsys` or `ncu` for detailed analysis
