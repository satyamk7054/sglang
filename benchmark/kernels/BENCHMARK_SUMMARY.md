# Summary: MoE Quantization Benchmark Setup

## What You Asked For

You wanted to understand if **FP8** weights would be better than **MXFP4** weights for GPT-OSS MoE on H200, given that H200 doesn't have native FP4 support but does have native FP8 Tensor Core support.

## What I Created

I've created **3 benchmark scripts** with increasing levels of detail, plus comprehensive documentation:

### 1. **Simple Benchmark** (Recommended Starting Point)
**File**: `benchmark/kernels/simple_moe_quant_bench.py`

**What it does:**
- Uses SGLang's existing `fused_moe` kernel infrastructure
- Creates synthetic weights and quantizes them to FP8 or MXFP4
- Measures latency, throughput, and memory footprint
- Easy to run and understand

**Run it:**
```bash
cd /home/jobuser/oss_sglang/sglang

# GPT-OSS-120B configuration with TP=4 (typical for H200)
python benchmark/kernels/simple_moe_quant_bench.py --batch-size 1024 --tp-size 4

# Custom parameters
python benchmark/kernels/simple_moe_quant_bench.py \
    --batch-size 2048 \
    --hidden 7168 \
    --intermediate 24576 \
    --experts 128 \
    --topk 8 \
    --tp-size 4
```

### 2. **Detailed Benchmark** (Advanced)
**File**: `benchmark/kernels/moe_quant_comparison.py`

**What it does:**
- Directly tests the `matmul_ogs` kernel from `triton_kernels`
- More precise control over quantization and routing
- Tests the exact computation path used in production
- Requires `triton_kernels` package

**Run it:**
```bash
python benchmark/kernels/moe_quant_comparison.py \
    --model-config gpt-oss-120b \
    --tp-size 4 \
    --batch-size 1024
```

### 3. **Real Model Weights Benchmark**
**File**: `benchmark/kernels/bench_gpt_oss_moe_kernel.py`

**What it does:**
- Loads actual GPT-OSS model configuration
- Uses real weight shapes from the model
- Most realistic benchmark scenario
- Good for validating results

**Run it:**
```bash
python benchmark/kernels/bench_gpt_oss_moe_kernel.py \
    --model openai/gpt-oss-120b \
    --tp-size 4 \
    --batch-size 1024
```

### 4. **Documentation**
**File**: `benchmark/kernels/MoE_QUANT_BENCHMARK.md`

**Contains:**
- Complete explanation of FP8 vs MXFP4 trade-offs
- H200-specific considerations
- How to interpret results
- Integration with SGLang
- References to related benchmarks

## Key Findings (Expected)

### Why FP8 Should Be Faster on H200

1. **Native Hardware Support**
   - H200 (Hopper) has FP8 Tensor Cores
   - FP8 operations execute directly in hardware
   - No software conversion overhead

2. **MXFP4 Limitations on H200**
   - No native FP4 Tensor Cores
   - Requires runtime conversion to FP8/BF16
   - Software emulation adds latency

3. **Memory Trade-off**
   - FP8: 8 bits per parameter
   - MXFP4: ~4 bits per parameter (2x more efficient)
   - FP8 uses 2x more memory but should be faster

## How to Use the Results

### If FP8 is Faster (Expected)

**Speedup > 1.2x**: Clear winner for H200
```
✓ FP8 is 1.5x FASTER than MXFP4
  Use: python -m sglang.launch_server --model <fp8-model> --quantization w8a8_fp8
```

**Action Items:**
1. Consider quantizing GPT-OSS weights to FP8 format
2. Update deployment scripts to use FP8
3. Monitor memory usage vs batch size

### If MXFP4 is Faster (Unexpected)

**Speedup > 1.1x**: Investigate why
```
✗ MXFP4 is 1.3x FASTER than FP8
  Unexpected: H200 should favor FP8
```

**Possible reasons:**
1. Kernel not using FP8 Tensor Cores properly
2. Memory bandwidth bottleneck (FP8 moves 2x data)
3. Quantization overhead in FP8 path
4. Need kernel tuning for FP8

### If Results Are Close (<10% difference)

**Consider memory constraints:**
- MXFP4: Better for memory-limited scenarios
- FP8: Better for compute-limited scenarios

## Understanding the Benchmarks

### What Gets Measured

The benchmarks test the **MoE expert computation**:

```python
# Pseudo-code of what's being benchmarked
for each token:
    1. Select top-K experts based on router
    2. Compute gate_up projection: hidden @ w1.T + fused_swiglu
    3. Compute down projection: intermediate @ w2.T
    4. Combine expert outputs with router weights
```

### Key Metrics

1. **Latency (ms)**: Time per forward pass
   - Lower is better
   - Direct measure of inference speed

2. **Throughput (tokens/sec)**: Tokens processed per second
   - Higher is better
   - Batch size / latency

3. **Memory (MB/GB)**: Weight storage
   - Lower is better
   - Affects max batch size

## Existing SGLang Infrastructure

I built on top of existing benchmark infrastructure:

### Related Benchmarks You Can Also Run

1. **Fused MoE Tuning**
```bash
cd benchmark/kernels/fused_moe_triton
python tuning_fused_moe_triton.py \
    --model /shared/public/elr-models/openai/gpt-oss-120b-new/ \
    --tp-size 4 \
    --dtype auto
```

2. **Block-wise Quantization Tuning**
```bash
cd benchmark/kernels/quantization
python tuning_block_wise_kernel.py --tp-size 4
```

See their READMEs for details.

## Quick Start Guide

### Step 1: Choose a Benchmark

For first run, use the **simple benchmark**:
```bash
cd /home/jobuser/oss_sglang/sglang
python benchmark/kernels/simple_moe_quant_bench.py --batch-size 1024 --tp-size 4
```

### Step 2: Interpret Results

Look for output like:
```
COMPARISON
Format     Time (ms)    Throughput      Memory (MB)
FP8        2.345        436.78          9872.50
MXFP4      3.567        287.23          4936.25

✓ FP8 is 1.52x FASTER than MXFP4
```

### Step 3: Apply Results

Based on results, consider:
- Switching quantization format
- Tuning kernels for better performance
- Adjusting batch size based on memory

## Troubleshooting

### Import Errors

If you see:
```
ImportError: No module named 'triton_kernels'
```

**Solution**: The detailed benchmark requires `triton_kernels`. Use the simple benchmark instead.

### CUDA Errors

If you see CUDA OOM:
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size:
```bash
python benchmark/kernels/simple_moe_quant_bench.py --batch-size 512 --tp-size 4
```

### Unexpected Results

If MXFP4 is faster:
1. Verify you're on H200: `nvidia-smi`
2. Check compute capability: Should be ≥ 8.9
3. Profile with nsys to see what's happening

## Next Steps

1. **Run the simple benchmark** on your H200 node
2. **Check the results** - is FP8 faster?
3. **Read the full documentation** in `MoE_QUANT_BENCHMARK.md`
4. **Profile deeper** if needed with nsys/ncu
5. **Update deployment** based on findings

## Files Created

All files are in `benchmark/kernels/`:

```
benchmark/kernels/
├── simple_moe_quant_bench.py          # Easy to run, uses fused_moe
├── moe_quant_comparison.py            # Detailed, uses matmul_ogs directly
├── bench_gpt_oss_moe_kernel.py        # Uses real model config
├── MoE_QUANT_BENCHMARK.md             # Full documentation
└── BENCHMARK_SUMMARY.md               # This file
```

## Contact

If you see unexpected results or need help interpreting them, consider:
1. Profiling with `nsys profile --trace=cuda,nvtx python ...`
2. Checking kernel utilization with `ncu`
3. Comparing with existing tuned configs in `python/sglang/srt/layers/quantization/configs/`

The benchmarks should give you clear data on whether FP8 or MXFP4 is better for your H200 setup with GPT-OSS.
