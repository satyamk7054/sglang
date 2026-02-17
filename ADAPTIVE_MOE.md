# Adaptive MoE Kernel Selection

## Overview

This implementation adds adaptive kernel selection for MoE (Mixture of Experts) layers based on batch size. It dynamically switches between two MoE kernel implementations:

1. **`triton`** kernel (sglang's fused_moe): Better for small batches
2. **`triton_kernels`** kernel (v340 triton_kernel_moe_forward): Better for large batches

## Performance Characteristics

Based on benchmark results for GPT-OSS-120B model:

| Batch Size | triton_kernels (ms) | triton (ms) | Winner |
|------------|---------------------|-------------|---------|
| 8          | 1.504              | 0.290       | triton  |
| 16         | 1.528              | 0.355       | triton  |
| 32         | 1.522              | 0.514       | triton  |
| 64         | 1.518              | 0.736       | triton  |
| 128        | 1.514              | 0.790       | triton  |
| 256        | 1.514              | 1.200       | triton  |
| 512        | 1.517              | 1.254       | triton  |
| 1024       | 1.519              | 1.365       | triton  |
| 2048       | 1.515              | 1.535       | similar |
| 4096       | 1.514              | 2.043       | triton_kernels |
| 8192       | 1.905              | 3.522       | triton_kernels |

**Crossover point**: ~1536-2048 batch size

## Usage

### Enable Adaptive MoE

To enable adaptive MoE kernel selection when starting the server:

```bash
python -m sglang.launch_server \
    --model-path /path/to/gpt-oss-120b \
    --enable-adaptive-moe
```

### Customize Batch Size Threshold

You can adjust the batch size threshold (default is 1536):

```bash
python -m sglang.launch_server \
    --model-path /path/to/gpt-oss-120b \
    --enable-adaptive-moe \
    --adaptive-moe-batch-threshold 2048
```

### Example with GPT-OSS

```bash
python -m sglang.launch_server \
    --model-path /shared/public/elr-models/openai/gpt-oss-120b-new/ \
    --enable-adaptive-moe \
    --adaptive-moe-batch-threshold 1536 \
    --tp 8
```

## How It Works

The implementation works at the quantization method level (`UnquantizedFusedMoEMethod`):

1. When `--enable-adaptive-moe` is set, two MoE runners are created:
   - `runner_triton`: Uses the `triton` backend
   - `runner_triton_kernels`: Uses the `triton_kernels` backend

2. During forward pass, the batch size is checked:
   - If `batch_size < threshold`: Use `triton` backend (faster for small batches)
   - If `batch_size >= threshold`: Use `triton_kernels` backend (faster for large batches)

3. The selection happens dynamically per forward pass, allowing the model to adapt to varying batch sizes in real-time.

## Architecture

### Modified Files

1. **`python/sglang/srt/server_args.py`**
   - Added `enable_adaptive_moe` flag (default: False)
   - Added `adaptive_moe_batch_threshold` parameter (default: 1536)

2. **`python/sglang/srt/layers/quantization/unquant.py`**
   - Modified `UnquantizedFusedMoEMethod.create_moe_runner()` to create both runners when adaptive mode is enabled
   - Modified `forward_cuda()` to select runner based on batch size

### Key Implementation Details

```python
# In create_moe_runner()
if use_adaptive:
    self.use_adaptive = True
    self.batch_size_threshold = adaptive_moe_batch_threshold
    self.runner_triton_kernels = MoeRunner(MoeRunnerBackend.TRITON_KERNELS, config)
    self.runner_triton = MoeRunner(MoeRunnerBackend.TRITON, config)

# In forward_cuda()
if self.use_adaptive:
    batch_size = x.shape[0]
    if batch_size >= self.batch_size_threshold:
        runner = self.runner_triton_kernels  # Large batch
    else:
        runner = self.runner_triton  # Small batch
```

## Benchmarking

To run benchmarks and verify performance:

```bash
# Run the existing kernel benchmark
cd /home/jobuser/oss_sglang/sglang
source /home/jobuser/oss_sglang/sglang_venv/bin/activate

python benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py \
    --model /shared/public/elr-models/openai/gpt-oss-120b-new/
```

## Benefits

1. **Automatic Optimization**: No need to manually tune which kernel to use
2. **Dynamic Adaptation**: Handles varying batch sizes efficiently
3. **Best of Both Worlds**: Get optimal performance across all batch sizes
4. **Low Overhead**: Selection happens once per forward pass with minimal cost

## Future Improvements

1. Make threshold configurable per model (auto-tuning)
2. Add warmup profiling to automatically determine optimal threshold
3. Support for other MoE models beyond GPT-OSS
4. Metrics/logging for kernel selection decisions

## Testing

A test script is provided to verify the implementation:

```bash
cd /home/jobuser/oss_sglang/sglang
source /home/jobuser/oss_sglang/sglang_venv/bin/activate
python test_adaptive_moe.py
```

This will verify:
- Server arguments are correctly parsed
- Both MoE runners can be created
- The implementation is working correctly
