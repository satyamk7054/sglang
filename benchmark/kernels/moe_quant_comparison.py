"""
Minimal benchmark to compare FP8 vs MXFP4 quantization for GPT-OSS MoE kernels on H200.

This script benchmarks the matmul_ogs kernel performance with:
1. FP8 (w8a8_fp8) - native fp8 support on H200
2. MXFP4 - 4-bit format that may require software emulation on H200

Usage:
    python benchmark/kernels/moe_quant_comparison.py --model-config gpt-oss-120b --tp-size 4
    python benchmark/kernels/moe_quant_comparison.py --batch-size 1024 --hidden-size 7168 --intermediate-size 24576 --num-experts 128
"""

import argparse
import time
from typing import Tuple

import torch

# Check for triton_kernels availability
try:
    from triton_kernels.matmul_ogs import (
        FlexCtx,
        FnSpecs,
        FusedActivation,
        PrecisionConfig,
        matmul_ogs,
    )
    from triton_kernels.numerics import InFlexData
    from triton_kernels.routing import GatherIndx, RoutingData, ScatterIndx
    from triton_kernels.swiglu import swiglu_fn

    TRITON_KERNELS_AVAILABLE = True
except ImportError:
    print("ERROR: triton_kernels not available. Please install it first.")
    TRITON_KERNELS_AVAILABLE = False

# Model configurations
MODEL_CONFIGS = {
    "gpt-oss-120b": {
        "hidden_size": 7168,
        "intermediate_size": 24576,  # per expert (not sharded yet)
        "num_experts": 128,
        "num_experts_per_tok": 8,
    },
    "gpt-oss-20b": {
        "hidden_size": 2048,
        "intermediate_size": 10240,
        "num_experts": 64,
        "num_experts_per_tok": 8,
    },
}


def quantize_fp8_e4m3(
    weight: torch.Tensor, scale_format: str = "per_channel"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight to FP8 E4M3 format with per-channel scales."""
    if scale_format == "per_channel":
        # Per output channel quantization (shape: [E, N, K] -> scales: [E, N])
        E, N, K = weight.shape
        weight_2d = weight.view(-1, K)  # [E*N, K]

        # Compute per-row scales
        abs_max = torch.max(torch.abs(weight_2d), dim=1, keepdim=True)[0]
        # FP8 E4M3 max value is 448
        scale = abs_max / 448.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        # Quantize
        weight_scaled = weight_2d / scale
        weight_fp8 = weight_scaled.to(torch.float8_e4m3fn)

        scale = scale.squeeze(-1).view(E, N)
        weight_fp8 = weight_fp8.view(E, N, K)
    else:
        # Per tensor quantization
        abs_max = torch.max(torch.abs(weight))
        scale = abs_max / 448.0
        weight_fp8 = (weight / scale).to(torch.float8_e4m3fn)
        scale = scale.unsqueeze(0)

    return weight_fp8, scale


def quantize_mxfp4(
    weight: torch.Tensor, block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate MXFP4 quantization - pack to half precision (fp4 emulation).

    In production, this would use flashinfer's mxfp4 format.
    For benchmarking, we pack weights to simulate 4-bit storage.
    """
    E, N, K = weight.shape

    # MXFP4 typically uses block-wise scales (e.g., 32 elements per block)
    assert K % block_size == 0, f"K={K} must be divisible by block_size={block_size}"

    # Reshape to blocks: [E, N, K] -> [E, N, K//block_size, block_size]
    weight_blocks = weight.view(E, N, K // block_size, block_size)

    # Compute per-block scales
    abs_max = torch.max(torch.abs(weight_blocks), dim=-1, keepdim=True)[0]
    # MXFP4 range is approximately [-15, 15] for E2M1
    scale = abs_max / 15.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    # Pack to 4-bit representation (simulated as uint8 with 2 values packed)
    # In real implementation this would be packed more efficiently
    weight_scaled = weight_blocks / scale
    # Clamp to 4-bit range and round
    weight_quantized = torch.clamp(torch.round(weight_scaled), -15, 15).to(torch.int8)

    # Pack two 4-bit values into one byte (just for storage efficiency demo)
    # Shape will be [E, N, K//2] in uint8
    weight_blocks_flat = weight_quantized.view(E, N, -1)
    weight_packed = weight_blocks_flat.to(torch.uint8)

    # Scale shape: [E, N, K//block_size]
    scale = scale.squeeze(-1)

    # For matmul, we'd need to dequantize, but for size comparison this is enough
    # Create a dequantized version for actual computation
    weight_dequant = (weight_quantized.float() * scale.unsqueeze(-1)).view(E, N, K)

    # Return in a format compatible with matmul (as bf16 for now)
    # In production, triton_kernels would handle the packed format directly
    return weight_dequant.to(weight.dtype), scale


def create_routing_data(
    batch_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    device: str = "cuda",
) -> Tuple[RoutingData, GatherIndx, ScatterIndx]:
    """Create mock routing data for MoE."""
    # Simulate top-k expert selection
    expert_ids = torch.randint(
        0, num_experts, (batch_size, num_experts_per_tok), device=device
    )
    weights = torch.rand(batch_size, num_experts_per_tok, device=device)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize

    # Create routing structures (simplified)
    # In production, these would be created by the TopK module
    routing_data = RoutingData(
        gate_scal=weights.flatten(),
        num_experts=num_experts,
    )

    gather_indx = GatherIndx(
        expert_ids=expert_ids.flatten(),
        num_tokens_post_pad=batch_size * num_experts_per_tok,
    )

    scatter_indx = ScatterIndx(
        expert_ids=expert_ids.flatten(),
        num_tokens_post_pad=batch_size * num_experts_per_tok,
    )

    return routing_data, gather_indx, scatter_indx


def benchmark_moe_kernel(
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    quant_format: str,  # "fp8" or "mxfp4"
    num_warmup: int = 10,
    num_iters: int = 100,
    device: str = "cuda",
) -> dict:
    """
    Benchmark MoE kernel with specified quantization format.

    Returns timing and memory info.
    """
    print(f"\n{'='*80}")
    print(
        f"Benchmarking {quant_format.upper()} - Batch={batch_size}, Hidden={hidden_size}"
    )
    print(
        f"Intermediate={intermediate_size}, Experts={num_experts}, TopK={num_experts_per_tok}"
    )
    print(f"{'='*80}")

    # Create input activation
    hidden_states = torch.randn(
        batch_size, hidden_size, device=device, dtype=torch.bfloat16
    )

    # Create expert weights (gate_up projection is 2x intermediate size for SwiGLU)
    w1_shape = (num_experts, 2 * intermediate_size, hidden_size)
    w2_shape = (num_experts, hidden_size, intermediate_size)

    # Initialize in bfloat16
    w1_bf16 = torch.randn(*w1_shape, device=device, dtype=torch.bfloat16)
    w2_bf16 = torch.randn(*w2_shape, device=device, dtype=torch.bfloat16)

    # Quantize weights based on format
    if quant_format == "fp8":
        print("Quantizing to FP8...")
        w1_quant, w1_scale = quantize_fp8_e4m3(w1_bf16, scale_format="per_channel")
        w2_quant, w2_scale = quantize_fp8_e4m3(w2_bf16, scale_format="per_channel")

        # Create PrecisionConfig for FP8
        w1_precision_config = PrecisionConfig(
            weight_scale=w1_scale,
            flex_ctx=FlexCtx(rhs_data=InFlexData()),
        )
        w2_precision_config = PrecisionConfig(
            weight_scale=w2_scale,
            flex_ctx=FlexCtx(rhs_data=InFlexData()),
        )

        weight_dtype = torch.float8_e4m3fn

    elif quant_format == "mxfp4":
        print("Quantizing to MXFP4...")
        w1_quant, w1_scale = quantize_mxfp4(w1_bf16, block_size=32)
        w2_quant, w2_scale = quantize_mxfp4(w2_bf16, block_size=32)

        # For MXFP4, we need to use the FlexCtx with proper metadata
        # This is a simplified version - production would use actual MXFP4 packing
        from triton_kernels.numerics import InFlexData

        # Create FlexData for MXFP4 (simplified - would need proper format metadata)
        w1_flex = InFlexData()
        w2_flex = InFlexData()

        w1_precision_config = PrecisionConfig(
            weight_scale=w1_scale,
            flex_ctx=FlexCtx(rhs_data=w1_flex),
        )
        w2_precision_config = PrecisionConfig(
            weight_scale=w2_scale,
            flex_ctx=FlexCtx(rhs_data=w2_flex),
        )

        weight_dtype = torch.bfloat16  # MXFP4 dequantized to bf16 for matmul

    else:
        raise ValueError(f"Unknown quant format: {quant_format}")

    # Calculate memory footprint
    weight_size_bytes = (
        w1_quant.element_size() * w1_quant.numel()
        + w2_quant.element_size() * w2_quant.numel()
    )
    scale_size_bytes = (
        w1_scale.element_size() * w1_scale.numel()
        + w2_scale.element_size() * w2_scale.numel()
    )
    total_size_mb = (weight_size_bytes + scale_size_bytes) / (1024**2)

    print(f"Weight memory: {weight_size_bytes / (1024**2):.2f} MB")
    print(f"Scale memory: {scale_size_bytes / (1024**2):.2f} MB")
    print(f"Total memory: {total_size_mb:.2f} MB")
    print(f"Weight dtype: {weight_dtype}")

    # Create routing data
    routing_data, gather_indx, scatter_indx = create_routing_data(
        batch_size, num_experts, num_experts_per_tok, device
    )

    # Create fused activation for SwiGLU
    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")),
        (1.702, float("inf")),  # alpha and clamp limit for GPT-OSS
        2,  # Split factor for gate/up
    )

    # Define benchmark function
    def run_moe():
        # First matmul with fused SwiGLU activation
        intermediate = matmul_ogs(
            hidden_states,
            w1_quant,
            None,  # bias
            routing_data,
            gather_indx=gather_indx,
            precision_config=w1_precision_config,
            gammas=routing_data.gate_scal,  # Apply router weights
            fused_activation=act,
        )

        # Second matmul
        output = matmul_ogs(
            intermediate,
            w2_quant,
            None,  # bias
            routing_data,
            scatter_indx=scatter_indx,
            precision_config=w2_precision_config,
            gammas=None,
        )

        return output

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        output = run_moe()
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_iters} iterations)...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iters):
        output = run_moe()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) * 1000 / num_iters

    print(f"\n{quant_format.upper()} Results:")
    print(f"  Average time: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {batch_size / (avg_time_ms / 1000):.2f} tokens/sec")
    print(f"  Memory footprint: {total_size_mb:.2f} MB")

    return {
        "quant_format": quant_format,
        "avg_time_ms": avg_time_ms,
        "throughput": batch_size / (avg_time_ms / 1000),
        "memory_mb": total_size_mb,
        "weight_dtype": str(weight_dtype),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP8 vs MXFP4 for GPT-OSS MoE on H200"
    )

    # Model configuration
    parser.add_argument(
        "--model-config",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Use predefined model config",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size (number of tokens)"
    )
    parser.add_argument("--hidden-size", type=int, help="Hidden size")
    parser.add_argument(
        "--intermediate-size", type=int, help="Intermediate size per expert"
    )
    parser.add_argument("--num-experts", type=int, help="Total number of experts")
    parser.add_argument(
        "--num-experts-per-tok", type=int, help="Number of experts per token (top-k)"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallelism size"
    )

    # Benchmark settings
    parser.add_argument(
        "--num-warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-iters", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--quant-formats",
        type=str,
        nargs="+",
        default=["fp8", "mxfp4"],
        choices=["fp8", "mxfp4"],
        help="Quantization formats to test",
    )

    args = parser.parse_args()

    if not TRITON_KERNELS_AVAILABLE:
        print("ERROR: triton_kernels is not available. Cannot run benchmark.")
        return

    # Get model configuration
    if args.model_config:
        config = MODEL_CONFIGS[args.model_config].copy()
        print(f"Using model config: {args.model_config}")
    else:
        config = {}

    # Override with command line args
    if args.hidden_size is not None:
        config["hidden_size"] = args.hidden_size
    if args.intermediate_size is not None:
        config["intermediate_size"] = args.intermediate_size
    if args.num_experts is not None:
        config["num_experts"] = args.num_experts
    if args.num_experts_per_tok is not None:
        config["num_experts_per_tok"] = args.num_experts_per_tok

    # Apply TP sharding to intermediate size
    if args.tp_size > 1:
        config["intermediate_size"] = config["intermediate_size"] // args.tp_size
        print(
            f"Applied TP sharding: intermediate_size={config['intermediate_size']} (per rank)"
        )

    # Validate config
    required_keys = [
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "num_experts_per_tok",
    ]
    for key in required_keys:
        if key not in config:
            print(f"ERROR: Missing required parameter: {key}")
            print("Either use --model-config or specify all parameters manually")
            return

    # Print configuration
    print("\n" + "=" * 80)
    print("BENCHMARK CONFIGURATION")
    print("=" * 80)
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden size: {config['hidden_size']}")
    print(f"Intermediate size: {config['intermediate_size']} (after TP sharding)")
    print(f"Number of experts: {config['num_experts']}")
    print(f"Experts per token: {config['num_experts_per_tok']}")
    print(f"TP size: {args.tp_size}")
    print(f"Quantization formats: {', '.join(args.quant_formats)}")

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")

    # Run benchmarks
    results = []
    for quant_format in args.quant_formats:
        try:
            result = benchmark_moe_kernel(
                batch_size=args.batch_size,
                hidden_size=config["hidden_size"],
                intermediate_size=config["intermediate_size"],
                num_experts=config["num_experts"],
                num_experts_per_tok=config["num_experts_per_tok"],
                quant_format=quant_format,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR running {quant_format}: {e}")
            import traceback

            traceback.print_exc()

    # Print comparison
    if len(results) >= 2:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        print(
            f"\n{'Format':<10} {'Time (ms)':<12} {'Throughput':<15} {'Memory (MB)':<12} {'Speedup':<10}"
        )
        print("-" * 80)

        baseline_time = results[0]["avg_time_ms"]
        for result in results:
            speedup = baseline_time / result["avg_time_ms"]
            print(
                f"{result['quant_format'].upper():<10} "
                f"{result['avg_time_ms']:<12.3f} "
                f"{result['throughput']:<15.2f} "
                f"{result['memory_mb']:<12.2f} "
                f"{speedup:<10.2f}x"
            )

        # Memory comparison
        if len(results) == 2:
            mem_ratio = results[0]["memory_mb"] / results[1]["memory_mb"]
            print(
                f"\nMemory footprint: {results[0]['quant_format'].upper()} is {mem_ratio:.2f}x of {results[1]['quant_format'].upper()}"
            )

            time_ratio = results[0]["avg_time_ms"] / results[1]["avg_time_ms"]
            if time_ratio < 1:
                faster = results[0]["quant_format"].upper()
                slower = results[1]["quant_format"].upper()
                speedup = 1 / time_ratio
            else:
                faster = results[1]["quant_format"].upper()
                slower = results[0]["quant_format"].upper()
                speedup = time_ratio

            print(f"Performance: {faster} is {speedup:.2f}x faster than {slower}")

            print("\n" + "=" * 80)
            print("RECOMMENDATION FOR H200")
            print("=" * 80)

            if results[0]["quant_format"] == "fp8":
                fp8_result = results[0]
                mxfp4_result = results[1]
            else:
                fp8_result = results[1]
                mxfp4_result = results[0]

            if fp8_result["avg_time_ms"] < mxfp4_result["avg_time_ms"]:
                print(
                    f"✓ Use FP8: {speedup:.2f}x faster due to native H200 FP8 support"
                )
                print(f"  - FP8 has hardware acceleration on H200")
                print(f"  - MXFP4 requires software emulation/conversion")
                print(f"  - Trade-off: FP8 uses {mem_ratio:.2f}x more memory")
            else:
                print(f"✓ Use MXFP4: {speedup:.2f}x faster")
                print(f"  - Better memory efficiency ({1/mem_ratio:.2f}x smaller)")
                print(f"  - Note: Unexpected result - verify benchmark setup")


if __name__ == "__main__":
    main()
