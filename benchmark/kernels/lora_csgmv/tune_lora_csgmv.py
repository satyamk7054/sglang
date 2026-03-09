#!/usr/bin/env python3
"""
Offline auto-tuning script for LoRA CSGMV kernel block sizes.

Sweeps block size configurations for the shrink (lora_a) and expand (lora_b)
kernels, benchmarks each on your GPU, and saves the best configs as JSON files
that the server automatically picks up at startup.

Usage:
    # Tune for a specific model + rank combo:
    python3 benchmark/kernels/lora_csgmv/tune_lora_csgmv.py \
        --model Qwen/Qwen3-Embedding-0.6B \
        --max-lora-rank 64

    # Tune with explicit dimensions (no model download needed):
    python3 benchmark/kernels/lora_csgmv/tune_lora_csgmv.py \
        --hidden-size 1024 --intermediate-size 3072 \
        --max-lora-rank 64

    # Configs are saved to:
    #   python/sglang/srt/lora/triton_ops/configs/<triton_version>/

    # Server automatically loads them when using --lora-backend csgmv
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
import triton

from sglang.srt.lora.triton_ops.chunked_sgmv_expand import _chunked_lora_expand_kernel

# Import the actual kernels we're tuning
from sglang.srt.lora.triton_ops.chunked_sgmv_shrink import _chunked_lora_shrink_kernel
from sglang.srt.lora.triton_ops.lora_tuning_config import get_lora_config_file_name
from sglang.srt.utils import get_device_name


def get_shrink_configs() -> List[Dict[str, int]]:
    """Generate candidate block size configurations for the shrink kernel."""
    configs = []
    for block_n in [16, 32, 64]:
        for block_k in [64, 128, 256]:
            for num_warps in [4, 8]:
                for num_stages in [2, 3, 4, 5]:
                    configs.append(
                        {
                            "BLOCK_N": block_n,
                            "BLOCK_K": block_k,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                        }
                    )
    return configs


def get_expand_configs() -> List[Dict[str, int]]:
    """Generate candidate block size configurations for the expand kernel."""
    configs = []
    for block_n in [32, 64, 128]:
        for block_k in [16, 32, 64]:
            for num_warps in [4, 8]:
                for num_stages in [2, 3, 4, 5]:
                    configs.append(
                        {
                            "BLOCK_N": block_n,
                            "BLOCK_K": block_k,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                        }
                    )
    return configs


def get_chunk_sizes() -> List[int]:
    """Chunk sizes (BLOCK_M) to tune for."""
    return [16, 32, 64, 128]


def benchmark_shrink(
    S: int,
    K: int,
    N: int,
    num_slices: int,
    chunk_size: int,
    config: Dict[str, int],
    num_loras: int = 4,
    dtype: torch.dtype = torch.float16,
    num_iters: int = 200,
    warmup: int = 50,
) -> float:
    """Benchmark the shrink kernel with a given config. Returns median time in ms."""
    device = "cuda"

    # Create test data
    x = torch.randn(S, K, dtype=dtype, device=device)
    weights = torch.randn(num_loras, N, K, dtype=dtype, device=device)
    output = torch.empty(S, N, dtype=dtype, device=device)

    # Create batch info: evenly distribute S tokens across num_loras adapters
    num_segments = (S + chunk_size - 1) // chunk_size
    seg_indptr = (
        torch.arange(0, num_segments + 1, dtype=torch.int32, device=device) * chunk_size
    )
    seg_indptr[-1] = S  # last segment may be shorter
    weight_indices = (
        torch.arange(num_segments, dtype=torch.int32, device=device) % num_loras
    )
    lora_ranks = torch.full(
        (num_loras,), N // num_slices, dtype=torch.int32, device=device
    )
    permutation = torch.arange(S, dtype=torch.int32, device=device)

    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]

    grid = (triton.cdiv(N, BLOCK_N), num_segments)

    # Warmup
    for _ in range(warmup):
        _chunked_lora_shrink_kernel[grid](
            x=x,
            weights=weights,
            output=output,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            permutation=permutation,
            num_segs=num_segments,
            N=N,
            K=K,
            NUM_SLICES=num_slices,
            BLOCK_M=chunk_size,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=config.get("num_warps", 4),
            num_stages=config.get("num_stages", 2),
        )

    # Benchmark
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        _chunked_lora_shrink_kernel[grid](
            x=x,
            weights=weights,
            output=output,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            permutation=permutation,
            num_segs=num_segments,
            N=N,
            K=K,
            NUM_SLICES=num_slices,
            BLOCK_M=chunk_size,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=config.get("num_warps", 4),
            num_stages=config.get("num_stages", 2),
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def benchmark_expand(
    S: int,
    output_dim: int,
    max_rank: int,
    num_slices: int,
    chunk_size: int,
    config: Dict[str, int],
    max_slice_size: int,
    num_loras: int = 4,
    dtype: torch.dtype = torch.float16,
    num_iters: int = 200,
    warmup: int = 50,
) -> float:
    """Benchmark the expand kernel with a given config. Returns median time in ms."""
    device = "cuda"

    x = torch.randn(S, num_slices * max_rank, dtype=dtype, device=device)
    weights = torch.randn(num_loras, output_dim, max_rank, dtype=dtype, device=device)
    output = torch.zeros(S, output_dim, dtype=dtype, device=device)

    num_segments = (S + chunk_size - 1) // chunk_size
    seg_indptr = (
        torch.arange(0, num_segments + 1, dtype=torch.int32, device=device) * chunk_size
    )
    seg_indptr[-1] = S
    weight_indices = (
        torch.arange(num_segments, dtype=torch.int32, device=device) % num_loras
    )
    lora_ranks = torch.full((num_loras,), max_rank, dtype=torch.int32, device=device)
    scalings = torch.ones(num_loras, dtype=torch.float32, device=device)
    permutation = torch.arange(S, dtype=torch.int32, device=device)

    # For simple case: slice_offsets = [0, output_dim]
    # For qkv: [0, q_dim, q_dim+kv_dim, q_dim+2*kv_dim]
    slice_size = output_dim // num_slices
    slice_offsets = torch.tensor(
        [i * slice_size for i in range(num_slices)] + [output_dim],
        dtype=torch.int32,
        device=device,
    )

    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]

    grid = (triton.cdiv(max_slice_size, BLOCK_N), num_slices, num_segments)

    # Warmup
    for _ in range(warmup):
        _chunked_lora_expand_kernel[grid](
            x=x,
            weights=weights,
            output=output,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            permutation=permutation,
            num_segs=num_segments,
            scalings=scalings,
            slice_offsets=slice_offsets,
            NUM_SLICES=num_slices,
            OUTPUT_DIM=output_dim,
            MAX_RANK=max_rank,
            BLOCK_M=chunk_size,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=config.get("num_warps", 4),
            num_stages=config.get("num_stages", 2),
        )

    # Benchmark
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        output.zero_()
        start = time.perf_counter()
        _chunked_lora_expand_kernel[grid](
            x=x,
            weights=weights,
            output=output,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            permutation=permutation,
            num_segs=num_segments,
            scalings=scalings,
            slice_offsets=slice_offsets,
            NUM_SLICES=num_slices,
            OUTPUT_DIM=output_dim,
            MAX_RANK=max_rank,
            BLOCK_M=chunk_size,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=config.get("num_warps", 4),
            num_stages=config.get("num_stages", 2),
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]


def get_model_dims(args) -> Dict[str, int]:
    """Get model dimensions either from args or by loading the config."""
    if args.hidden_size and args.max_lora_rank:
        return {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size or args.hidden_size * 4,
            "max_lora_rank": args.max_lora_rank,
        }
    elif args.model:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        return {
            "hidden_size": config.hidden_size,
            "intermediate_size": getattr(
                config, "intermediate_size", config.hidden_size * 4
            ),
            "max_lora_rank": args.max_lora_rank,
        }
    else:
        raise ValueError("Provide either --model or --hidden-size + --max-lora-rank")


def tune_kernel(
    kernel_name: str,
    configs: List[Dict[str, int]],
    benchmark_fn,
    benchmark_kwargs: Dict[str, Any],
    chunk_sizes: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Tune a kernel across chunk sizes and config candidates."""
    best_configs = {}

    for chunk_size in chunk_sizes:
        print(f"\n  chunk_size={chunk_size}:")
        best_time = float("inf")
        best_config = None

        for i, config in enumerate(configs):
            try:
                t = benchmark_fn(
                    chunk_size=chunk_size, config=config, **benchmark_kwargs
                )
                if t < best_time:
                    best_time = t
                    best_config = config.copy()
                if (i + 1) % 20 == 0:
                    print(
                        f"    tested {i + 1}/{len(configs)} configs, best so far: {best_time:.4f} ms"
                    )
            except Exception as e:
                # Some configs may be invalid (e.g., BLOCK_K > K)
                continue

        if best_config is not None:
            best_configs[chunk_size] = best_config
            print(f"    best: {best_config} -> {best_time:.4f} ms")
        else:
            print(f"    no valid config found!")

    return best_configs


def save_config(configs: Dict[int, Dict], filename: str, output_dir: str):
    """Save tuned configs to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")
    print(f"Saved config to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune LoRA CSGMV kernel block sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, help="HuggingFace model name to infer dimensions"
    )
    parser.add_argument(
        "--hidden-size", type=int, help="Model hidden size (alternative to --model)"
    )
    parser.add_argument("--intermediate-size", type=int, help="Model intermediate size")
    parser.add_argument(
        "--max-lora-rank", type=int, required=True, help="Max LoRA rank"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=4096,
        help="Number of tokens for benchmarking (default: 4096)",
    )
    parser.add_argument(
        "--num-iters", type=int, default=200, help="Benchmark iterations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for config JSON files",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"]
    )
    args = parser.parse_args()

    dims = get_model_dims(args)
    hidden_size = dims["hidden_size"]
    intermediate_size = dims["intermediate_size"]
    max_rank = dims["max_lora_rank"]
    S = args.num_tokens
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # Default output dir: alongside the kernel code
    if args.output_dir is None:
        triton_version = triton.__version__
        version_dir = f"triton_{triton_version.replace('.', '_')}"
        args.output_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "..",
            "python",
            "sglang",
            "srt",
            "lora",
            "triton_ops",
            "configs",
            version_dir,
        )

    device_name = get_device_name()
    print(f"Device: {device_name}")
    print(f"Triton: {triton.__version__}")
    print(f"Hidden size: {hidden_size}, Intermediate size: {intermediate_size}")
    print(f"Max LoRA rank: {max_rank}, Num tokens: {S}, Dtype: {dtype}")
    print(f"Output dir: {args.output_dir}")

    chunk_sizes = get_chunk_sizes()

    # Collect all unique (kernel, K, R) combos to tune.
    # For a typical model, the LoRA layers are:
    #   qkv_proj: shrink(hidden_size, 3*rank), expand(qkv_out_dim, rank) with 3 slices
    #   o_proj:   shrink(head_dim*num_heads, rank), expand(hidden_size, rank)
    #   gate_up:  shrink(hidden_size, 2*rank), expand(2*intermediate_size, rank) with 2 slices
    #   down_proj: shrink(intermediate_size, rank), expand(hidden_size, rank)
    #
    # We tune for the unique dimension combos:
    tune_jobs = []

    # Shrink: (K=input_dim, N=num_slices*rank)
    shrink_dims = set()
    shrink_dims.add((hidden_size, 3 * max_rank, 3))  # qkv
    shrink_dims.add((hidden_size, max_rank, 1))  # o_proj (input side)
    shrink_dims.add((hidden_size, 2 * max_rank, 2))  # gate_up
    shrink_dims.add((intermediate_size, max_rank, 1))  # down_proj

    # Expand: (output_dim, rank, num_slices)
    expand_dims = set()
    expand_dims.add((hidden_size, max_rank, 1))  # o_proj output, down_proj output
    expand_dims.add(
        (intermediate_size, max_rank, 2)
    )  # gate_up (2 slices, each intermediate_size)

    # ---- Tune shrink kernels ----
    shrink_candidates = get_shrink_configs()
    for K, N, num_slices in sorted(shrink_dims):
        print(f"\n{'='*60}")
        print(f"Tuning SHRINK: K={K}, N={N}, num_slices={num_slices}")
        print(f"{'='*60}")

        best = tune_kernel(
            kernel_name="shrink",
            configs=shrink_candidates,
            benchmark_fn=lambda chunk_size, config, **kw: benchmark_shrink(
                S=S,
                K=K,
                N=N,
                num_slices=num_slices,
                chunk_size=chunk_size,
                config=config,
                dtype=dtype,
                num_iters=args.num_iters,
            ),
            benchmark_kwargs={},
            chunk_sizes=chunk_sizes,
        )

        filename = get_lora_config_file_name("shrink", K, N)
        save_config(best, filename, args.output_dir)

    # ---- Tune expand kernels ----
    expand_candidates = get_expand_configs()
    for output_dim, rank, num_slices in sorted(expand_dims):
        print(f"\n{'='*60}")
        print(
            f"Tuning EXPAND: output_dim={output_dim}, rank={rank}, num_slices={num_slices}"
        )
        print(f"{'='*60}")

        max_slice_size = output_dim // num_slices

        best = tune_kernel(
            kernel_name="expand",
            configs=expand_candidates,
            benchmark_fn=lambda chunk_size, config, **kw: benchmark_expand(
                S=S,
                output_dim=output_dim * num_slices,
                max_rank=rank,
                num_slices=num_slices,
                chunk_size=chunk_size,
                config=config,
                max_slice_size=max_slice_size,
                dtype=dtype,
                num_iters=args.num_iters,
            ),
            benchmark_kwargs={},
            chunk_sizes=chunk_sizes,
        )

        filename = get_lora_config_file_name("expand", output_dim, rank)
        save_config(best, filename, args.output_dir)

    print(f"\nDone! Configs saved to {args.output_dir}")
    print(f"The server will automatically use these configs with --lora-backend csgmv")


if __name__ == "__main__":
    main()
