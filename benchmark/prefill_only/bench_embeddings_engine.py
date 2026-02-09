"""
SGLang Embeddings Benchmark Script using Engine API

This script benchmarks SGLang's embeddings using the Engine API directly.
This is the only way to test the torch.Tensor format path (encoding_format=None or "tensor").

Features:
- Direct Engine API access (no HTTP overhead)
- Supports torch.Tensor return format for performance testing
- Accurate RPS control with separate sender and completion tracking
- Configurable RPS, duration, and batch sizes
- Real-time metrics and warnings when target RPS is not achieved

Usage:
- Update configuration variables at the top of the file
- Ensure SGLang server is running
- Run: python bench_embeddings_engine.py
"""

import asyncio
import time
from typing import Any, Dict, Optional

import numpy as np
from transformers import AutoTokenizer

from sglang import Engine

# Configure parameters
###############################################################################
# Benchmark Configuration
###############################################################################
TARGET_RPS_VALUES = [920]  # Target requests per second to test
DURATION_SECS = 30  # Duration of each benchmark run in seconds
BATCH_SIZE = 1  # Number of texts per request

PROFILE = False
if PROFILE:
    DURATION_SECS = 1

# Model Configuration
MODEL_PATH = "/home/jobuser/models/Qwen3-Embedding-0.6B/"
EMBEDDING_DIM = None  # Set to None for full dimension, or specify matryoshka dimension

# Input Configuration
INPUT_TOKEN_LENGTH = 512  # Number of tokens in input text

# Encoding Format - determines return type
# None or "tensor" -> returns torch.Tensor (engine-only path)
# "float" -> returns list[float]
ENCODING_FORMAT = "tensor"
# ENCODING_FORMAT = None

# Distribution
DISTRIBUTION = "CONSTANT"  # "CONSTANT" or "POISSON"

# RPS Achievement Threshold
RPS_WARNING_THRESHOLD = 0.95  # Warn if achieved RPS < 95% of target

###############################################################################
# Helper Functions
###############################################################################


def generate_text_with_token_count(
    model_path: str, target_tokens: int, tokenizer=None
) -> str:
    """Generate text with approximately the target number of tokens."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Use a repeating pattern to reach target length
    base_text = "The quick brown fox jumps over the lazy dog. "
    text = base_text

    while True:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) >= target_tokens:
            # Trim to exact length
            trimmed_tokens = tokens[:target_tokens]
            return tokenizer.decode(trimmed_tokens, skip_special_tokens=False)
        text += base_text

    return text


async def sleep_with_distribution(distribution: str, rps: float) -> None:
    """Sleep according to the specified distribution."""
    if distribution == "CONSTANT":
        interval = 1.0 / rps
        await asyncio.sleep(interval)
    elif distribution == "POISSON":
        import random

        interval = random.expovariate(rps)
        await asyncio.sleep(interval)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


###############################################################################
# Benchmark Functions
###############################################################################


async def send_request(
    engine: Engine,
    text: str,
    request_id: int,
    encoding_format: Optional[str],
    dimensions: Optional[int],
    results_queue: asyncio.Queue,
):
    """Send a single embedding request to the engine."""
    try:
        start_time = time.perf_counter()

        # Use async_encode to get embeddings
        result = await engine.async_encode(
            prompt=text,
            encoding_format=encoding_format,
            dimensions=dimensions,
        )

        if encoding_format is None:
            result["embedding"] = np.asarray(
                result["embedding"], dtype=np.float32
            ).tobytes()

        # Record time immediately after await completes
        after_await_time = time.perf_counter()

        # Validate result
        if "embedding" in result:
            embedding = result["embedding"]
            success = True

            # Check format based on encoding_format
            if encoding_format == "tensor":
                if not isinstance(embedding, bytes):
                    print(f"Warning: Expected bytes but got {type(embedding)}")
                    success = False
                else:
                    # Validate device type (accessing .device doesn't cause GPU sync)
                    # device_type = embedding.device.type
                    # if device_type not in ["cpu"]:
                    #     print(f"Warning: Unexpected device type: {device_type}")
                    #     success = False
                    pass
            elif encoding_format == "float" or encoding_format is None:
                if not isinstance(embedding, bytes):
                    print(f"Warning: Expected bytes but got {type(embedding)}")
                    success = False
            else:
                raise ValueError(f"Unknown encoding format: {encoding_format}")
        else:
            success = False
            print(f"Request {request_id} failed: no embedding in result")

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        validation_overhead_ms = (end_time - after_await_time) * 1000

        # Print diagnostic info for first few requests
        if request_id <= 3 and encoding_format == "tensor" and success:
            print(
                f"Request {request_id}: Total={elapsed_ms:.2f}ms, Validation overhead={validation_overhead_ms:.4f}ms, Type={type(embedding)}"
            )

        await results_queue.put(
            {
                "request_id": request_id,
                "latency_ms": elapsed_ms,
                "success": success,
                "completion_time": end_time,
            }
        )

    except Exception as e:
        end_time = time.perf_counter()
        print(f"Request {request_id} failed with error: {e}")
        await results_queue.put(
            {
                "request_id": request_id,
                "latency_ms": 0,
                "success": False,
                "completion_time": end_time,
            }
        )


async def request_sender(
    engine: Engine,
    text: str,
    num_requests: int,
    target_rps: float,
    encoding_format: Optional[str],
    dimensions: Optional[int],
    results_queue: asyncio.Queue,
    distribution: str,
):
    """Separate coroutine to send requests at the target rate using scheduled sending."""
    send_times = []
    start_time = time.perf_counter()

    for i in range(num_requests):
        # Calculate exact target time for this request
        if distribution == "CONSTANT":
            # For constant distribution, space requests evenly
            target_send_time = start_time + (i / target_rps)
        elif distribution == "POISSON":
            # For Poisson distribution, use exponential inter-arrival times
            import random

            if i == 0:
                target_send_time = start_time
            else:
                interval = random.expovariate(target_rps)
                target_send_time = send_times[-1] + interval
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Sleep until target time (if we're ahead of schedule)
        current_time = time.perf_counter()
        sleep_duration = target_send_time - current_time
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)

        # Record actual send time
        actual_send_time = time.perf_counter()
        send_times.append(actual_send_time)

        # Create task without awaiting (fire and forget)
        asyncio.create_task(
            send_request(
                engine, text, i + 1, encoding_format, dimensions, results_queue
            )
        )

    return send_times


async def run_benchmark(
    engine: Engine,
    text: str,
    target_rps: int,
    duration_secs: int,
    batch_size: int,
    encoding_format: Optional[str],
    dimensions: Optional[int],
    distribution: str,
) -> Dict[str, Any]:
    """Run a single benchmark with the given parameters."""
    num_requests = target_rps * duration_secs

    print(f"\n{'='*80}")
    print(f"Starting benchmark:")
    print(f"  Target RPS: {target_rps}")
    print(f"  Duration: {duration_secs}s")
    print(f"  Expected requests: {num_requests}")
    print(f"  Batch size: {batch_size}")
    print(f"  Encoding format: {encoding_format}")
    print(f"  Distribution: {distribution}")
    print(f"{'='*80}")

    results_queue = asyncio.Queue()

    if PROFILE:
        await engine.tokenizer_manager.start_profile()

    # Start sending requests
    benchmark_start = time.perf_counter()
    send_times = await request_sender(
        engine,
        text,
        num_requests,
        target_rps,
        encoding_format,
        dimensions,
        results_queue,
        distribution,
    )
    send_duration = time.perf_counter() - benchmark_start

    # Wait for all requests to complete
    print(f"All requests sent in {send_duration:.2f}s, waiting for completion...")
    results = []
    while len(results) < num_requests:
        result = await results_queue.get()
        results.append(result)

    benchmark_end = time.perf_counter()
    total_duration = benchmark_end - benchmark_start

    if PROFILE:
        await engine.tokenizer_manager.stop_profile()

    # Calculate metrics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = [r["latency_ms"] for r in successful]

    achieved_rps = len(send_times) / send_duration if send_duration > 0 else 0

    metrics = {
        "target_rps": target_rps,
        "achieved_rps": achieved_rps,
        "duration_secs": duration_secs,
        "total_requests": num_requests,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "send_duration": send_duration,
        "total_duration": total_duration,
    }

    if latencies:
        metrics.update(
            {
                "avg_latency_ms": np.mean(latencies),
                "p50_latency_ms": np.percentile(latencies, 50),
                "p90_latency_ms": np.percentile(latencies, 90),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
            }
        )
    else:
        metrics.update(
            {
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p90_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
            }
        )

    # Print results
    print(f"\n{'='*80}")
    print(f"Benchmark Results:")
    print(f"  Target RPS: {target_rps}")
    print(f"  Achieved RPS: {achieved_rps:.2f}")
    print(f"  RPS Achievement: {(achieved_rps/target_rps)*100:.1f}%")

    # Warning if RPS not achieved
    if achieved_rps < target_rps * RPS_WARNING_THRESHOLD:
        print(
            f"  ⚠️  WARNING: Achieved RPS is below {RPS_WARNING_THRESHOLD*100:.0f}% of target!"
        )
        print(f"      This may indicate the benchmark sender is bottlenecked,")
        print(
            f"      not the server. Consider reducing target RPS or optimizing sender."
        )

    print(f"  Total Requests: {num_requests}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Send Duration: {send_duration:.2f}s")
    print(f"  Total Duration: {total_duration:.2f}s")

    if latencies:
        print(f"\nLatency Statistics:")
        print(f"  Average: {metrics['avg_latency_ms']:.2f} ms")
        print(f"  P50: {metrics['p50_latency_ms']:.2f} ms")
        print(f"  P90: {metrics['p90_latency_ms']:.2f} ms")
        print(f"  P95: {metrics['p95_latency_ms']:.2f} ms")
        print(f"  P99: {metrics['p99_latency_ms']:.2f} ms")
        print(f"  Min: {metrics['min_latency_ms']:.2f} ms")
        print(f"  Max: {metrics['max_latency_ms']:.2f} ms")

    print(f"{'='*80}\n")

    return metrics


###############################################################################
# Main Function
###############################################################################


async def main():
    """Main function to run all benchmarks."""
    print(f"{'='*80}")
    print(f"SGLang Embeddings Engine Benchmark")
    print(f"{'='*80}")
    print(f"Model: {MODEL_PATH}")
    print(f"Input Tokens: {INPUT_TOKEN_LENGTH}")
    print(f"Encoding Format: {ENCODING_FORMAT}")
    print(f"Embedding Dimension: {EMBEDDING_DIM}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Distribution: {DISTRIBUTION}")
    print(f"{'='*80}\n")

    # Load tokenizer and generate input text
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Generating input text with {INPUT_TOKEN_LENGTH} tokens...")
    input_text = generate_text_with_token_count(
        MODEL_PATH, INPUT_TOKEN_LENGTH, tokenizer
    )
    actual_tokens = len(tokenizer.encode(input_text, add_special_tokens=True))
    print(f"Generated text with {actual_tokens} tokens")

    # Initialize engine
    print(f"\nInitializing SGLang Engine with {MODEL_PATH}...")
    engine = Engine(
        model_path=MODEL_PATH,
        is_embedding=True,
        disable_radix_cache=True,
    )
    print("Engine initialized successfully")

    # Warmup
    print("\nPerforming warmup...")
    for i in range(3):
        await engine.async_encode(
            prompt=input_text,
            encoding_format=ENCODING_FORMAT,
            dimensions=EMBEDDING_DIM,
        )
    print("Warmup completed")

    # Freeze GC
    print("Freezing garbage collector...")
    await engine.async_freeze_gc()
    print("GC frozen")

    # Run benchmarks
    all_results = []
    for target_rps in TARGET_RPS_VALUES:
        result = await run_benchmark(
            engine=engine,
            text=input_text,
            target_rps=target_rps,
            duration_secs=DURATION_SECS,
            batch_size=BATCH_SIZE,
            encoding_format=ENCODING_FORMAT,
            dimensions=EMBEDDING_DIM,
            distribution=DISTRIBUTION,
        )
        all_results.append(result)

        # Brief pause between benchmarks
        await asyncio.sleep(2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"Summary of All Benchmarks")
    print(f"{'='*80}")
    print(
        f"{'Target RPS':<12} {'Achieved RPS':<15} {'Success %':<12} {'Avg Latency (ms)':<18} {'P99 Latency (ms)':<18}"
    )
    print(f"{'-'*80}")

    for result in all_results:
        success_rate = (result["successful_requests"] / result["total_requests"]) * 100
        rps_achievement = (result["achieved_rps"] / result["target_rps"]) * 100

        # Add warning indicator
        warning = (
            " ⚠️"
            if result["achieved_rps"] < result["target_rps"] * RPS_WARNING_THRESHOLD
            else ""
        )

        print(
            f"{result['target_rps']:<12} "
            f"{result['achieved_rps']:<15.2f}{warning:<4} "
            f"{success_rate:<12.1f} "
            f"{result['avg_latency_ms']:<18.2f} "
            f"{result['p99_latency_ms']:<18.2f}"
        )

    print(f"{'='*80}\n")

    # Shutdown engine
    engine.shutdown()
    print("Engine shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
