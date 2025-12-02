#!/usr/bin/env python3
"""Benchmark Silero VAD and Smart-Turn ONNX models on CUDA vs CPU."""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from vad import silero_vad

# Constants pulled from silero_vad module
SILERO_CONTEXT = 64
SILERO_STATE_SHAPE = (2, 1, 128)
DEFAULT_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--silero-model",
        type=Path,
        default=Path(silero_vad.ONNX_MODEL_PATH),
        help="Path to silero VAD ONNX file",
    )
    parser.add_argument(
        "--smart-turn-model",
        type=Path,
        default=Path("ckpts/smart-turn-v3.0.onnx"),
        help="Path to Smart-Turn ONNX file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of inference calls per provider per concurrency level",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Thread counts to benchmark (include 1 for single-thread latency)",
    )
    parser.add_argument(
        "--smart-turn-seconds",
        type=int,
        default=8,
        help="Smart-Turn audio length in seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for numpy RNG so benchmarks are repeatable",
    )
    return parser.parse_args()


def provider_targets() -> List[str]:
    available = set(ort.get_available_providers())
    targets: List[str] = []
    if "CUDAExecutionProvider" in available:
        targets.append("CUDAExecutionProvider")
    targets.append("CPUExecutionProvider")
    return targets


def format_ms(value: float) -> str:
    return f"{value * 1000:.2f} ms"


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.fromiter(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def run_benchmark(
    label: str,
    run_callable: Callable[[], None],
    iterations: int,
    concurrency: int,
) -> dict:
    def timed_call() -> float:
        start = time.perf_counter()
        run_callable()
        return time.perf_counter() - start

    start_wall = time.perf_counter()
    if concurrency == 1:
        latencies = [timed_call() for _ in range(iterations)]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(timed_call) for _ in range(iterations)]
            latencies = [f.result() for f in futures]
    wall = time.perf_counter() - start_wall

    return {
        "label": label,
        "concurrency": concurrency,
        "avg": statistics.fmean(latencies),
        "p50": percentile(latencies, 50),
        "p90": percentile(latencies, 90),
        "throughput": iterations / wall,
        "wall": wall,
    }


def silero_runner(model_path: Path, providers: List[str]) -> Callable[[], None]:
    silero_vad.ensure_model(str(model_path))
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
        sess_options=opts,
    )

    chunk = np.random.randn(1, silero_vad.CHUNK).astype(np.float32)
    context = np.zeros((1, SILERO_CONTEXT), dtype=np.float32)
    input_tensor = np.concatenate((context, chunk), axis=1)
    feeds = {
        "input": input_tensor,
        "state": np.zeros(SILERO_STATE_SHAPE, dtype=np.float32),
        "sr": np.array(DEFAULT_SAMPLE_RATE, dtype=np.int64),
    }

    def _run() -> None:
        session.run(None, feeds)

    return _run


def smart_turn_runner(
    model_path: Path,
    providers: List[str],
    feature_extractor: WhisperFeatureExtractor,
    seconds: int,
) -> Callable[[], None]:
    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
        sess_options=opts,
    )

    samples = seconds * DEFAULT_SAMPLE_RATE
    dummy_audio = np.random.randn(samples).astype(np.float32)
    inputs = feature_extractor(
        dummy_audio,
        sampling_rate=DEFAULT_SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=samples,
        truncation=True,
        do_normalize=True,
    )
    input_features = np.expand_dims(
        inputs.input_features.squeeze(0).astype(np.float32), axis=0
    )
    feeds = {"input_features": input_features}

    def _run() -> None:
        session.run(None, feeds)

    return _run


def build_providers_list(target: str) -> List[str]:
    if target == "CPUExecutionProvider":
        return ["CPUExecutionProvider"]
    return [target, "CPUExecutionProvider"]


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    silero_model = args.silero_model
    smart_turn_model = args.smart_turn_model
    if not smart_turn_model.exists():
        raise FileNotFoundError(f"Smart-Turn model not found: {smart_turn_model}")

    concurrency_levels = sorted(set(args.concurrency_levels))
    if 1 not in concurrency_levels:
        concurrency_levels = [1] + concurrency_levels

    extractor = WhisperFeatureExtractor(chunk_length=args.smart_turn_seconds)

    for provider in provider_targets():
        provider_list = build_providers_list(provider)
        print(f"\n==== Provider: {provider} -> {provider_list} ====")

        silero = silero_runner(silero_model, provider_list)
        smart_turn = smart_turn_runner(
            smart_turn_model, provider_list, extractor, args.smart_turn_seconds
        )

        for concurrency in concurrency_levels:
            silero_stats = run_benchmark(
                label="Silero VAD",
                run_callable=silero,
                iterations=args.iterations,
                concurrency=concurrency,
            )
            smart_turn_stats = run_benchmark(
                label="Smart-Turn",
                run_callable=smart_turn,
                iterations=args.iterations,
                concurrency=concurrency,
            )

            print(
                f"[Silero VAD] concurrency={concurrency} "
                f"avg={format_ms(silero_stats['avg'])} "
                f"p90={format_ms(silero_stats['p90'])} "
                f"throughput={silero_stats['throughput']:.2f}/s"
            )
            print(
                f"[Smart-Turn] concurrency={concurrency} "
                f"avg={format_ms(smart_turn_stats['avg'])} "
                f"p90={format_ms(smart_turn_stats['p90'])} "
                f"throughput={smart_turn_stats['throughput']:.2f}/s"
            )


if __name__ == "__main__":
    main()
