#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Serial IxFormer / torch / torch.compile / FlagGems paged-MQA baseline.

Run from the checkout, using the Python in the working CoreX container:
  python tools/benchmark_paged_mqa.py --vendor iluvatar
  python tools/benchmark_paged_mqa.py --vendor iluvatar --torch-mode both --quant both

This is a benchmark, not a production fallback. next_n=1, D=128 only.
Native uses prepared BF16 Q/K/weights and an explicitly allocated FP32 output.
Torch and Gems consume the original packed FP8/FP4 inputs. Native/Gems ratios
therefore describe different input formats and are NOT a same-format 95% gate.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Inputs:
    quant: str
    q: torch.Tensor
    q_scale: object
    cache: torch.Tensor
    weights: torch.Tensor
    lengths: torch.Tensor
    tables: torch.Tensor
    contexts: list
    width: int


def decode_q(q, q_scale):
    """Decode the stored values, not the pre-quantization random tensor."""
    if q_scale is None:
        return q[:, 0].float()
    packed = q[:, 0].to(torch.int32)
    codes = torch.stack((packed & 15, (packed >> 4) & 15), dim=-1)
    codes = codes.flatten(-2)
    exponent = (codes >> 1) & 3
    mantissa = (codes & 1).float()
    magnitude = torch.where(
        exponent == 0,
        mantissa * 0.5,
        (1.0 + mantissa * 0.5) * torch.exp2(exponent.float() - 1.0),
    )
    shifts = torch.arange(4, device=q.device, dtype=torch.int32) * 8
    scale_codes = (q_scale[:, 0, :, None] >> shifts) & 255
    scales = torch.exp2(scale_codes.float() - 127.0).repeat_interleave(32, -1)
    return magnitude * (1.0 - 2.0 * ((codes >> 3) & 1).float()) * scales


def unpack_cache(cache):
    """Each page holds all FP8 K bytes first, then all FP32 scale bytes."""
    pages, page, _, packed_dim = cache.shape
    dim = packed_dim - 4
    raw = cache.reshape(pages, -1)
    k = raw[:, : page * dim].contiguous().view(torch.float8_e4m3fn)
    k = k.reshape(pages, page, dim).float()
    scales = raw[:, page * dim :].contiguous().view(torch.float32)
    return k, scales.reshape(pages, page)


def pack_cache(k_fp8, scales):
    pages, page, dim = k_fp8.shape
    raw = torch.empty((pages, page * (dim + 4)), dtype=torch.uint8)
    raw[:, : page * dim] = k_fp8.contiguous().view(torch.uint8).reshape(pages, -1)
    raw[:, page * dim :] = scales.contiguous().view(torch.uint8).reshape(pages, -1)
    return raw.view(pages, page, 1, dim + 4)


def make_inputs(args, quant, device):
    rng = torch.Generator().manual_seed(args.seed)
    contexts = args.contexts or [args.context] * args.batch
    batch, heads, dim = len(contexts), args.heads, 128
    width = args.max_model_len or max(contexts)
    required = sum(math.ceil(n / args.page_size) for n in contexts)
    pages = args.cache_pages or required
    if pages < required:
        raise ValueError(f"--cache-pages must be at least {required} (no shared pages)")
    if width < max(contexts):
        raise ValueError("--max-model-len must be >= every context length")

    if quant == "fp8":
        q = torch.randn((batch, 1, heads, dim), generator=rng)
        q = q.to(torch.float8_e4m3fn)
        q_scale = None
    else:
        # Legal synthetic MXFP4 codes: no dependency on a vendor quantizer.
        codes = torch.randint(0, 16, (batch, 1, heads, dim), generator=rng)
        q = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
        scale_codes = torch.randint(124, 128, (batch, 1, heads, 4), generator=rng)
        shifts = torch.arange(4, dtype=torch.int64) * 8
        q_scale = (scale_codes << shifts).sum(-1).to(torch.int32)

    k = torch.randn((pages, args.page_size, dim), generator=rng)
    scales = k.abs().amax(-1).clamp_min(1e-4) / 448.0
    k_fp8 = (k / scales[..., None]).to(torch.float8_e4m3fn)
    cache = pack_cache(k_fp8, scales)
    weights = torch.randn((batch, heads), generator=rng) / math.sqrt(heads)
    tables = torch.zeros((batch, math.ceil(width / args.page_size)), dtype=torch.int32)
    physical_ids = torch.randperm(pages, generator=rng).to(torch.int32)
    offset = 0
    for row, length in enumerate(contexts):
        count = math.ceil(length / args.page_size)
        tables[row, :count] = physical_ids[offset : offset + count]
        offset += count

    return Inputs(
        quant,
        q.to(device),
        None if q_scale is None else q_scale.to(device),
        cache.to(device),
        weights.to(device),
        torch.tensor(contexts, dtype=torch.int32, device=device),
        tables.to(device),
        contexts,
        width,
    )


def torch_chunk(q, k, scales, weights, tables, lengths, positions):
    """FP32 arithmetic, logical token -> physical page -> weighted ReLU dot."""
    page = k.shape[1]
    logical_page = (positions // page).clamp(max=tables.shape[1] - 1)
    physical_page = tables[:, logical_page].long()
    slot = positions % page
    keys = k[physical_page, slot[None, :]]
    sf = scales[physical_page, slot[None, :]]
    dots = torch.bmm(q, keys.transpose(1, 2))
    scores = (dots * sf[:, None, :]).clamp_min(0)
    result = (scores * weights[:, :, None]).sum(1)
    return torch.where(positions[None, :] < lengths[:, None], result, 0.0)


def make_torch_call(data, chunk_tokens, compiled=False, compile_backend="inductor"):
    # Compile one fixed-sized chunk, avoiding a giant unrolled long-context graph.
    chunk_fn = torch_chunk
    if compiled:
        chunk_fn = torch.compile(
            torch_chunk, backend=compile_backend, fullgraph=True, dynamic=False
        )
    compute_width = max(data.contexts)
    position_chunks = [
        torch.arange(start, start + chunk_tokens, device=data.q.device)
        for start in range(0, compute_width, chunk_tokens)
    ]

    def run():
        # Decoding, cache extraction, allocation and all chunk calls are timed.
        q = decode_q(data.q, data.q_scale)
        k, scales = unpack_cache(data.cache)
        result = torch.zeros(
            (q.shape[0], data.width), device=q.device, dtype=torch.float32
        )
        for start, positions in zip(
            range(0, compute_width, chunk_tokens), position_chunks
        ):
            tile = chunk_fn(
                q, k, scales, data.weights, data.tables, data.lengths, positions
            )
            size = min(chunk_tokens, data.width - start)
            result[:, start : start + size] = tile[:, :size]
        return result

    return run


def reference(q, k, scales, weights, data, chunk_tokens):
    """Independent per-request gather/mm, used outside all timing windows."""
    result = torch.zeros((q.shape[0], data.width), device=q.device, dtype=torch.float32)
    page = k.shape[1]
    for row, length in enumerate(data.contexts):
        for start in range(0, length, chunk_tokens):
            pos = torch.arange(
                start, min(start + chunk_tokens, length), device=q.device
            )
            physical = data.tables[row, pos // page].long()
            keys = k[physical, pos % page]
            sf = scales[physical, pos % page]
            dots = q[row].float() @ keys.float().T
            scores = torch.relu(dots * sf[None, :])
            result[row, start : start + pos.numel()] = (
                scores * weights[row, :, None]
            ).sum(0)
    return result


def make_native_call(data):
    native_ops = importlib.import_module("ixformer.inference.functions")
    fn = native_ops.dsa_indexer_mqa_logits_with_blocks
    q = decode_q(data.q, data.q_scale).to(torch.bfloat16).contiguous()
    k, scales = unpack_cache(data.cache)
    k = (k * scales[..., None]).to(torch.bfloat16).contiguous()
    weights = data.weights.to(torch.bfloat16).contiguous()
    batch = q.shape[0]
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    cu_kv = torch.cat(
        (data.lengths.new_zeros(1), data.lengths.cumsum(0, dtype=torch.int32))
    )
    metadata = {
        "function": "ixformer.inference.functions.dsa_indexer_mqa_logits_with_blocks",
        "source": inspect.getsourcefile(fn),
        "input_format": "BF16 Q/K/weights prepared from the quantized input",
        "output_format": "FP32, freshly allocated on every timed call",
        "adapter_conversion_timed": False,
    }
    config = getattr(fn, "__globals__", {}).get("config")
    if config is not None:
        metadata["IXFORMER_USE_TORCH_OPS"] = config.IXFORMER_USE_TORCH_OPS

    def run():
        out = torch.empty((batch, data.width), dtype=torch.float32, device=q.device)
        result = fn(
            q,
            cu_q,
            cu_kv,
            k,
            data.tables,
            weights,
            logits=out,
            max_q_len=1,
            max_kv_len=max(data.contexts),
            max_context_len=data.width,
        )
        if result is None:
            raise RuntimeError("Native wrapper returned None; expected a logits tensor")
        if result.dtype != torch.float32 or result.data_ptr() != out.data_ptr():
            raise RuntimeError("Native wrapper did not honor the supplied FP32 output")
        return result

    native_ref = reference(
        q.float(),
        k.float(),
        torch.ones(k.shape[:2], device=q.device),
        weights.float(),
        data,
        512,
    )
    return run, native_ref, metadata


def make_gems_call(data, op_file=None):
    name = "flaggems_vllm.ops.fp8_fp4_paged_mqa_logits"
    if op_file:
        spec = importlib.util.spec_from_file_location("paged_mqa_candidate", op_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {op_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(name)
    fn = module.fp8_fp4_paged_mqa_logits
    source = Path(inspect.getsourcefile(fn)).resolve()

    def run():
        return fn(
            q=(data.q, data.q_scale),
            kv_cache=data.cache,
            weights=data.weights,
            context_lens=data.lengths,
            block_tables=data.tables,
            schedule_metadata=None,
            max_model_len=data.width,
            clean_logits=False,
        )

    return run, {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_format": f"{data.quant} Q + packed FP8 K / FP32 scales and weights",
        "output_format": "FP32",
        "scope": "complete public operator, including KV copies and output initialization",
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compare(actual, expected, data, rtol, atol):
    if actual.shape != expected.shape:
        raise ValueError(f"output shape {actual.shape}, expected {expected.shape}")
    valid = (
        torch.arange(data.width, device=actual.device)[None, :] < data.lengths[:, None]
    )
    a, b = actual.float()[valid], expected.float()[valid]
    if not torch.isfinite(b).all().item():
        return {"passed": False, "reason": "nonfinite reference logits"}
    if not torch.isfinite(a).all().item():
        return {"passed": False, "reason": "nonfinite valid logits"}
    if a.numel() == 0:
        return {"passed": True, "elements": 0, "max_abs": 0.0, "nrmse": 0.0}
    delta = a - b
    failures = (delta.abs() > atol + rtol * b.abs()).sum().item()
    return {
        "passed": failures == 0,
        "mismatched": failures,
        "elements": a.numel(),
        "max_abs": delta.abs().max().item(),
        "nrmse": (
            delta.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)
        ).item(),
        "rtol": rtol,
        "atol": atol,
        "region": "valid logits only; padding values are not a common native contract",
    }


def benchmark(fn, device, warmup, iterations, repeats):
    for _ in range(warmup):
        value = fn()
        del value
    synchronize(device)
    wall_samples, device_samples = [], []
    for _ in range(repeats):
        synchronize(device)
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        for _ in range(iterations):
            value = fn()
            del value
        if device.type == "cuda":
            end_event.record()
        synchronize(device)
        wall_samples.append((time.perf_counter() - start) * 1e6 / iterations)
        if device.type == "cuda":
            device_samples.append(
                start_event.elapsed_time(end_event) * 1000 / iterations
            )
    return {
        "wall_us_median": statistics.median(wall_samples),
        "wall_us_samples": wall_samples,
        "device_timeline_us_median": (
            statistics.median(device_samples) if device_samples else None
        ),
        "device_timeline_us_samples": device_samples,
        "warmup_calls": warmup,
        "iterations_per_repeat": iterations,
        "repeats": repeats,
    }


def environment(device):
    versions = {}
    for name in ("torch", "triton", "flagtree", "ixformer", "vllm", "flaggems-vllm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    repo = Path(__file__).resolve().parents[1]
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = git.stdout.strip() or None
    except FileNotFoundError:
        commit = None
    return {
        "python": sys.version,
        "packages": versions,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "torch_cuda_build": torch.version.cuda,
        "repo_commit": commit,
        "benchmark_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "GEMS_VENDOR": os.environ.get("GEMS_VENDOR"),
        "FLAGGEMS_FP8_FP4_PAGED_MQA_LOGITS_TLE": os.environ.get(
            "FLAGGEMS_FP8_FP4_PAGED_MQA_LOGITS_TLE"
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--vendor", default="iluvatar", help="GEMS_VENDOR (default: iluvatar)"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--quant", choices=("fp8", "fp4", "both"), default="fp8")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--context", type=int, default=512)
    p.add_argument(
        "--contexts",
        type=lambda x: [int(v) for v in x.split(",")],
        help="per-request lengths, e.g. 63,64,65,512; overrides --batch/--context",
    )
    p.add_argument("--heads", type=int, choices=(16, 32, 64), default=64)
    p.add_argument("--page-size", type=int, choices=(16, 32, 64, 128, 256), default=64)
    p.add_argument("--max-model-len", type=int)
    p.add_argument(
        "--cache-pages",
        type=int,
        help="physical cache capacity; default: unique pages for all requests",
    )
    p.add_argument("--chunk-tokens", type=int, default=512)
    p.add_argument(
        "--torch-mode", choices=("eager", "compile", "both"), default="eager"
    )
    p.add_argument(
        "--backends",
        nargs="+",
        choices=("native", "torch", "gems"),
        default=["native", "torch", "gems"],
    )
    p.add_argument("--compile-backend", default="inductor")
    p.add_argument(
        "--op-file",
        type=Path,
        help="optional replacement file exporting fp8_fp4_paged_mqa_logits",
    )
    p.add_argument(
        "--warmup", type=int, default=10, help="number of calls, not milliseconds"
    )
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("paged_mqa_baseline.json"))
    p.add_argument(
        "--enable-tle",
        action="store_true",
        help="opt in; disabled by default for initial Iluvatar measurements",
    )
    args = p.parse_args(argv)
    if args.enable_tle and args.quant != "fp8":
        p.error(
            "FP4 TLE is not safe in the current operator; use --quant fp8 or disable TLE"
        )
    for name in ("batch", "context", "chunk_tokens", "iterations", "repeats"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or args.rtol < 0 or args.atol < 0:
        p.error("warmup and tolerances must be nonnegative")
    if args.contexts is not None and (not args.contexts or min(args.contexts) <= 0):
        p.error("--contexts must contain positive lengths")
    for name in ("max_model_len", "cache_pages"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["GEMS_VENDOR"] = args.vendor
    os.environ["FLAGGEMS_FP8_FP4_PAGED_MQA_LOGITS_TLE"] = (
        "1" if args.enable_tle else "0"
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Use a CoreX CUDA device, or CPU for torch-only checks")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA/CoreX is unavailable to this Python; use the working server container"
            )
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    report = {
        "environment": environment(device),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "measurement": "sequential eager API calls; first-call compilation and warmup excluded; no CUDA graphs",
        "comparison": "native BF16 prepared-input baseline differs from FP8/FP4 API; no 95% acceptance claim",
        "results": [],
    }
    print(json.dumps(report["environment"], indent=2, ensure_ascii=False), flush=True)
    print(
        "Native: prepared BF16 -> FP32. Torch/Gems: packed FP8/FP4 -> FP32.", flush=True
    )
    print(
        "Native conversion is outside timing. Ratios are not a same-format 95% test.",
        flush=True,
    )
    failed = False
    names = []
    if "native" in args.backends:
        names.append("native")
    if "torch" in args.backends:
        if args.torch_mode in ("eager", "both"):
            names.append("torch_fp32")
        if args.torch_mode in ("compile", "both"):
            names.append("torch_compile_chunked_fp32")
    if "gems" in args.backends:
        names.append("gems")
    quantizations = ("fp8", "fp4") if args.quant == "both" else (args.quant,)
    with torch.inference_mode():
        for quant in quantizations:
            data = make_inputs(args, quant, device)
            q_ref = decode_q(data.q, data.q_scale)
            k_ref, scales_ref = unpack_cache(data.cache)
            expected = reference(
                q_ref, k_ref, scales_ref, data.weights, data, args.chunk_tokens
            )
            del q_ref, k_ref, scales_ref
            print(
                f"\n{quant}: B={len(data.contexts)}, H={args.heads}, D=128, "
                f"page={args.page_size}, contexts={data.contexts}, "
                f"output_width={data.width}",
                flush=True,
            )
            for name in names:
                print(f"  {name}: preparing / first call ...", flush=True)
                row = {
                    "quant": quant,
                    "backend": name,
                    "status": "ERROR",
                    "case": {
                        "batch": len(data.contexts),
                        "next_n": 1,
                        "heads": args.heads,
                        "head_dim": 128,
                        "page_size": args.page_size,
                        "contexts": data.contexts,
                        "max_model_len": data.width,
                        "physical_pages": data.cache.shape[0],
                    },
                }
                fn = value = native_ref = None
                try:
                    synchronize(device)
                    if name in ("native", "gems") and device.type != "cuda":
                        raise RuntimeError(f"{name} requires the target GPU")
                    reference_for_backend = expected
                    if name == "native":
                        fn, native_ref, details = make_native_call(data)
                        reference_for_backend = native_ref
                        row["bf16_input_rounding_vs_quantized_reference"] = compare(
                            native_ref, expected, data, args.rtol, args.atol
                        )
                    elif name == "gems":
                        fn, details = make_gems_call(data, args.op_file)
                        print(f"    source: {details['source']}", flush=True)
                    else:
                        compiled = "compile" in name
                        fn = make_torch_call(
                            data, args.chunk_tokens, compiled, args.compile_backend
                        )
                        details = {
                            "arithmetic": "FP32",
                            "compile_backend": (
                                args.compile_backend if compiled else None
                            ),
                            "scope": "decode + allocation + paged gather/dot/ReLU/reduction chunks + output copies",
                            "position_indices_prepared_outside_timing": True,
                            "compiled_region": (
                                "fixed-size torch_chunk, fullgraph=True"
                                if compiled
                                else None
                            ),
                        }
                    row["implementation"] = details
                    synchronize(device)
                    start = time.perf_counter()
                    value = fn()
                    synchronize(device)
                    row["first_call_ms"] = (time.perf_counter() - start) * 1000
                    row["correctness"] = compare(
                        value, reference_for_backend, data, args.rtol, args.atol
                    )
                    row["vs_quantized_reference"] = compare(
                        value, expected, data, args.rtol, args.atol
                    )
                    if value.dtype != torch.float32:
                        raise RuntimeError(
                            f"Expected FP32 output, received {value.dtype}"
                        )
                    if not row["correctness"]["passed"]:
                        row["status"] = "INCORRECT"
                        failed = True
                        print(f"  {name}: INCORRECT {row['correctness']}", flush=True)
                    else:
                        del value
                        value = None
                        row["timing"] = benchmark(
                            fn, device, args.warmup, args.iterations, args.repeats
                        )
                        row["status"] = "PASS"
                        print(
                            f"  {name}: PASS  "
                            f"wall={row['timing']['wall_us_median']:.3f} us  "
                            f"max_abs={row['correctness']['max_abs']:.6g}",
                            flush=True,
                        )
                        if name == "native":
                            cross = row["vs_quantized_reference"]
                            print(
                                "    BF16 reference passed; versus packed-input reference: "
                                f"max_abs={cross.get('max_abs')}, nrmse={cross.get('nrmse')}",
                                flush=True,
                            )
                except Exception as exc:
                    failed = True
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"  {name}: ERROR {row['error']}", flush=True)
                finally:
                    report["results"].append(row)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
                    )
                    fn = value = native_ref = reference_for_backend = None
            del data, expected
    for quant in quantizations:
        rows = [
            r
            for r in report["results"]
            if r["quant"] == quant and r["status"] == "PASS"
        ]
        gems = next((r for r in rows if r["backend"] == "gems"), None)
        if gems:
            for row in rows:
                ratio = (
                    row["timing"]["wall_us_median"] / gems["timing"]["wall_us_median"]
                )
                row["wall_time_over_gems"] = ratio
                row["ratio_scope"] = (
                    "mixed input formats, diagnostic only"
                    if row["backend"] == "native"
                    else "same quantized input API"
                )
                print(
                    f"{quant} {row['backend']}/gems wall-time ratio: {ratio:.4f}",
                    flush=True,
                )
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved {args.output.resolve()}", flush=True)
    print(
        "PASS means valid-region numerical check passed, not native 95% acceptance.",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
