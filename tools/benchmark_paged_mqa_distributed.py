#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""One paged-MQA workload sharded by KV context across torchrun ranks.

torchrun --standalone --nproc-per-node=16 tools/benchmark_paged_mqa_distributed.py \
    --libdevice-path /actual/libdevice.compute_bi.10.bc --output distributed.json

The original single-device kernels are unchanged. Every measured end-to-end
step broadcasts Q/scales/weights, computes local logits, all-gathers FP32 logits,
and restores a contiguous global [B,L] output on EVERY rank. KV stays resident
and sharded; initial KV generation/distribution is not timed.
"""

import argparse
import hashlib
import json
import math
import os
import socket
import statistics
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_paged_mqa import (
    compare,
    compiler_preflight,
    decode_q,
    environment,
    make_gems_call,
    make_torch_call,
    reference,
    synchronize,
    unpack_cache,
)
from paged_mqa_distributed_inputs import make_shard_inputs, sample_reference, shard_plan
from run_paged_mqa_suite import host_available_memory

GIB = 1024**3
REPORT_CREATED = False


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def protocol(args):
    names = (
        "batch",
        "context",
        "heads",
        "page_size",
        "quant",
        "backends",
        "torch_mode",
        "device",
        "dist_backend",
        "vendor",
        "chunk_tokens",
        "warmup",
        "iterations",
        "repeats",
        "seed",
        "rtol",
        "atol",
        "cpu_threads",
        "memory_fraction",
        "host_memory_fraction",
    )
    return {name: getattr(args, name) for name in names}


def source_hashes():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "benchmark_paged_mqa_distributed.py",
            "paged_mqa_distributed_inputs.py",
            "benchmark_paged_mqa.py",
        )
    }


def estimates(args, world, rank):
    plan = shard_plan(args.context, args.page_size, world, rank)
    local_tokens = (
        max(1, args.batch * math.ceil(plan["length"] / args.page_size)) * args.page_size
    )
    cache = local_tokens * 132
    local_output = args.batch * plan["capacity"] * 4
    gather_buffer = world * local_output
    global_output = args.batch * args.context * 4
    chunk_work = (
        4 * args.batch * args.chunk_tokens * 128
        + 12 * args.batch * args.heads * args.chunk_tokens
    )
    validation = 41 * args.batch * plan["capacity"]
    gpu = int(
        1.15
        * (local_tokens * 776 + gather_buffer + global_output + chunk_work + validation)
        + 2 * GIB
    )
    host = cache + min(local_tokens, 65536) * 128 * 40 + GIB
    if args.device == "cpu":
        host += gpu - 2 * GIB
    return {
        **plan,
        "local_physical_tokens": local_tokens,
        "packed_kv_bytes": cache,
        "local_logits_bytes": local_output,
        "gather_buffer_bytes": gather_buffer,
        "global_logits_bytes_per_rank": global_output,
        "estimated_gpu_bytes": gpu,
        "estimated_host_bytes": host,
        "logical_allgather_receive_bytes_per_rank": (world - 1) * local_output,
    }


def gather_objects(value, world):
    values = [None] * world
    dist.all_gather_object(values, value)
    return values


def collective_smoke(device, rank, world):
    raw = (
        torch.arange(16, dtype=torch.uint8, device=device)
        if rank == 0
        else torch.zeros(16, dtype=torch.uint8, device=device)
    )
    dist.broadcast(raw, src=0)
    if not torch.equal(raw.cpu(), torch.arange(16, dtype=torch.uint8)):
        raise RuntimeError("uint8 broadcast smoke failed")
    scale_expected = torch.tensor([0x7F7F7F7F, -(2**31)], dtype=torch.int32)
    scale = (
        scale_expected.to(device)
        if rank == 0
        else torch.zeros(2, device=device, dtype=torch.int32)
    )
    dist.broadcast(scale, src=0)
    if not torch.equal(scale.cpu(), scale_expected):
        raise RuntimeError("int32 scale broadcast smoke failed")
    item = torch.full((2, 3), float(rank), device=device)
    gathered = torch.empty((world * 2, 3), device=device)
    dist.all_gather_into_tensor(gathered, item)
    restored = gathered.reshape(world, 2, 3).permute(1, 0, 2).reshape(2, world * 3)
    expected = (
        torch.arange(world, device=device, dtype=torch.float32)
        .repeat_interleave(3)
        .expand(2, -1)
    )
    if not torch.equal(restored, expected):
        raise RuntimeError("FP32 all-gather/layout smoke failed")
    synchronize(device)


def broadcast_query(data):
    # Communicate storage bytes: the collective need not support float8 itself.
    dist.broadcast(data.q.view(torch.uint8), src=0)
    if data.q_scale is not None:
        dist.broadcast(data.q_scale, src=0)
    dist.broadcast(data.weights, src=0)


def collector(args, data, device, world):
    capacity = data.width
    buffer = torch.empty(
        (world * args.batch, capacity), device=device, dtype=torch.float32
    )
    result = torch.empty((args.batch, args.context), device=device, dtype=torch.float32)
    by_rank = buffer.view(world, args.batch, capacity)
    parts = []
    for rank in range(world):
        length = shard_plan(args.context, args.page_size, world, rank)["length"]
        if length:
            parts.append(by_rank[rank, :, :length])

    def collect(local):
        if local.shape != (args.batch, capacity) or local.dtype != torch.float32:
            raise RuntimeError("Local implementation returned incompatible shape/dtype")
        dist.all_gather_into_tensor(buffer, local.contiguous())
        # Concatenate columns, not rank-major flattening. Tail padding is omitted.
        torch.cat(parts, dim=1, out=result)
        return result

    return collect


def sampled_positions(args, world):
    positions = {0, args.context - 1, args.context // 2}
    for rank in range(world):
        plan = shard_plan(args.context, args.page_size, world, rank)
        for position in (
            plan["start"] - 1,
            plan["start"],
            plan["start"] + plan["length"] - 1,
        ):
            if 0 <= position < args.context:
                positions.add(position)
    for position in (args.page_size - 1, args.page_size, args.page_size + 1):
        if position < args.context:
            positions.add(position)
    rng = torch.Generator().manual_seed((args.seed + 7919) & ((1 << 64) - 1))
    positions.update(torch.randint(args.context, (16,), generator=rng).tolist())
    return sorted(positions)


def check_global_samples(output, expected, positions, args):
    index = torch.tensor(positions, device=output.device, dtype=torch.long)
    actual = output.index_select(1, index).cpu().float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = (actual - expected).abs()
    passed = finite and bool(
        torch.all(difference <= args.atol + args.rtol * expected.abs())
    )
    return {
        "passed": passed,
        "max_abs": difference.max().item() if finite else None,
        "elements": actual.numel(),
        "columns": positions,
    }


def measure(fn, args, device, world):
    for _ in range(args.warmup):
        value = fn()
        del value
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples, by_rank = [], []
    local_time = torch.empty(1, dtype=torch.float32, device=device)
    all_times = torch.empty(world, dtype=torch.float32, device=device)
    for _ in range(args.repeats):
        synchronize(device)
        dist.barrier()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            value = fn()
            del value
        synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000 / args.iterations
        # Aggregation is outside the measured interval. All ranks' critical path matters.
        local_time.fill_(elapsed)
        dist.all_gather_into_tensor(all_times, local_time)
        times = all_times.cpu().tolist()
        by_rank.append(times)
        samples.append(max(times))
    peaks = gather_objects(
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        world,
    )
    return {
        "critical_ms_median": statistics.median(samples),
        "critical_ms_samples": samples,
        "rank_ms_samples": by_rank,
        "torch_peak_allocated_bytes_by_rank": peaks,
    }


def workload(args, quant):
    return {
        "batch": args.batch,
        "context": args.context,
        "heads": args.heads,
        "head_dim": 128,
        "next_n": 1,
        "page_size": args.page_size,
        "quant": quant,
        "seed": args.seed,
        "generator": "stateless-global-coordinates-v1",
        "chunk_tokens": args.chunk_tokens,
        "torch_mode": args.torch_mode,
        "scope": "resident KV; broadcast + local API + all-gather + contiguous assembly",
    }


def add_scaling(report, path):
    if path is None:
        return
    baseline = json.loads(path.read_text())
    if baseline.get("world_size") != 1:
        raise ValueError("--single-rank-baseline must come from world_size=1")
    if baseline.get("source_hashes") != report["source_hashes"]:
        raise ValueError(
            "Single-rank and multi-rank benchmark/helper source hashes differ"
        )
    if baseline.get("dist_backend") != report["dist_backend"]:
        raise ValueError("Single-rank and multi-rank communication backends differ")
    previous_env = baseline.get("ranks", [{}])[0].get("environment", {})
    for item in report["ranks"]:
        current_env = item["environment"]
        if any(
            previous_env.get(key) != current_env.get(key)
            for key in ("device_name", "packages")
        ):
            raise ValueError(
                "Scaling comparison requires matching device model and software versions"
            )
    for row in report["results"]:
        if row["status"] != "PASS":
            continue
        match = next(
            (
                item
                for item in baseline.get("results", [])
                if item.get("workload") == row["workload"]
                and item.get("backend") == row["backend"]
                and item.get("status") == "PASS"
            ),
            None,
        )
        if match is None:
            raise ValueError(
                "Single-rank baseline lacks a matching shape/format/backend/seed/mode"
            )
        if row["backend"] == "gems" and match.get("implementation", {}).get(
            "source_sha256"
        ) != row["implementation"].get("source_sha256"):
            raise ValueError("Single-rank and multi-rank Gems source hashes differ")
        if baseline.get("harness_sha256") != report["harness_sha256"]:
            raise ValueError("Single-rank and multi-rank benchmark scripts differ")
        speedup = (
            match["timing"]["end_to_end"]["critical_ms_median"]
            / row["timing"]["end_to_end"]["critical_ms_median"]
        )
        row["strong_scaling_speedup"] = speedup
        row["strong_scaling_efficiency"] = speedup / report["world_size"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--context", type=int, default=1048576)
    parser.add_argument("--heads", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument(
        "--page-size", type=int, choices=(16, 32, 64, 128, 256), default=256
    )
    parser.add_argument("--quant", choices=("fp8", "fp4", "both"), default="fp8")
    parser.add_argument(
        "--backends", nargs="+", choices=("torch", "gems"), default=["torch", "gems"]
    )
    parser.add_argument("--torch-mode", choices=("compile", "eager"), default="compile")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--vendor", default="iluvatar")
    parser.add_argument("--libdevice-path", type=Path)
    parser.add_argument("--op-file", type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--host-memory-fraction", type=float, default=0.7)
    parser.add_argument("--single-rank-baseline", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("paged_mqa_distributed.json")
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print shard/memory plan without initializing distributed or CUDA",
    )
    parser.add_argument("--plan-world-size", type=int, default=16)
    args = parser.parse_args()
    for name in (
        "batch",
        "context",
        "chunk_tokens",
        "iterations",
        "repeats",
        "cpu_threads",
        "timeout",
        "plan_world_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.warmup < 0 or args.seed < 0:
        parser.error("warmup/seed must be nonnegative")
    if not all(math.isfinite(x) and x >= 0 for x in (args.rtol, args.atol)):
        parser.error("tolerances must be finite and nonnegative")
    if not 0 < args.memory_fraction <= 1 or not 0 < args.host_memory_fraction <= 1:
        parser.error("memory fractions must be in (0,1]")
    if args.device == "cpu" and (
        args.dist_backend != "gloo" or "gems" in args.backends
    ):
        parser.error("CPU tests require --dist-backend gloo --backends torch")
    return args


def run(args):
    global REPORT_CREATED
    if args.plan:
        plans = [
            estimates(args, args.plan_world_size, rank)
            for rank in range(args.plan_world_size)
        ]
        print(
            json.dumps(
                {
                    "world_size": args.plan_world_size,
                    "total_packed_kv_GiB": sum(p["packed_kv_bytes"] for p in plans)
                    / GIB,
                    "global_logits_GiB_per_rank": args.batch * args.context * 4 / GIB,
                    "ranks": plans,
                },
                indent=2,
            )
        )
        return 0
    if "RANK" not in os.environ:
        raise RuntimeError(
            "Launch with torchrun (even for one rank); use --plan for an offline plan"
        )
    rank, world, local_rank = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if rank == 0:
        if args.output.exists():
            raise FileExistsError(
                "Use a new --output path; existing results are not overwritten"
            )
        write_json(
            args.output,
            {
                "status": "RUNNING",
                "world_size": world,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": protocol(args),
            },
        )
        REPORT_CREATED = True
    torch.set_num_threads(args.cpu_threads)
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ["GEMS_VENDOR"] = args.vendor
    os.environ["FLAGGEMS_FP8_FP4_PAGED_MQA_LOGITS_TLE"] = "0"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    device = (
        torch.device("cuda", local_rank)
        if args.device == "cuda"
        else torch.device("cpu")
    )
    preflight = None
    if device.type == "cuda":
        torch.cuda.set_device(device)
        preflight = compiler_preflight(args.vendor, args.libdevice_path)
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group(
        args.dist_backend, init_method="env://", timeout=timedelta(seconds=args.timeout)
    )
    collective_smoke(device, rank, world)
    estimate = estimates(args, world, rank)
    info = {
        "rank": rank,
        "local_rank": local_rank,
        "node": os.environ.get("GROUP_RANK", socket.gethostname()),
        "source_hashes": source_hashes(),
        "protocol": protocol(args),
        "environment": environment(device),
        "compiler_preflight": preflight,
        "memory_estimate": estimate,
        "host_available_bytes": host_available_memory(),
        "gpu_free_bytes": (
            torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None
        ),
    }
    ranks = gather_objects(info, world)
    failures = []
    if any(item["source_hashes"] != ranks[0]["source_hashes"] for item in ranks):
        raise RuntimeError("Workers have different benchmark/helper source files")
    if any(item["protocol"] != ranks[0]["protocol"] for item in ranks):
        raise RuntimeError(
            "Workers have different workload or collective-loop arguments"
        )
    for item in ranks:
        est = item["memory_estimate"]
        if (
            item["gpu_free_bytes"] is not None
            and est["estimated_gpu_bytes"]
            > item["gpu_free_bytes"] * args.memory_fraction
        ):
            failures.append(f"rank {item['rank']}: estimated GPU memory exceeds budget")
        if "gems" in args.backends and est["local_physical_tokens"] * 128 > 2**31:
            failures.append(
                f"rank {item['rank']}: local FP8 K offsets exceed the current kernel's conservative int32 bound"
            )
    for node in {item["node"] for item in ranks}:
        members = [item for item in ranks if item["node"] == node]
        available = [
            item["host_available_bytes"]
            for item in members
            if item["host_available_bytes"] is not None
        ]
        if (
            available
            and sum(item["memory_estimate"]["estimated_host_bytes"] for item in members)
            > min(available) * args.host_memory_fraction
        ):
            failures.append(
                f"node {node}: combined host-memory estimate exceeds budget"
            )
    report = {
        "world_size": world,
        "dist_backend": args.dist_backend,
        "ranks": ranks,
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "measurement": (
            "shared workload; context-sharded resident KV; max rank wall time; "
            "startup/JIT/reference excluded"
        ),
        "phase_note": (
            "The three timing phases are independent measurements; do not sum "
            "compute-only and communication-only to predict end-to-end latency."
        ),
        "status": "RUNNING",
        "results": [],
    }

    def save():
        if rank == 0:
            write_json(args.output, report)

    if failures:
        report.update(status="SKIPPED_RESOURCE", errors=failures)
        save()
        if rank == 0:
            print("\n".join(failures), flush=True)
        dist.destroy_process_group()
        return 2
    if rank == 0:
        print(
            f"One workload: B={args.batch}, L={args.context}, {world} ranks; "
            f"total KV={sum(item['memory_estimate']['packed_kv_bytes'] for item in ranks)/GIB:.3f} GiB",
            flush=True,
        )
    failed = False
    quants = ("fp8", "fp4") if args.quant == "both" else (args.quant,)
    with torch.no_grad():
        for quant in quants:
            if rank == 0:
                print(
                    f"{quant}: building KV shards and references (outside timing) ...",
                    flush=True,
                )
            data, plan = make_shard_inputs(args, quant, rank, world, device)
            broadcast_query(data)
            q = decode_q(data.q, data.q_scale)
            k, scales = unpack_cache(data.cache)
            expected_local = reference(
                q, k, scales, data.weights, data, args.chunk_tokens
            )
            del q, k, scales
            positions = sampled_positions(args, world)
            expected_samples = sample_reference(args, quant, positions)
            collect = collector(args, data, device, world)
            for backend in args.backends:
                synchronize(device)
                dist.barrier()
                name = (
                    "gems"
                    if backend == "gems"
                    else f"torch_{args.torch_mode}_chunked_fp32"
                )
                if rank == 0:
                    print(f"{quant} {name}: compiling and validating ...", flush=True)
                if plan["length"] == 0:
                    fn = lambda width=data.width: torch.zeros(
                        (args.batch, width), device=device
                    )
                    details = {"empty_shard": True}
                elif backend == "gems":
                    fn, details = make_gems_call(data, args.op_file)
                else:
                    fn = make_torch_call(
                        data, args.chunk_tokens, compiled=args.torch_mode == "compile"
                    )
                    details = {
                        "compiled_region": (
                            "torch_chunk" if args.torch_mode == "compile" else None
                        ),
                        "arithmetic": "FP32",
                        "source": str(
                            Path(__file__).with_name("benchmark_paged_mqa.py")
                        ),
                    }
                implementations = gather_objects(details, world)
                hashes = {
                    item["source_sha256"]
                    for item in implementations
                    if "source_sha256" in item
                }
                if len(hashes) > 1:
                    raise RuntimeError("Ranks loaded different Gems operator sources")
                first = fn()
                synchronize(device)
                local_check = compare(first, expected_local, data, args.rtol, args.atol)
                assembled = collect(first)
                global_check = check_global_samples(
                    assembled, expected_samples, positions, args
                )
                checks = gather_objects(
                    {
                        "rank": rank,
                        "local": local_check,
                        "global_samples": global_check,
                    },
                    world,
                )
                row = {
                    "workload": workload(args, quant),
                    "backend": name,
                    "implementation": details,
                    "implementation_by_rank": implementations,
                    "correctness": checks,
                    "status": "INCORRECT",
                    "communication": {
                        "query_broadcast_bytes": data.q.numel() * data.q.element_size()
                        + data.weights.numel() * data.weights.element_size()
                        + (
                            data.q_scale.numel() * data.q_scale.element_size()
                            if data.q_scale is not None
                            else 0
                        ),
                        "allgather_local_input_bytes": first.numel()
                        * first.element_size(),
                        "logical_allgather_receive_bytes_per_rank": estimate[
                            "logical_allgather_receive_bytes_per_rank"
                        ],
                        "global_output_bytes_per_rank": estimate[
                            "global_logits_bytes_per_rank"
                        ],
                        "initial_kv_generation_or_transfer_timed": False,
                    },
                }
                if not all(
                    item["local"]["passed"] and item["global_samples"]["passed"]
                    for item in checks
                ):
                    failed = True
                    report["results"].append(row)
                    save()
                    if rank == 0:
                        print(f"INCORRECT {quant} {name}; see JSON", flush=True)
                    del first, assembled, fn
                    continue

                def communication_only(current_data=data, gather=collect, cached=first):
                    broadcast_query(current_data)
                    return gather(cached)

                def end_to_end(current_data=data, compute=fn, gather=collect):
                    broadcast_query(current_data)
                    local = compute()
                    return gather(local)

                timing = {
                    "compute_only": measure(fn, args, device, world),
                    "communication_and_assembly_only": measure(
                        communication_only, args, device, world
                    ),
                    "end_to_end": measure(end_to_end, args, device, world),
                }
                seconds = timing["end_to_end"]["critical_ms_median"] / 1000
                row.update(
                    status="PASS",
                    timing=timing,
                    global_logits_per_second=args.batch * args.context / seconds,
                    queries_per_second=args.batch / seconds,
                )
                report["results"].append(row)
                save()
                if rank == 0:
                    print(
                        f"PASS {quant} {name}: "
                        f"compute={timing['compute_only']['critical_ms_median']:.3f} ms, "
                        f"comm+assembly={timing['communication_and_assembly_only']['critical_ms_median']:.3f} ms, "
                        f"end-to-end={timing['end_to_end']['critical_ms_median']:.3f} ms",
                        flush=True,
                    )
                del first, assembled, fn, communication_only, end_to_end
            del collect, data, expected_local, expected_samples
    if rank == 0:
        for quant in quants:
            rows = {
                row["backend"]: row
                for row in report["results"]
                if row["workload"]["quant"] == quant and row["status"] == "PASS"
            }
            torch_row = rows.get(f"torch_{args.torch_mode}_chunked_fp32")
            gems_row = rows.get("gems")
            if torch_row and gems_row:
                speedup = (
                    torch_row["timing"]["end_to_end"]["critical_ms_median"]
                    / gems_row["timing"]["end_to_end"]["critical_ms_median"]
                )
                report.setdefault("comparisons", []).append(
                    {"quant": quant, "gems_end_to_end_speedup_vs_torch": speedup}
                )
        add_scaling(report, args.single_rank_baseline)
        report["status"] = "INCORRECT" if failed else "PASS"
        save()
        print(f"Saved {args.output.resolve()}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    args = parse_args()
    try:
        raise SystemExit(run(args))
    except Exception:
        # Let torchrun terminate peer workers. Do not enter another collective on error.
        text = traceback.format_exc()
        rank = os.environ.get("RANK", "0")
        error_file = args.output.with_name(args.output.stem + f".rank{rank}.error.json")
        write_json(error_file, {"rank": rank, "error": text})
        if rank == "0" and REPORT_CREATED:
            report = json.loads(args.output.read_text())
            report.update(status="ERROR", error=text, error_file=str(error_file))
            write_json(args.output, report)
        print(text, file=sys.stderr, flush=True)
        raise
