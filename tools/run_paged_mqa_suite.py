#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run a size matrix; each case keeps native/torch/compile/Gems sequential.

python tools/run_paged_mqa_suite.py --preset core --devices 0 --output-dir results/before
python tools/run_paged_mqa_suite.py --preset all --devices all --jobs 4 --output-dir results/sweep
python tools/run_paged_mqa_suite.py --preset all --list-cases

Device IDs are Torch-visible IDs on THIS host, not physical board IDs.
Parallel sweeps are screening runs. Recheck important timings with --jobs 1.
"""

import argparse
import concurrent.futures
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3
BENCHMARK = Path(__file__).with_name("benchmark_paged_mqa.py")
RUNNING = set()
PROCESS_LOCK = threading.Lock()
CANCELLED = threading.Event()


def interrupt_children(signum, frame):
    CANCELLED.set()
    with PROCESS_LOCK:
        processes = list(RUNNING)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 3
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(0.05)
    for process in processes:
        # The leader can exit while a compiler grandchild remains in its group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    raise KeyboardInterrupt


def case(name, batch=4, context=512, **kwargs):
    return {
        "id": name,
        "batch": batch,
        "context": context,
        "page_size": 64,
        "heads": 64,
        **kwargs,
    }


def cases_for(preset):
    smoke = [
        case("boundary_single", 1, 65),
        case("boundary_mixed", contexts=[1, 63, 64, 65], max_model_len=128),
        case("b4_l512"),
    ]
    core = smoke + [
        case("b1_l8192", 1, 8192),
        case("b8_l2048", 8, 2048),
        case("b32_l2048", 32, 2048),
        case("b128_l2048", 128, 2048),
        case("b256_l2048", 256, 2048),
        case("b32_l16384", 32, 16384),
        case("b128_l16384", 128, 16384),
        case("b64_l32768", 64, 32768),
        case("b32_l65536", 32, 65536),
        case("b32_l131072", 32, 131072),
        case("page256_b32_l16384", 32, 16384, page_size=256),
        case("skewed_b32", contexts=[512] * 24 + [4096] * 7 + [65536]),
        case("wide_output", 32, 1024, max_model_len=131072),
        case("large_cache_small_query", cache_pages=4096),
    ]
    stress = [
        case("stress_b1024_l8192", 1024, 8192),
        case("stress_b512_l32768", 512, 32768),
        case("stress_b256_l65536", 256, 65536),
        case("stress_b64_l131072", 64, 131072, page_size=256),
        case("stress_cache_pool", 32, 32768, page_size=256, cache_pages=32768),
        case("stress_b256_l131072", 256, 131072, page_size=256),
    ]
    return {"smoke": smoke, "core": core, "stress": stress, "all": core + stress}[
        preset
    ]


def dimensions(c):
    lengths = c.get("contexts") or [c["context"]] * c["batch"]
    page = c["page_size"]
    width = c.get("max_model_len") or max(lengths)
    required = sum((n + page - 1) // page for n in lengths)
    pages = c.get("cache_pages") or required
    return lengths, width, pages


def validate_cases(cases):
    allowed = {
        "id",
        "batch",
        "context",
        "contexts",
        "page_size",
        "heads",
        "max_model_len",
        "cache_pages",
    }
    ids = set()
    for c in cases:
        if not isinstance(c, dict) or set(c) - allowed:
            raise ValueError(f"Invalid case fields: {c}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", c.get("id", "")) or c["id"] in ids:
            raise ValueError(
                "Case IDs must be unique and contain only letters, digits, _ or -"
            )
        ids.add(c["id"])
        c.setdefault("batch", len(c.get("contexts", [])) or 4)
        c.setdefault("context", 512)
        c.setdefault("page_size", 64)
        c.setdefault("heads", 64)
        for key in (
            "batch",
            "context",
            "page_size",
            "heads",
            "max_model_len",
            "cache_pages",
        ):
            if key in c and (type(c[key]) is not int or c[key] <= 0):
                raise ValueError(f"{c['id']}: {key} must be a positive integer")
        if "contexts" in c and (
            not isinstance(c["contexts"], list)
            or not c["contexts"]
            or any(type(n) is not int or n <= 0 for n in c["contexts"])
        ):
            raise ValueError(f"{c['id']}: contexts must be positive integers")
        if c["page_size"] not in (16, 32, 64, 128, 256) or c["heads"] not in (
            16,
            32,
            64,
        ):
            raise ValueError(f"{c['id']}: unsupported page size or heads")
        lengths, width, pages = dimensions(c)
        if width < max(lengths) or pages < sum(
            math.ceil(n / c["page_size"]) for n in lengths
        ):
            raise ValueError(f"{c['id']}: output width/cache capacity too small")
    if not cases:
        raise ValueError("No cases selected")


def estimate_memory(c, chunk, cpu=False):
    """Conservative tensor estimate + workspace reserve, not an OOM guarantee."""
    lengths, width, pages = dimensions(c)
    batch, heads, dim = len(lengths), c["heads"], 128
    tokens = pages * c["page_size"]
    outputs, valid = batch * width, sum(lengths)
    decode_peak = tokens * (6 * dim + 8)
    torch_chunk_peak = 4 * batch * chunk * dim + 12 * batch * heads * chunk
    validation_peak = 17 * outputs + 24 * valid
    gpu = int(1.15 * (decode_peak + torch_chunk_peak + validation_peak) + 2 * GIB)
    host = tokens * (dim + 4) + min(tokens, 65536) * (10 * dim + 8) + GIB
    if cpu:
        # In CPU test mode, decode/reference/compute buffers also live in RAM.
        host += gpu - 2 * GIB
    return {
        "gpu_bytes_estimate": gpu,
        "host_bytes_estimate": host,
        "packed_cache_bytes": tokens * (dim + 4),
        "logits_bytes": outputs * 4,
        "physical_tokens": tokens,
        "valid_tokens": valid,
    }


def host_available_memory():
    """Account for both the machine and a common cgroup-v2 container limit."""
    available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        current = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if limit != "max":
            remaining = max(0, int(limit) - current)
            available = (
                min(available, remaining) if available is not None else remaining
            )
    except (OSError, ValueError):
        pass
    return available


def inventory(selection):
    if selection == "cpu":
        return [{"device": "cpu", "name": "CPU test mode", "total_memory": None}]
    import torch

    count = torch.cuda.device_count()
    ids = (
        list(range(count))
        if selection == "all"
        else [int(x) for x in selection.split(",")]
    )
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= count for i in ids):
        raise ValueError(
            f"--devices must select unique Torch-visible IDs in 0..{count - 1}"
        )
    result = []
    for i in ids:
        p = torch.cuda.get_device_properties(i)
        result.append(
            {
                "device": f"cuda:{i}",
                "name": p.name,
                "total_memory": p.total_memory,
                "uuid": str(getattr(p, "uuid", "unknown")),
            }
        )
    return result


def free_device_memory(device):
    if device == "cpu":
        return None
    import torch

    return torch.cuda.mem_get_info(torch.device(device))[0]


def command_for(job, device, args, output):
    c = job["case"]
    lengths, _, _ = dimensions(c)
    large = sum(lengths) >= 4 * 1024**2 or max(lengths) >= 65536
    warmup = args.warmup if args.warmup is not None else (2 if large else 5)
    iterations = (
        args.iterations if args.iterations is not None else (5 if large else 20)
    )
    cmd = [
        sys.executable,
        str(BENCHMARK),
        "--device",
        device,
        "--quant",
        job["quant"],
        "--vendor",
        args.vendor,
        "--page-size",
        str(c["page_size"]),
        "--heads",
        str(c["heads"]),
        "--chunk-tokens",
        str(args.chunk_tokens),
        "--torch-mode",
        args.torch_mode,
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--output",
        str(output),
        "--backends",
        *args.backends,
    ]
    if c.get("contexts"):
        cmd += ["--contexts", ",".join(str(n) for n in lengths)]
    else:
        cmd += ["--batch", str(c["batch"]), "--context", str(c["context"])]
    for key in ("max_model_len", "cache_pages"):
        if key in c:
            cmd += ["--" + key.replace("_", "-"), str(c[key])]
    if args.op_file:
        cmd += ["--op-file", str(args.op_file.resolve())]
    return cmd


def expected_backends(args):
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
    return names


def execute(job, device, args, free_before):
    name = job["id"]
    output = (args.output_dir / f"{name}.json").resolve()
    log = (args.output_dir / f"{name}.log").resolve()
    cmd = command_for(job, device, args, output)
    env = (
        os.environ.copy()
    )  # Preserve CUDA_VISIBLE_DEVICES: IDs already refer to this mask.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(args.cpu_threads)
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    row = {
        **job,
        "device": device,
        "hostname": socket.gethostname(),
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "free_device_memory_before": free_before,
        "command": cmd,
        "report": str(output),
        "log": str(log),
    }
    start = time.monotonic()
    process = None
    try:
        with log.open("w") as stream:
            with PROCESS_LOCK:
                if CANCELLED.is_set():
                    raise RuntimeError("Suite interrupted before launch")
                process = subprocess.Popen(
                    cmd,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                RUNNING.add(process)
            try:
                code = process.wait(timeout=args.timeout)
                row["status"] = "PASS" if code == 0 else "FAILED"
                row["exit_code"] = code
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                row["status"] = "TIMEOUT"
        if output.exists():
            details = json.loads(output.read_text())
            row["environment"] = details.get("environment")
            row["results"] = details.get("results", [])
            expected = expected_backends(args)
            recorded = [result.get("backend") for result in row["results"]]
            if row["status"] == "PASS" and (
                recorded != expected
                or any(
                    result.get("status") != "PASS"
                    or result.get("quant") != job["quant"]
                    for result in row["results"]
                )
            ):
                row["status"] = "ERROR"
                row["error"] = (
                    "Child result set is incomplete or contains failed/unexpected backends"
                )
        elif row["status"] == "PASS":
            row["status"] = "ERROR"
            row["error"] = "Child exited successfully without a result JSON"
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with PROCESS_LOCK:
            RUNNING.discard(process)
    if CANCELLED.is_set():
        row["status"] = "CANCELLED"
    recorded = {result.get("backend") for result in row.get("results", [])}
    for name in expected_backends(args):
        if name not in recorded:
            row.setdefault("results", []).append(
                {
                    "backend": name,
                    "status": "NOT_COMPLETED",
                    "error": f"Case ended with {row['status']}",
                }
            )
    row["elapsed_s"] = time.monotonic() - start
    row["end_utc"] = datetime.now(timezone.utc).isoformat()
    return row


def save_summary(report, directory):
    report["results"].sort(key=lambda r: r["id"])
    (directory / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "**Paged MQA suite results**",
        "",
        report["measurement_note"],
        "",
        "| case | device | case status | backend | backend status | wall us | max abs error |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for run in report["results"]:
        for result in run.get("results") or [{"backend": "-", "status": run["status"]}]:
            wall = result.get("timing", {}).get("wall_us_median")
            error = result.get("correctness", {}).get("max_abs")
            prefix = (
                f"| {run['id']} | {run.get('device', '-')} | {run['status']} | "
                f"{result['backend']} | {result['status']} | "
            )
            numbers = (
                f"{wall:.3f} | {error:.6g} |"
                if wall is not None and error is not None
                else "- | - |"
            )
            lines.append(prefix + numbers)
    (directory / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--preset", choices=("smoke", "core", "stress", "all"), default="core"
    )
    p.add_argument(
        "--cases-file", type=Path, help="JSON array of cases; overrides preset"
    )
    p.add_argument("--case", nargs="+", help="select exact case IDs")
    p.add_argument(
        "--devices",
        default="0",
        help="Torch-visible IDs, e.g. 0,1,2,3; all; or cpu for tests",
    )
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--quant", choices=("fp8", "fp4", "both"), default="both")
    p.add_argument(
        "--torch-mode", choices=("eager", "compile", "both"), default="eager"
    )
    p.add_argument(
        "--backends",
        choices=("native", "torch", "gems"),
        nargs="+",
        default=["native", "torch", "gems"],
    )
    p.add_argument("--vendor", default="iluvatar")
    p.add_argument("--op-file", type=Path)
    p.add_argument("--chunk-tokens", type=int, default=512)
    p.add_argument("--warmup", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="seconds per case, including compilation and correctness",
    )
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument(
        "--memory-fraction",
        type=float,
        default=0.7,
        help="maximum fraction of currently free memory allowed by estimate",
    )
    p.add_argument("--host-memory-fraction", type=float, default=0.6)
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("paged_mqa_suite"))
    p.add_argument(
        "--list-cases",
        action="store_true",
        help="print shapes and memory estimates; no GPU needed",
    )
    args = p.parse_args()
    for key in ("jobs", "chunk_tokens", "repeats", "timeout", "cpu_threads"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if (
        args.warmup is not None
        and args.warmup < 0
        or args.iterations is not None
        and args.iterations <= 0
    ):
        p.error("invalid warmup/iterations")
    if not 0 < args.memory_fraction <= 1 or not 0 < args.host_memory_fraction <= 1:
        p.error("memory fractions must be in (0, 1]")
    if not all(math.isfinite(x) and x >= 0 for x in (args.rtol, args.atol)):
        p.error("tolerances must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    cases = (
        json.loads(args.cases_file.read_text())
        if args.cases_file
        else cases_for(args.preset)
    )
    if not isinstance(cases, list):
        raise ValueError("--cases-file must contain a JSON array")
    validate_cases(cases)
    if args.case:
        unknown = set(args.case) - {c["id"] for c in cases}
        if unknown:
            raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in args.case]
    if args.list_cases:
        print(
            "case                         B    max L   page   packed GiB   estimated GPU GiB   estimated host GiB"
        )
        for c in cases:
            lengths, _, _ = dimensions(c)
            est = estimate_memory(c, args.chunk_tokens, args.devices == "cpu")
            print(
                f"{c['id']:28s} {len(lengths):4d} {max(lengths):8d} {c['page_size']:5d} "
                f"{est['packed_cache_bytes']/GIB:12.3f} "
                f"{est['gpu_bytes_estimate']/GIB:19.3f} "
                f"{est['host_bytes_estimate']/GIB:20.3f}"
            )
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            "Use a new/empty --output-dir; existing baseline results are not overwritten"
        )
    devices = inventory(args.devices)
    workers = min(args.jobs, len(devices))
    available_host = host_available_memory()
    host_budget = (
        available_host * args.host_memory_fraction
        if available_host is not None
        else None
    )
    quants = ("fp8", "fp4") if args.quant == "both" else (args.quant,)
    pending = deque(
        {
            "id": f"{c['id']}_{q}",
            "case": c,
            "quant": q,
            "memory_estimate": estimate_memory(
                c, args.chunk_tokens, args.devices == "cpu"
            ),
        }
        for c in cases
        for q in quants
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    report = {
        "hostname": socket.gethostname(),
        "devices": devices,
        "jobs": workers,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host_available_before": available_host,
        "host_budget": host_budget,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "measurement_note": (
            "Concurrent sweep: screening only; remeasure key cases with jobs=1."
            if workers > 1
            else "Serial size sweep; each case uses one device and sequential backends."
        ),
        "comparison": "Native BF16 prepared inputs differ from the packed FP8/FP4 API; no same-format 95% claim.",
        "results": [],
    }
    print(
        json.dumps(
            {"devices": devices, "jobs": workers, "cases": len(pending)}, indent=2
        ),
        flush=True,
    )
    active = {}
    last_notice = time.monotonic()
    CANCELLED.clear()
    old_handler = signal.signal(signal.SIGINT, interrupt_children)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        while pending or active:
            busy = {info["device"] for info in active.values()}
            for entry in devices:
                if not pending or len(active) >= workers:
                    break
                device = entry["device"]
                if device in busy:
                    continue
                free_gpu = free_device_memory(device)
                host_now = host_available_memory()
                reserved_host = sum(
                    info["job"]["memory_estimate"]["host_bytes_estimate"]
                    for info in active.values()
                )
                selected = None
                for job in pending:
                    est = job["memory_estimate"]
                    gpu_ok = (
                        free_gpu is None
                        or est["gpu_bytes_estimate"] <= free_gpu * args.memory_fraction
                    )
                    host_ok = (
                        host_budget is None
                        or reserved_host + est["host_bytes_estimate"] <= host_budget
                    )
                    host_ok = host_ok and (
                        host_now is None
                        or est["host_bytes_estimate"]
                        <= host_now * args.host_memory_fraction
                    )
                    if gpu_ok and host_ok:
                        selected = job
                        break
                if selected is None:
                    continue
                pending.remove(selected)
                print(f"START {selected['id']} on {device}", flush=True)
                future = pool.submit(execute, selected, device, args, free_gpu)
                active[future] = {"device": device, "job": selected}
            if not active:
                while pending:
                    job = pending.popleft()
                    report["results"].append(
                        {
                            **job,
                            "status": "SKIPPED_MEMORY",
                            "reason": "Conservative estimate exceeds current free-device or host-memory budget",
                        }
                    )
                    print(f"SKIPPED_MEMORY {job['id']}", flush=True)
                save_summary(report, args.output_dir)
                break
            finished, _ = concurrent.futures.wait(
                active, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in finished:
                row = future.result()
                report["results"].append(row)
                del active[future]
                print(
                    f"{row['status']} {row['id']} on {row['device']} ({row['elapsed_s']:.1f}s); {row['log']}",
                    flush=True,
                )
                save_summary(report, args.output_dir)
            if active and time.monotonic() - last_notice >= 30:
                print(
                    "Running: "
                    + ", ".join(info["job"]["id"] for info in active.values()),
                    flush=True,
                )
                last_notice = time.monotonic()
    except KeyboardInterrupt:
        report["interrupted"] = True
        for future in active:
            report["results"].append(future.result())
        for job in pending:
            report["results"].append(
                {
                    **job,
                    "status": "CANCELLED",
                    "reason": "Suite interrupted before launch",
                }
            )
        save_summary(report, args.output_dir)
        print(
            "Interrupted; child process groups stopped and partial results saved.",
            flush=True,
        )
        return 130
    finally:
        pool.shutdown(wait=True)
        signal.signal(signal.SIGINT, old_handler)
    print(f"Saved {args.output_dir.resolve() / 'summary.md'}", flush=True)
    statuses = {row["status"] for row in report["results"]}
    return (
        1
        if statuses - {"PASS", "SKIPPED_MEMORY"}
        else (2 if "SKIPPED_MEMORY" in statuses else 0)
    )


if __name__ == "__main__":
    raise SystemExit(main())
