#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Five sizes x dense/paged x FP8/FP4 x Torch/original Gems/TLE candidate.

One entry point; it launches isolated, sequential torchrun process groups.
Freeze the original package BEFORE editing kernels, then benchmark both source
snapshots under the same installed Torch/FlagTree stack. See benchmark_mqa_suite.md.
"""

import argparse
import csv
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = ("4x4096", "16x16384", "64x65536", "128x262144", "256x1048576")
BACKENDS = ("torch", "baseline", "tle")
OPS = {"dense": "fp8_fp4_mqa_logits", "paged": "fp8_fp4_paged_mqa_logits"}
PHASES = ("compute_only", "communication_and_assembly_only", "end_to_end")
TLE_FLAGS = (
    "FLAGGEMS_FP8_FP4_MQA_LOGITS_TLE",
    "FLAGGEMS_FP8_FP4_PAGED_MQA_LOGITS_TLE",
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def package_hashes(src):
    package = Path(src) / "flaggems_vllm"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"Not a source directory containing flaggems_vllm: {src}")
    for name in OPS.values():
        if not (package / "ops" / f"{name}.py").is_file():
            raise ValueError(f"Missing operator {name} in {src}")
    files = {
        str(p.relative_to(package)): digest(p.read_bytes())
        for p in sorted(package.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in (".pyc", ".pyo")
    }
    return files, digest(json.dumps(files, sort_keys=True).encode())


def freeze_source(src, destination):
    """Freeze the whole package, including imported helpers and tune configs."""
    src, destination = Path(src).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Snapshot already exists; refusing to replace: {destination}"
        )
    if destination == src or src in destination.parents:
        raise ValueError("A snapshot must be outside its source directory")
    before, tree = package_hashes(src)
    destination.mkdir(parents=True)
    shutil.copytree(
        src / "flaggems_vllm",
        destination / "src" / "flaggems_vllm",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    copied, copied_tree = package_hashes(destination / "src")
    if copied != before or copied_tree != tree:
        raise RuntimeError("Sources changed while copying; this snapshot is incomplete")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "files": copied,
        "tree_sha256": tree,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def verify_snapshot(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    files, tree = package_hashes(directory / "src")
    if manifest.get("schema_version") != 1 or files != manifest.get("files"):
        raise ValueError(f"Snapshot is incomplete or was modified: {directory}")
    if tree != manifest.get("tree_sha256"):
        raise ValueError(f"Snapshot manifest hash mismatch: {directory}")
    return manifest


def harness_hashes():
    return {
        name: digest(Path(__file__).with_name(name).read_bytes())
        for name in (
            "benchmark_mqa_suite.py",
            "mqa_suite_inputs.py",
            "benchmark_paged_mqa.py",
            "benchmark_paged_mqa_distributed.py",
            "paged_mqa_distributed_inputs.py",
            "run_paged_mqa_suite.py",
        )
    }


def load_frozen_operator(operator, directory, manifest):
    """Resolve the public vendor dispatch and verify its frozen source provenance."""
    package_root = (Path(directory) / "src" / "flaggems_vllm").resolve()

    def verified_file(filename, label):
        if not filename:
            raise RuntimeError(f"Cannot verify {label}: no source file")
        path = Path(filename).resolve()
        try:
            relative = str(path.relative_to(package_root))
        except ValueError as exc:
            raise RuntimeError(
                f"Imported {label} from {path}, outside frozen source {package_root}"
            ) from exc
        expected = manifest["files"].get(relative)
        if (
            expected is None
            or not path.is_file()
            or digest(path.read_bytes()) != expected
        ):
            raise RuntimeError(f"Imported {label} does not match frozen source: {path}")
        return path

    package = importlib.import_module("flaggems_vllm")
    verified_file(getattr(package, "__file__", None), "public package")
    name = OPS[operator]
    fn = getattr(package, name, None)
    if not callable(fn):
        raise RuntimeError(f"Public operator flaggems_vllm.{name} is not callable")
    module_name = getattr(fn, "__module__", "")
    if not module_name.startswith("flaggems_vllm."):
        raise RuntimeError(
            f"Public operator {name} has unexpected module {module_name!r}"
        )
    module = importlib.import_module(module_name)
    module_file = verified_file(getattr(module, "__file__", None), "operator module")
    try:
        filename = inspect.getsourcefile(fn)
    except TypeError as exc:
        raise RuntimeError(f"Cannot verify source of public operator {name}") from exc
    function_file = verified_file(filename, "operator function")
    if function_file != module_file:
        raise RuntimeError(
            f"Public operator {name} source {function_file} "
            f"differs from its module {module_file}"
        )
    return (
        package,
        module,
        {
            "public_operator": f"flaggems_vllm.{name}",
            "module": module_name,
            "file": str(module_file),
            "sha256": digest(module_file.read_bytes()),
            "function_file": str(function_file),
            "package_tree_sha256": manifest["tree_sha256"],
        },
    )


def parse_shapes(values):
    result, seen = [], set()
    for value in values:
        parts = value.lower().split("x")
        if len(parts) != 2 or not all(p.isdecimal() for p in parts):
            raise ValueError(f"Shape must be MxN / BxL, for example 4x4096: {value}")
        batch, context = map(int, parts)
        if min(batch, context) <= 0 or context > 2**31 - 1:
            raise ValueError("Shapes must be positive and context must fit int32")
        if (batch, context) in seen:
            raise ValueError(f"Duplicate shape: {value}")
        seen.add((batch, context))
        result.append(
            {
                "id": f"s{len(result) + 1}_b{batch}_l{context}",
                "batch": batch,
                "context": context,
            }
        )
    return result


def observed_call(module, launcher, fn):
    """Observe a host TLE launcher for ONE untimed validation call, then restore.

    This establishes dispatch, not instruction-level profiling. Future candidate
    implementations can name their host launcher with --{dense,paged}-tle-launcher.
    """
    original = getattr(module, launcher, None) if module is not None else None
    if not callable(original):
        return fn(), None
    calls = 0

    def traced(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    setattr(module, launcher, traced)
    try:
        value = fn()
    finally:
        setattr(module, launcher, original)
    return value, calls


def row_status(checks, backend):
    if not all(c["local"]["passed"] and c["global_samples"]["passed"] for c in checks):
        return "INCORRECT"
    active = [c for c in checks if not c["empty_shard"]]
    if backend == "tle":
        if any(c["tle_launcher_calls"] is None for c in active):
            return "UNVERIFIED_TLE"
        if any(c["tle_launcher_calls"] == 0 for c in active):
            return "FALLBACK"
    if backend == "baseline" and any(c["tle_launcher_calls"] for c in active):
        return "BASELINE_USED_TLE"
    return "PASS"


def finite_latency(row, phase):
    if not row or row.get("status") != "PASS":
        return None
    value = row.get("timing", {}).get(phase, {}).get("critical_ms_median")
    return (
        value
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        else None
    )


def comparison(rows, phase):
    values = {b: finite_latency(rows.get(b), phase) for b in BACKENDS}
    signatures = [
        r.get("comparison_signature")
        for r in rows.values()
        if r.get("status") == "PASS"
    ]
    compatible = (
        bool(signatures) and None not in signatures and len(set(signatures)) == 1
    )
    ratios = {}
    for name, numerator, denominator in (
        ("baseline_vs_torch", "torch", "baseline"),
        ("tle_vs_torch", "torch", "tle"),
        ("tle_vs_baseline", "baseline", "tle"),
    ):
        a, b = values[numerator], values[denominator]
        ratios[name] = a / b if compatible and a is not None and b is not None else None
    return {"comparable": compatible, **ratios}


def summarize(directory):
    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text())
    pending_status = "PENDING" if run.get("status") == "RUNNING" else "NOT_COMPLETED"
    collected, issues = {}, []
    for job in run["jobs"]:
        report_path = directory / job["report"]
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if report.get("status") in ("ERROR", "TIMEOUT", "INTERRUPTED"):
            issues.append(
                {
                    "job": job["id"],
                    "quant": "all",
                    "status": report["status"],
                    "error": report.get("error"),
                    "log": job["log"],
                }
            )
        rows = {r["quant"]: r for r in report.get("results", [])}
        for quant in run["quants"]:
            key = (job["case"]["id"], job["operator"], quant)
            row = rows.get(quant, {"status": report.get("status", pending_status)})
            if row.get("status") in ("RUNNING", "PASS") and "timing" not in row:
                row = {**row, "status": pending_status}
            if row.get("status") == "PASS" and any(
                finite_latency(row, phase) is None for phase in PHASES
            ):
                row = {**row, "status": "INVALID_TIMING"}
            collected.setdefault(key, {})[job["backend"]] = row
            if row["status"] != "PASS":
                issues.append(
                    {
                        "job": job["id"],
                        "quant": quant,
                        "status": row["status"],
                        "error": row.get("error") or report.get("error"),
                        "log": job["log"],
                    }
                )
    records = []
    for case in run["cases"]:
        for operator in run["operators"]:
            for quant in run["quants"]:
                rows = collected.get((case["id"], operator, quant), {})
                passed = [row for row in rows.values() if row.get("status") == "PASS"]
                if (
                    len(passed) > 1
                    and not comparison(rows, "compute_only")["comparable"]
                ):
                    issues.append(
                        {
                            "job": f"{case['id']}_{operator}",
                            "quant": quant,
                            "status": "INCOMPARABLE",
                            "error": (
                                "Input, protocol, harness or runtime/device signatures "
                                "differ across implementations"
                            ),
                            "log": "run.json",
                        }
                    )
                for phase in PHASES:
                    record = {
                        "case": case["id"],
                        "operator": operator,
                        "quant": quant,
                        "batch": case["batch"],
                        "context": case["context"],
                        "phase": phase,
                    }
                    for backend in BACKENDS:
                        row = rows.get(backend, {})
                        record[f"{backend}_status"] = row.get("status", "NOT_SELECTED")
                        record[f"{backend}_ms"] = (
                            row.get("timing", {})
                            .get(phase, {})
                            .get("critical_ms_median")
                        )
                    record.update(comparison(rows, phase))
                    records.append(record)
    failed = any(
        i["status"]
        in (
            "ERROR",
            "TIMEOUT",
            "INCORRECT",
            "BASELINE_USED_TLE",
            "NOT_COMPLETED",
            "INTERRUPTED",
            "INCOMPARABLE",
            "INVALID_TIMING",
        )
        for i in issues
    )
    summary = {
        "status": (
            "RUNNING"
            if run.get("status") == "RUNNING"
            else "ERROR" if failed else "PARTIAL" if issues else "PASS"
        ),
        "device": run["settings"]["device"],
        "records": records,
        "issues": issues,
    }
    write_json(directory / "summary.json", summary)
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    def latency(record, backend):
        status, value = record[f"{backend}_status"], record[f"{backend}_ms"]
        text = f"{value:.3f}" if value is not None else "—"
        return text if status == "PASS" else f"{text} ({status})"

    def ratio(value):
        return "—" if value is None else f"{value:.3f}×"

    lines = [
        "# MQA 三方对比",
        "",
        f"状态：{summary['status']}；{run['settings']['nproc_per_node']} ranks；单位：ms/调用。",
        "",
        "三项独立测量；每轮取最慢 rank 的每次调用平均耗时，再取多轮中位数。",
        "",
        "普通版所有 Q 共享 K；paged 版每个请求有独立 K，两个算子之间不计算加速比。",
        "",
        "FALLBACK 表示数值检查通过但至少一个非空 rank 未进入 TLE launcher；UNVERIFIED_TLE 表示缺少可观察的 launcher。它们不产生 TLE 加速比。",
        "",
    ]
    if run["settings"]["device"] == "cpu":
        lines += [
            "**CPU 检查模式：仅验证输入、参考和通信，不代表 GPU/Gems/TLE 性能。**",
            "",
        ]
    titles = dict(zip(PHASES, ("本地完整算子 API", "通信与拼接", "端到端")))
    for phase in PHASES:
        lines += [
            f"## {titles[phase]}",
            "",
            "| 算子 | M/B × N/L | Q 格式 | Torch | Gems baseline | Gems TLE | "
            "baseline/Torch 加速比 | TLE/Torch 加速比 | TLE/baseline 加速比 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for record in records:
            if record["phase"] != phase:
                continue
            cells = [
                record["operator"],
                f"{record['batch']} × {record['context']}",
                record["quant"],
            ]
            cells += [latency(record, backend) for backend in BACKENDS]
            cells += [
                ratio(record[name])
                for name in ("baseline_vs_torch", "tle_vs_torch", "tle_vs_baseline")
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "加速比均为对照耗时 / 被评估实现耗时，大于 1 表示被评估实现更快。",
        "",
        "## 源码与诊断",
        "",
        "源码快照及校验清单保存在 `sources/`；完整 rank 样本、数值误差和路径证据保存在 `reports/`。",
        "",
    ]
    for issue in issues:
        detail = str(issue.get("error") or "查看报告中的正确性和 TLE 路径记录").replace(
            "\n", " "
        )
        lines.append(
            f"- `{issue['job']}` / {issue['quant']}: **{issue['status']}**；{detail[:1000]}；[日志]({issue['log']})"
        )
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--plan", action="store_true", help="print the size matrix without Torch/GPU"
    )
    action.add_argument(
        "--freeze-baseline",
        type=Path,
        metavar="DIR",
        help="snapshot the original package, then exit",
    )
    action.add_argument("--summarize-only", type=Path, metavar="DIR")
    action.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--baseline-dir", type=Path, help="original snapshot made before kernel edits"
    )
    parser.add_argument(
        "--candidate-src",
        type=Path,
        default=ROOT / "src",
        help="src directory used for freezing, or for the TLE candidate",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--shapes", nargs="+", default=list(DEFAULT_SHAPES), metavar="BxL"
    )
    parser.add_argument("--operators", nargs="+", choices=tuple(OPS), default=list(OPS))
    parser.add_argument("--quant", choices=("fp8", "fp4", "both"), default="both")
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument("--nproc-per-node", type=int, default=16)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29501)
    parser.add_argument("--vendor", default="iluvatar")
    parser.add_argument("--libdevice-path", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--torch-mode", choices=("compile", "eager"), default="compile")
    parser.add_argument(
        "--page-size", type=int, choices=(16, 32, 64, 128, 256), default=256
    )
    parser.add_argument("--heads", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--host-memory-fraction", type=float, default=0.7)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--timeout", type=int, default=600, help="collective timeout, seconds"
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=7200,
        help="whole process-group timeout, seconds",
    )
    parser.add_argument("--dense-tle-launcher", default="_launch_tle_kernel")
    parser.add_argument("--paged-tle-launcher", default="_launch_tle_kernel")
    args = parser.parse_args(argv)
    if args.worker_spec or args.summarize_only or args.freeze_baseline:
        return args
    for name in (
        "nproc_per_node",
        "chunk_tokens",
        "iterations",
        "repeats",
        "cpu_threads",
        "timeout",
        "job_timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.warmup < 0 or args.seed < 0 or not 1 <= args.master_port <= 65535:
        parser.error("invalid warmup, seed or port")
    if any(
        not math.isfinite(getattr(args, k)) or getattr(args, k) < 0
        for k in ("rtol", "atol")
    ):
        parser.error("tolerances must be finite and nonnegative")
    if any(
        not 0 < getattr(args, k) <= 1
        for k in ("memory_fraction", "host_memory_fraction")
    ):
        parser.error("memory fractions must be in (0, 1]")
    for key in ("backends", "operators"):
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            parser.error(f"duplicate {key}")
    try:
        args.cases = parse_shapes(args.shapes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.device == "cpu" and args.backends != ["torch"] and not args.plan:
        parser.error("CPU validation requires --backends torch")
    return args


def stop_group(process):
    # Include torchrun workers/compilers even if the torchrun leader exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_suite(args):
    if "RANK" in os.environ:
        raise ValueError(
            "Run this entry point with python, not torchrun; it launches torchrun itself"
        )
    if "baseline" in args.backends and args.baseline_dir is None:
        raise ValueError(
            "First use --freeze-baseline DIR, then pass --baseline-dir DIR"
        )
    if args.baseline_dir:
        verify_snapshot(args.baseline_dir)
    output = (
        args.output_dir
        or ROOT / "results" / datetime.now().strftime("mqa_%Y%m%d_%H%M%S")
    ).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new --output-dir: {output}")
    import torch

    if args.device == "cuda" and torch.cuda.device_count() < args.nproc_per_node:
        raise ValueError(
            f"Requested {args.nproc_per_node} ranks, but only {torch.cuda.device_count()} Torch-visible devices"
        )
    output.mkdir(parents=True)
    sources = {}
    for backend, src in (
        ("baseline", args.baseline_dir / "src" if args.baseline_dir else None),
        ("tle", args.candidate_src),
    ):
        if backend in args.backends:
            directory = output / "sources" / backend
            manifest = freeze_source(src, directory)
            sources[backend] = {
                "directory": str(directory),
                "tree_sha256": manifest["tree_sha256"],
            }
    settings = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key
        not in (
            "cases",
            "shapes",
            "plan",
            "worker_spec",
            "freeze_baseline",
            "summarize_only",
        )
    }
    quants = ["fp8", "fp4"] if args.quant == "both" else [args.quant]
    jobs = []
    for case in args.cases:
        for operator in args.operators:
            for backend in args.backends:
                name = f"{case['id']}_{operator}_{backend}"
                jobs.append(
                    {
                        "id": name,
                        "case": case,
                        "operator": operator,
                        "backend": backend,
                        "report": f"reports/{name}.json",
                        "log": f"logs/{name}.log",
                    }
                )
    run = {
        "schema_version": 1,
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": args.cases,
        "operators": args.operators,
        "quants": quants,
        "settings": settings,
        "sources": sources,
        "harness_hashes": harness_hashes(),
        "jobs": jobs,
    }
    write_json(output / "run.json", run)
    interrupted = False
    for index, job in enumerate(jobs, 1):
        spec = {
            **job,
            "settings": settings,
            "quants": quants,
            "source": sources.get(job["backend"]),
            "harness_hashes": run["harness_hashes"],
            "output": str(output / job["report"]),
        }
        spec_path = output / "jobs" / f"{job['id']}.json"
        write_json(spec_path, spec)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--node-rank=0",
            f"--nproc-per-node={args.nproc_per_node}",
            "--rdzv-backend=static",
            f"--master-addr={args.master_addr}",
            f"--master-port={args.master_port}",
            str(Path(__file__).resolve()),
            "--worker-spec",
            str(spec_path),
        ]
        logfile = output / job["log"]
        logfile.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(jobs)}] {job['id']} -> {logfile}", flush=True)
        job_error = None
        started = time.monotonic()
        with logfile.open("w") as stream:
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=args.job_timeout)
            except subprocess.TimeoutExpired:
                job_error = (
                    "TIMEOUT",
                    f"Process group exceeded {args.job_timeout} seconds",
                )
                stop_group(process)
            except KeyboardInterrupt:
                job_error = ("INTERRUPTED", "Interrupted by user")
                stop_group(process)
                interrupted = True
        report_path = output / job["report"]
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if job_error:
            report.update(status=job_error[0], error=job_error[1])
        elif code != 0 and report.get("status") in (None, "RUNNING", "PASS"):
            report.update(
                status="ERROR", error=f"torchrun exited {code}; see {job['log']}"
            )
        report["process_group_seconds"] = time.monotonic() - started
        write_json(report_path, report)
        summary = summarize(output)
        print(
            f"  {report.get('status', 'ERROR')}; summary: {output / 'summary.md'}",
            flush=True,
        )
        if interrupted:
            break
    run["status"] = "INTERRUPTED" if interrupted else "FINISHED"
    write_json(output / "run.json", run)
    summary = summarize(output)
    return (
        130
        if interrupted
        else (
            1
            if summary["status"] == "ERROR"
            else 0 if summary["status"] == "PASS" else 2
        )
    )


def input_sample_signature(data, operator):
    """Bounded byte fingerprint, plus generator source hashes in the report."""
    import torch

    tensors = [data.q, data.weights, data.lengths]
    if data.q_scale is not None:
        tensors.append(data.q_scale)
    if operator == "paged":
        tensors += [data.tables, data.cache[:1], data.cache[-1:]]
    else:
        tensors += [data.k[:1], data.k[-1:], data.scales[:1], data.scales[-1:]]
    state = hashlib.sha256()
    for tensor in tensors:
        state.update(
            tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        )
    return state.hexdigest()


def run_worker(spec):
    import torch
    import torch.distributed as dist
    from benchmark_paged_mqa import (
        compare,
        compiler_preflight,
        decode_q,
        environment,
        make_torch_call,
        reference,
        synchronize,
        unpack_cache,
    )
    from benchmark_paged_mqa_distributed import (
        broadcast_query,
        check_global_samples,
        collective_smoke,
        collector,
        gather_objects,
        measure,
        sampled_positions,
    )
    from mqa_suite_inputs import (
        dense_reference,
        dense_sample_reference,
        estimate_memory,
        make_dense_inputs,
        make_dense_torch_call,
        make_operator_call,
    )
    from paged_mqa_distributed_inputs import make_shard_inputs, sample_reference
    from run_paged_mqa_suite import host_available_memory

    settings = spec["settings"]
    args = SimpleNamespace(**{**settings, **spec["case"]})
    rank, local_rank, world = (
        int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    backend, operator = spec["backend"], spec["operator"]
    if world != args.nproc_per_node:
        raise ValueError(
            "Worker world size differs from the requested single-node plan"
        )
    os.environ["GEMS_VENDOR"] = args.vendor
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    for flag in TLE_FLAGS:
        os.environ[flag] = "1" if backend == "tle" else "0"
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    device = (
        torch.device("cuda", local_rank)
        if args.device == "cuda"
        else torch.device("cpu")
    )
    preflight = None
    if device.type == "cuda":
        torch.cuda.set_device(device)
        preflight = compiler_preflight(args.vendor, args.libdevice_path)
    if harness_hashes() != spec["harness_hashes"]:
        raise ValueError("Benchmark/helper sources changed after the run started")
    source_manifest = None
    if spec["source"]:
        source_manifest = verify_snapshot(spec["source"]["directory"])
        if source_manifest["tree_sha256"] != spec["source"]["tree_sha256"]:
            raise ValueError("Source snapshot no longer matches the run manifest")
        src = str(Path(spec["source"]["directory"]) / "src")
        sys.path.insert(0, src)
        os.environ["PYTHONPATH"] = src + os.pathsep + os.environ.get("PYTHONPATH", "")
    dist.init_process_group(
        "nccl" if device.type == "cuda" else "gloo",
        init_method="env://",
        timeout=timedelta(seconds=args.timeout),
    )
    collective_smoke(device, rank, world)
    estimate = estimate_memory(args, operator, world, rank)
    info = {
        "rank": rank,
        "local_rank": local_rank,
        "node": socket.gethostname(),
        "spec_sha256": digest(json.dumps(spec, sort_keys=True).encode()),
        "environment": environment(device),
        "compiler_preflight": preflight,
        "memory_estimate": estimate,
        "host_available_bytes": host_available_memory(),
        "gpu_free_bytes": (
            torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None
        ),
    }
    ranks = gather_objects(info, world)
    if len({item["spec_sha256"] for item in ranks}) != 1:
        raise ValueError("Ranks received different case/protocol/source specifications")
    report = {
        "schema_version": 1,
        "status": "RUNNING",
        "operator": operator,
        "backend": backend,
        "case": spec["case"],
        "world_size": world,
        "settings": settings,
        "harness_hashes": spec["harness_hashes"],
        "source": spec["source"],
        "ranks": ranks,
        "measurement": (
            "max-rank synchronized wall ms per API call, median over repeats; "
            "independent compute/communication/end-to-end experiments"
        ),
        "excluded": "initial KV generation/transfer, compilation, validation, warmup; no CUDA graphs",
        "kv_contract": (
            "shared K[N,D]"
            if operator == "dense"
            else "independent paged K per request"
        ),
        "results": [],
    }

    def save():
        if rank == 0:
            write_json(spec["output"], report)

    save()
    failures = []
    for item in ranks:
        est = item["memory_estimate"]
        if (
            item["gpu_free_bytes"] is not None
            and est["estimated_gpu_bytes"]
            > item["gpu_free_bytes"] * args.memory_fraction
        ):
            failures.append(f"rank {item['rank']}: estimated GPU memory exceeds budget")
        if backend != "torch" and est["local_physical_tokens"] * 128 > 2**31:
            failures.append(
                f"rank {item['rank']}: local FP8 K offsets exceed conservative int32 bound"
            )
    available = [
        item["host_available_bytes"]
        for item in ranks
        if item["host_available_bytes"] is not None
    ]
    if (
        available
        and sum(item["memory_estimate"]["estimated_host_bytes"] for item in ranks)
        > min(available) * args.host_memory_fraction
    ):
        failures.append("Combined host-memory estimate exceeds budget")
    if failures:
        report.update(status="SKIPPED_RESOURCE", error="; ".join(failures))
        report["results"] = [
            {"quant": q, "status": "SKIPPED_RESOURCE", "error": report["error"]}
            for q in spec["quants"]
        ]
        save()
        dist.destroy_process_group()
        return 0

    package = module = None
    if backend != "torch":
        package, module, implementation = load_frozen_operator(
            operator, spec["source"]["directory"], source_manifest
        )
        report["implementation"] = {
            **implementation,
            "tle_environment": {flag: os.environ[flag] for flag in TLE_FLAGS},
            "tle_launcher": getattr(args, f"{operator}_tle_launcher"),
            "tle_evidence_scope": "host launcher observed during untimed validation; not an instruction-level profiler",
        }
        save()
    env_keys = (
        "packages",
        "device_name",
        "device_total_memory_bytes",
        "torch_cuda_build",
        "float32_matmul_precision",
        "TRITON_LIBDEVICE_PATH",
        "TRITON_LIBCUDA_PATH",
    )
    environment_signature = [{k: r["environment"][k] for k in env_keys} for r in ranks]
    with torch.no_grad():
        for quant in spec["quants"]:
            if rank == 0:
                print(
                    f"{operator} {quant} {backend}: building resident KV and independent references ...",
                    flush=True,
                )
            if operator == "paged":
                data, plan = make_shard_inputs(args, quant, rank, world, device)
            else:
                data, plan = make_dense_inputs(args, quant, rank, world, device)
            broadcast_query(data)
            fingerprints = gather_objects(input_sample_signature(data, operator), world)
            if operator == "paged":
                q = decode_q(data.q, data.q_scale)
                k, scales = unpack_cache(data.cache)
                expected_local = reference(
                    q, k, scales, data.weights, data, args.chunk_tokens
                )
                del q, k, scales
            else:
                expected_local = dense_reference(data, args.chunk_tokens)
            positions = sampled_positions(args, world)
            expected_global = (
                sample_reference if operator == "paged" else dense_sample_reference
            )(args, quant, positions)
            collect = collector(args, data, device, world)
            if plan["length"] == 0:
                fn = lambda width=data.width: torch.zeros(
                    (args.batch, width), device=device
                )
            elif backend == "torch":
                factory = (
                    make_torch_call if operator == "paged" else make_dense_torch_call
                )
                fn = factory(
                    data, args.chunk_tokens, compiled=args.torch_mode == "compile"
                )
            else:
                fn = make_operator_call(package, operator, data)
            if rank == 0:
                print(
                    f"{operator} {quant} {backend}: compiling and checking ...",
                    flush=True,
                )
            first, launched = observed_call(
                module, getattr(args, f"{operator}_tle_launcher"), fn
            )
            synchronize(device)
            local_check = compare(first, expected_local, data, args.rtol, args.atol)
            assembled = collect(first)
            global_check = check_global_samples(
                assembled, expected_global, positions, args
            )
            checks = gather_objects(
                {
                    "rank": rank,
                    "local": local_check,
                    "global_samples": global_check,
                    "empty_shard": plan["length"] == 0,
                    "tle_launcher_calls": launched,
                },
                world,
            )
            status = row_status(checks, backend)
            signature = {
                "case": spec["case"],
                "operator": operator,
                "quant": quant,
                "world_size": world,
                "environment": environment_signature,
                "harness_hashes": spec["harness_hashes"],
                "input_sample_signatures": fingerprints,
                "protocol": {
                    key: settings[key]
                    for key in (
                        "device",
                        "vendor",
                        "heads",
                        "page_size",
                        "chunk_tokens",
                        "torch_mode",
                        "warmup",
                        "iterations",
                        "repeats",
                        "seed",
                        "rtol",
                        "atol",
                        "cpu_threads",
                    )
                },
            }
            row = {
                "quant": quant,
                "status": status,
                "correctness": checks,
                "comparison_signature": digest(
                    json.dumps(signature, sort_keys=True).encode()
                ),
                "input_sample_signatures": fingerprints,
                "global_output_shape": [args.batch, args.context],
                "communication": {
                    "query_broadcast_bytes": data.q.numel() * data.q.element_size()
                    + data.weights.numel() * data.weights.element_size()
                    + (
                        data.q_scale.numel() * data.q_scale.element_size()
                        if data.q_scale is not None
                        else 0
                    ),
                    "allgather_local_input_bytes": first.numel() * first.element_size(),
                    "logical_allgather_receive_bytes_per_rank": (world - 1)
                    * first.numel()
                    * first.element_size(),
                    "global_output_bytes_per_rank": args.batch * args.context * 4,
                    "initial_kv_generation_or_transfer_timed": False,
                },
            }
            del expected_local, expected_global, assembled
            if status in ("PASS", "FALLBACK"):

                def communication_only(current=data, gather=collect, cached=first):
                    broadcast_query(current)
                    return gather(cached)

                def end_to_end(current=data, gather=collect, compute=fn):
                    broadcast_query(current)
                    return gather(compute())

                row["timing"] = {
                    "compute_only": measure(fn, args, device, world),
                    "communication_and_assembly_only": measure(
                        communication_only, args, device, world
                    ),
                    "end_to_end": measure(end_to_end, args, device, world),
                }
                del communication_only, end_to_end
            report["results"].append(row)
            save()
            if rank == 0:
                times = row.get("timing", {})
                print(
                    f"{quant}: {status}; end_to_end_ms={times.get('end_to_end', {}).get('critical_ms_median')}",
                    flush=True,
                )
            del fn, collect, first, data
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            dist.barrier()
    statuses = {row["status"] for row in report["results"]}
    report["status"] = (
        "ERROR"
        if statuses & {"INCORRECT", "BASELINE_USED_TLE"}
        else "PASS" if statuses == {"PASS"} else "PARTIAL"
    )
    save()
    dist.barrier()
    dist.destroy_process_group()
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.freeze_baseline:
        manifest = freeze_source(args.candidate_src, args.freeze_baseline)
        print(
            f"Frozen original Gems: {args.freeze_baseline.resolve()}\nSHA256: {manifest['tree_sha256']}"
        )
        return 0
    if args.summarize_only:
        summary = summarize(args.summarize_only)
        print(f"{summary['status']}: {args.summarize_only / 'summary.md'}")
        return (
            1
            if summary["status"] == "ERROR"
            else 0 if summary["status"] == "PASS" else 2
        )
    if args.worker_spec:
        spec = json.loads(args.worker_spec.read_text())
        try:
            return run_worker(spec)
        except Exception:
            error = traceback.format_exc()
            rank = int(os.environ.get("RANK", "0"))
            output = Path(spec["output"])
            write_json(
                output.with_name(output.stem + f".rank{rank}.error.json"),
                {"rank": rank, "status": "ERROR", "error": error},
            )
            if rank == 0:
                report = json.loads(output.read_text()) if output.exists() else {}
                report.update(status="ERROR", error=error)
                write_json(output, report)
            print(error, file=sys.stderr, flush=True)
            return 1
    if args.plan:
        print("| Size | M/B | N/L (global) | N/L capacity per rank | H | D | Page |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for case in args.cases:
            pages = math.ceil(case["context"] / args.page_size)
            local = math.ceil(pages / args.nproc_per_node) * args.page_size
            print(
                f"| {case['id']} | {case['batch']} | {case['context']} | {local} | "
                f"{args.heads} | 128 | {args.page_size} |"
            )
        quants = 2 if args.quant == "both" else 1
        print(
            f"\n{len(args.cases) * len(args.operators) * quants} op/shape/quant cases, "
            f"{len(args.backends)} backends, "
            f"{args.nproc_per_node} ranks working together per case."
        )
        return 0
    return run_suite(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
