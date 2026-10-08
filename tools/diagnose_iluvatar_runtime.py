#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded CoreX diagnostics without changing the container or importing MQA.

Each Runtime/Torch/Driver probe runs in a fresh process. SDK experiments change
only that child's library search order, and actual loaded libraries are recorded.
The default probes only device 0 within the existing visibility allocation.
"""

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

# The downloadable single-file edition embeds the same worker source here.
WORKER_SOURCE = None


def library_environments(environ, roots):
    """Prefer one SDK without mutating the caller's environment or removing deps."""
    variants = [("current", dict(environ), None)]
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        resolved = str(root.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        priority = [
            str(root / sub) for sub in ("lib64", "lib") if (root / sub).is_dir()
        ]
        if not priority:
            continue
        # Keep the image's cuDNN/cuBLAS fallback directories: copied SDKs may
        # contain only a runtime patch, not a complete deep-learning stack.
        paths = priority + environ.get("LD_LIBRARY_PATH", "").split(":")
        env = dict(environ)
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in paths if p))
        variants.append((f"prefer-{root.name}", env, resolved))
    return variants


def run_command(command, env, log, timeout):
    """Bound only our own process group; abort if even SIGKILL cannot reap it."""
    started = time.monotonic()
    with Path(log).open("w") as stream:
        try:
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return {"status": "NOT_RUN", "error": str(exc), "seconds": 0}
        timed_out = False
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            # The group can still contain compiler/helper descendants even if
            # its leader exited after TERM. Always complete group cleanup.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            code = proc.poll()
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            raise
    status = "TIMEOUT" if timed_out else "EXITED"
    if code is None:
        status = "KILL_PENDING"
    return {
        "status": status,
        "returncode": code,
        "pid": proc.pid,
        "seconds": round(time.monotonic() - started, 2),
    }


def read_events(path):
    events = []
    if not Path(path).exists():
        return events
    with Path(path).open(errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and "stage" in event and "event" in event:
                events.append(event)
    return events


def classify(result, events):
    result = dict(result)
    progress = [e for e in events if e["event"] in ("BEGIN", "RETURN", "SYNC")]
    result["last_stage"] = (
        progress[-1]["stage"] if progress else "worker startup/import"
    )
    errors = [e for e in events if e["event"] == "ERROR"]
    result["events"] = events
    paths = next((e["paths"] for e in reversed(events) if "paths" in e), [])
    loaded = []
    for raw_path in paths:
        path = Path(raw_path.removesuffix(" (deleted)"))
        if path.name.startswith(
            ("libcuda", "libix", "libcublas", "libcudnn", "libnccl", "libcuinfer")
        ):
            entry = {"path": raw_path, "resolved": str(path.resolve())}
            try:
                info = path.stat()
                entry.update(device=info.st_dev, inode=info.st_ino, size=info.st_size)
            except OSError:
                pass
            loaded.append(entry)
    result["loaded_libraries"] = loaded
    if result["status"] == "EXITED":
        passed = any(e["stage"] == "probe" and e["event"] == "PASS" for e in events)
        if result["returncode"] == 0 and passed:
            result["status"] = "PASS"
        else:
            result["status"] = (
                errors[-1].get("category", "ERROR") if errors else "ERROR"
            )
    return result


def inventory(output, roots):
    data = {
        "python": sys.executable,
        "platform": sys.platform,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "IX_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "TRITON_LIBCUDA_PATH",
                "TRITON_LIBDEVICE_PATH",
            )
        },
        "libraries": [],
        "device_nodes": [],
    }
    for root in roots:
        root = Path(root)
        for sub in ("lib", "lib64"):
            for pattern in (
                "libcuda.so*",
                "libcudart.so*",
                "libixthunk.so*",
                "libixml.so*",
                "libcudnn.so*",
            ):
                for path in (root / sub).glob(pattern):
                    try:
                        info = path.stat()
                        data["libraries"].append(
                            {
                                "path": str(path),
                                "resolved": str(path.resolve()),
                                "device": info.st_dev,
                                "inode": info.st_ino,
                                "size": info.st_size,
                            }
                        )
                    except OSError as exc:
                        data["libraries"].append({"path": str(path), "error": str(exc)})
    for path in [Path("/dev/itrctl"), *Path("/dev").glob("iluvatar*")]:
        if path.exists():
            info = path.stat()
            data["device_nodes"].append(
                {"path": str(path), "mode": stat.filemode(info.st_mode)}
            )
    commands = {
        "ixsmi": ["ixsmi"],
        "processes": ["ps", "-eo", "pid,ppid,etime,stat,comm"],
        "kernel_warnings": ["dmesg", "--level=err,warn"],
    }
    data["commands"] = {}
    for name, command in commands.items():
        log = output / f"{name}.log"
        result = run_command(command, dict(os.environ), log, 8)
        result["log"] = log.name
        if name == "kernel_warnings" and log.exists():
            lines = log.read_text(errors="replace").splitlines()
            relevant = [
                line
                for line in lines
                if any(
                    word in line.lower()
                    for word in (
                        "iluvatar",
                        "ix",
                        "gpu",
                        "pcie",
                        "iommu",
                        "denied",
                        "permitted",
                    )
                )
            ]
            log.write_text("\n".join((relevant or lines)[-100:]) + "\n")
        data["commands"][name] = result
    return data


def write_reports(output, report):
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "CoreX runtime diagnostic report",
        "",
        "Only child-process library search paths changed. No system changes or GPU resets.",
        "PASS requires the probe to finish; Torch PASS includes a tiny FP32 compute/readback check.",
        "SDK preference is not proof of isolation; inspect each probe's actual library maps.",
        "Mixed SDK paths alone do not establish incompatibility.",
        "",
    ]
    for result in report["results"]:
        lines.append(
            f"device={result['device']} {result['variant']} {result['probe']}: "
            f"{result['status']} [{result.get('last_stage', '')}] {result.get('seconds', 0)}s"
        )
        if result.get("same_library_map_as_current"):
            lines.append(
                "  Same relevant loaded library files as current: not an independent SDK comparison."
            )
    passes = [
        r for r in report["results"] if r["probe"] == "torch" and r["status"] == "PASS"
    ]
    if passes:
        lines += [
            "",
            "Torch compute passed in these process environments (MQA/TLE remains unverified):",
        ]
        for result in passes:
            lines.append(f"  device={result['device']}; variant={result['variant']}")
            lines.append(f"  LD_LIBRARY_PATH={result['library_path']}")
    else:
        lines += [
            "",
            "No tested Torch environment completed the compute check.",
            "Give this report plus per-probe logs to the platform maintainer; root cause is not established.",
        ]
    lines += [
        "",
        "Driver probe uses a private context; Torch/Runtime may use a primary context.",
        "Driver success alone does not prove the Torch initialization path is healthy.",
    ]
    if report.get("aborted"):
        lines += ["", f"Further probes stopped: {report['aborted']}"]
    (output / "summary.txt").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        default=[0],
        help="Only process-visible device ordinals you are authorized to use",
    )
    parser.add_argument(
        "--sdk-roots",
        nargs="+",
        default=[
            "/usr/local/corex-4.5.0.20260509",
            "/usr/local/corex-4.5.0",
            "/usr/local/corex",
        ],
    )
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout < 2 or any(device < 0 for device in args.devices):
        parser.error(
            "timeout must be >=2 seconds and device ordinals must be nonnegative"
        )
    variants = library_environments(os.environ, args.sdk_roots)
    if args.plan:
        print(
            json.dumps(
                {
                    "devices": args.devices,
                    "variants": [v[0] for v in variants],
                    "probes": ["runtime", "torch", "driver only if runtime fails"],
                },
                indent=2,
            )
        )
        return 0
    output = args.output or Path(
        f"/tmp/corex-diagnostic-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    worker = output / "probe_worker.py"
    source = WORKER_SOURCE
    if source is None:
        source = (
            Path(__file__).with_name("diagnose_iluvatar_runtime_worker.py").read_text()
        )
    worker.write_text(source)
    print(f"Report directory: {output}", flush=True)
    report = {
        "inventory": inventory(output, args.sdk_roots),
        "results": [],
        "worker_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "devices": args.devices,
    }
    abort = any(
        r["status"] == "KILL_PENDING" for r in report["inventory"]["commands"].values()
    )
    if abort:
        report["aborted"] = (
            "A diagnostic command could not be reaped after SIGKILL. No GPU probes started."
        )
    try:
        for device in dict.fromkeys(args.devices):
            if abort:
                break
            for name, env, root in variants:
                if abort:
                    break
                probes = ["runtime", "torch"]
                for probe in probes:
                    label = f"device{device}-{name}-{probe}"
                    print(f"BEGIN {label} (timeout {args.timeout}s)", flush=True)
                    log = output / f"{label}.log"
                    command = [
                        sys.executable,
                        "-u",
                        str(worker),
                        "--probe",
                        probe,
                        "--device",
                        str(device),
                        "--trace-timeout",
                        str(max(1, args.timeout // 2)),
                    ]
                    result = classify(
                        run_command(command, env, log, args.timeout), read_events(log)
                    )
                    result.update(
                        device=device,
                        variant=name,
                        requested_sdk=root,
                        probe=probe,
                        library_path=env.get("LD_LIBRARY_PATH"),
                        log=log.name,
                    )
                    current = next(
                        (
                            r
                            for r in report["results"]
                            if r["device"] == device
                            and r["probe"] == probe
                            and r["variant"] == "current"
                        ),
                        None,
                    )
                    if current and result["loaded_libraries"]:

                        def identities(item):
                            return sorted(
                                (p.get("resolved"), p.get("device"), p.get("inode"))
                                for p in item["loaded_libraries"]
                            )

                        result["same_library_map_as_current"] = identities(
                            result
                        ) == identities(current)
                    report["results"].append(result)
                    print(f"  {result['status']} at {result['last_stage']}", flush=True)
                    if probe == "runtime" and result["status"] != "PASS":
                        probes.append("driver")
                    if result["status"] == "KILL_PENDING":
                        report["aborted"] = (
                            "A GPU probe could not be reaped after SIGKILL; no more probes will be launched."
                        )
                        abort = True
                    write_reports(output, report)
                    if abort:
                        break
    except KeyboardInterrupt:
        report["aborted"] = "Interrupted by user; completed logs retained."
    finally:
        write_reports(output, report)
        archive = output.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(output, arcname=output.name)
        print((output / "summary.txt").read_text(), flush=True)
        print(f"Evidence archive: {archive}", flush=True)
    return (
        0
        if not report.get("aborted")
        and any(
            r["probe"] == "torch" and r["status"] == "PASS" for r in report["results"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
