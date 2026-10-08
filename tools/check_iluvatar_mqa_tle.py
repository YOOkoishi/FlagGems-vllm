#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate experimental Iluvatar MQA dispatch, TLE execution and edge cases.

Run from the repository's CoreX/FlagTree environment::

    python tools/check_iluvatar_mqa_tle.py --output /tmp/mqa-tle-check.json
    python tools/check_iluvatar_mqa_tle.py --bench

``--cpu-only`` checks fixtures and FP32 references only. It neither imports the
operator package nor validates a Triton kernel. GPU mode fails if the actual
compiler backend is not Iluvatar, the public API is not replaced, or an enabled
TLE call does not reach its launcher. No vLLM dependency is required by this tool.

Optional timings compare the same candidate algorithm with staging off and on.
Each mode autotunes independently; its selected configuration is recorded. This
does not compare against the original generic operator; use the full
benchmark_mqa_suite.py source-snapshot comparison for that separate question.
"""

import argparse
import faulthandler
import importlib
import inspect
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch
from benchmark_paged_mqa import (
    benchmark,
    compiler_preflight,
    decode_q,
    make_inputs,
    pack_cache,
    reference,
    synchronize,
    torch_chunk,
    unpack_cache,
)
from mqa_suite_inputs import dense_reference, make_dense_inputs
from paged_mqa_distributed_inputs import make_queries

OPERATORS = {
    "dense": "fp8_fp4_mqa_logits",
    "paged": "fp8_fp4_paged_mqa_logits",
}
SWITCHES = {
    "dense": "FLAGGEMS_FP8_FP4_MQA_LOGITS_TLE",
    "paged": "FLAGGEMS_FP8_FP4_PAGED_MQA_LOGITS_TLE",
}


def progress(message):
    print(f"[MQA] {message}", flush=True)


@dataclass
class Case:
    name: str
    operator: str
    kwargs: dict
    expected: torch.Tensor
    computed: torch.Tensor
    invalid_fill: float

    def on_device(self, device):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            return value

        return replace(
            self,
            kwargs={key: move(value) for key, value in self.kwargs.items()},
            expected=self.expected.to(device),
            computed=self.computed.to(device),
        )


def check_output(actual, case, rtol, atol):
    """Check finite computed values and exact invalid fills separately."""
    if actual.shape != case.expected.shape or actual.dtype != torch.float32:
        raise AssertionError(
            f"Expected FP32 {tuple(case.expected.shape)}, got "
            f"{actual.dtype} {tuple(actual.shape)}"
        )
    values, expected = actual[case.computed], case.expected[case.computed]
    if not bool(torch.isfinite(values).all()):
        raise AssertionError("Computed output contains NaN/Inf")
    torch.testing.assert_close(values, expected, rtol=rtol, atol=atol)
    invalid = actual[~case.computed]
    if case.invalid_fill == float("-inf"):
        if not bool(torch.isneginf(invalid).all()):
            raise AssertionError("Invalid output positions must be exactly -inf")
    elif not bool((invalid == case.invalid_fill).all()):
        raise AssertionError(f"Invalid output must be exactly {case.invalid_fill}")
    return {
        "computed_elements": values.numel(),
        "invalid_elements": invalid.numel(),
        "max_abs": (values - expected).abs().max().item() if values.numel() else 0,
    }


def dense_fixture(quant, rows, heads, width, clean):
    args = SimpleNamespace(
        batch=rows, heads=heads, context=width, page_size=1, seed=20261008
    )
    data, _ = make_dense_inputs(args, quant, 0, 1, torch.device("cpu"))
    starts = [0] if rows == 1 else [0, 7, width // 2]
    ends = [max(0, width - 3)] if rows == 1 else [0, width - 2, width]
    data.starts = torch.tensor(starts, dtype=torch.int32)
    data.lengths = torch.tensor(ends, dtype=torch.int32)
    # The dense clean=False contract computes ALL columns, including outside
    # the semantic [start, end) range. dense_reference's full contexts does so.
    expected = dense_reference(data, 64)
    q = decode_q(data.q, data.q_scale)
    dots = torch.einsum("mhd,nd->mhn", q, data.k.float())
    alternate = (
        torch.relu(dots * data.scales[None, None, :]) * data.weights[:, :, None]
    ).sum(1)
    torch.testing.assert_close(alternate, expected, rtol=2e-5, atol=2e-4)
    positions = torch.arange(width)[None, :]
    semantic = (positions >= data.starts[:, None]) & (positions < data.lengths[:, None])
    computed = semantic if clean else torch.ones_like(semantic)
    expected = expected.masked_fill(~computed, float("-inf"))
    case = Case(
        name=f"dense/{quant}/M{rows}/H{heads}/N{width}/clean{int(clean)}",
        operator="dense",
        kwargs={
            "q": (data.q[:, 0], None if data.q_scale is None else data.q_scale[:, 0]),
            "kv": (data.k, data.scales),
            "weights": data.weights,
            "cu_seqlen_ks": data.starts,
            "cu_seqlen_ke": data.lengths,
            "clean_logits": clean,
        },
        expected=expected,
        computed=computed,
        invalid_fill=float("-inf"),
    )
    check_output(expected, case, 0, 0)
    return case


def paged_fixture(quant, page, heads, next_n, context_rank, clean):
    width = 2 * page + 7
    if context_rank == 2:
        contexts = [[0, 1], [page - 1, page + 1], [2 * page + 3, 2 * page + 1]]
        contexts = [row[:next_n] for row in contexts]
        lengths = torch.tensor(contexts, dtype=torch.int32)
        flat_lengths = lengths.flatten()
        capacities = lengths.amax(1).tolist()
    else:
        capacities = [0, page - 1, 2 * page + 3]
        lengths = torch.tensor(capacities, dtype=torch.int32)
        flat_lengths = lengths.repeat_interleave(next_n)
    args = SimpleNamespace(
        batch=3,
        heads=heads,
        context=width,
        contexts=capacities,
        max_model_len=width,
        page_size=page,
        cache_pages=None,
        seed=20261008,
    )
    data = make_inputs(args, quant, torch.device("cpu"))
    # Force a nonidentity physical mapping, including for the tiny fixture.
    # Existing make_inputs randomizes pages; rotating cache and table together
    # preserves every logical value and guarantees a visible page permutation.
    count = data.cache.shape[0]
    occupied = torch.cat(
        [data.tables[i, : (n + page - 1) // page] for i, n in enumerate(capacities)]
    )
    if torch.equal(occupied, torch.arange(count, dtype=torch.int32)):
        data.cache = data.cache.roll(1, 0)
        data.tables = (data.tables + 1) % count
    k, scales = unpack_cache(data.cache)
    repacked = pack_cache(k.to(torch.float8_e4m3fn), scales)
    torch.testing.assert_close(repacked, data.cache, rtol=0, atol=0)
    if not data.cache.is_contiguous() or data.cache.shape[-2:] != (1, 132):
        raise AssertionError("Fixture must use contiguous packed [P,page,1,132]")

    query_args = SimpleNamespace(**vars(args))
    query_args.batch = 3 * next_n
    q, q_scale, weights = make_queries(query_args, quant)
    q_public = q.reshape(3, next_n, heads, q.shape[-1])
    scales_public = None if q_scale is None else q_scale.reshape(3, next_n, heads)
    flattened = replace(
        data,
        q=q,
        q_scale=q_scale,
        weights=weights,
        tables=data.tables.repeat_interleave(next_n, 0),
        lengths=flat_lengths,
        contexts=flat_lengths.tolist(),
    )
    decoded = decode_q(q, q_scale)
    expected = reference(decoded, k, scales, weights, flattened, 64)
    positions = torch.arange(width)
    alternate = torch_chunk(
        decoded, k, scales, weights, flattened.tables, flat_lengths, positions
    )
    torch.testing.assert_close(alternate, expected, rtol=2e-5, atol=2e-4)
    computed = positions[None, :] < flat_lengths[:, None]
    fill = float("-inf") if clean else 0.0
    expected = expected.masked_fill(~computed, fill)
    case = Case(
        name=(
            f"paged/{quant}/page{page}/H{heads}/next{next_n}/"
            f"ctx{context_rank}D/clean{int(clean)}"
        ),
        operator="paged",
        kwargs={
            "q": (q_public, scales_public),
            "kv_cache": data.cache,
            "weights": weights,
            "context_lens": lengths,
            "block_tables": data.tables,
            "schedule_metadata": None,
            "max_model_len": width,
            "clean_logits": clean,
        },
        expected=expected,
        computed=computed,
        invalid_fill=fill,
    )
    check_output(expected, case, 0, 0)
    return case


def fixtures(args):
    quants = ("fp8", "fp4") if args.quant == "both" else (args.quant,)
    dense_shapes = ((1, 1, 1), (1, 16, 67), (3, 64, 257))
    paged_shapes = ((1, 1, 1), (16, 2, 1), (64, 2, 2))
    if args.quick:
        dense_shapes = dense_shapes[1:]
        paged_shapes = paged_shapes[1:]
    for quant in quants:
        for clean in (False, True):
            if args.operator in ("both", "dense"):
                for rows, heads, width in dense_shapes:
                    yield dense_fixture(quant, rows, heads, width, clean)
            if args.operator in ("both", "paged"):
                for page in args.page_sizes:
                    for heads, next_n, context_rank in paged_shapes:
                        yield paged_fixture(
                            quant, page, heads, next_n, context_rank, clean
                        )


def load_operators(args, device):
    progress(f"Checking CUDA/CoreX device {device}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "GPU validation requires an Iluvatar CoreX device; "
            "--cpu-only checks fixtures/references only"
        )
    progress("Selecting device")
    torch.cuda.set_device(device)
    os.environ["GEMS_VENDOR"] = "iluvatar"
    os.environ["FLAGGEMS_ILUVATAR_MQA_EXPERIMENTAL"] = "1"
    from triton.compiler.compiler import make_backend
    from triton.runtime import driver

    progress("Resolving the active Triton compiler")
    target = driver.active.get_current_target()
    backend = make_backend(target)
    compiler = type(backend)
    source = inspect.getsourcefile(compiler) or ""
    identity = f"{compiler.__module__}.{compiler.__name__} {source}"
    if "iluvatar" not in identity.lower():
        raise RuntimeError(f"Actual compiler is not Iluvatar: {identity}; {target}")
    progress(f"Compiler: {identity}; target: {target}")
    progress("Checking CoreX libdevice")
    preflight = compiler_preflight("iluvatar", args.libdevice_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    progress("Importing flaggems_vllm (includes other operator modules)")
    package = importlib.import_module("flaggems_vllm")
    progress("Package import complete; checking public operator dispatch")
    functions = {}
    for operator, name in OPERATORS.items():
        if args.operator not in ("both", operator):
            continue
        fn = getattr(package, name)
        if "._iluvatar.fused." not in fn.__module__:
            raise RuntimeError(f"Top-level {name} is not replaced: {fn.__module__}")
        module = importlib.import_module(fn.__module__)
        if not callable(getattr(module, "_launch_tle_kernel", None)):
            raise RuntimeError(f"Missing observable TLE launcher in {fn.__module__}")
        functions[operator] = (fn, module)
        progress(f"{name} -> {fn.__module__}")
    return functions, {
        "device": torch.cuda.get_device_name(device),
        "target": str(target),
        "compiler": identity,
        "preflight": preflight,
        "public_apis": {key: fn.__module__ for key, (fn, _) in functions.items()},
    }


def observed_call(module, fn):
    """Trace one untimed host launcher call and restore it even on failure."""
    original = module._launch_tle_kernel
    calls = 0

    def traced(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    module._launch_tle_kernel = traced
    try:
        value = fn()
    finally:
        module._launch_tle_kernel = original
    return value, calls


def validate_host_contract(functions, device):
    """Empty output and rejected formats must not reach the TLE launcher."""
    results = []
    for operator, (fn, module) in functions.items():
        progress(f"{operator}: preparing empty/unsupported-input checks")
        if operator == "dense":
            base = dense_fixture("fp8", 1, 16, 67, True).on_device(device).kwargs
            empty_rows = dict(base)
            empty_rows["q"] = (base["q"][0][:0], None)
            for name in ("weights", "cu_seqlen_ks", "cu_seqlen_ke"):
                empty_rows[name] = base[name][:0]
            empty_width = dict(base)
            empty_width["kv"] = tuple(tensor[:0] for tensor in base["kv"])
            empty_checks = (
                ("M0", empty_rows, (0, 67)),
                ("N0", empty_width, (1, 0)),
            )
            unsupported_dim = dict(base)
            unsupported_dim["q"] = (base["q"][0][..., :64], None)
            unsupported_dim["kv"] = (base["kv"][0][:, :64], base["kv"][1])
            rejects = [("D64", unsupported_dim)]
        else:
            base = paged_fixture("fp8", 16, 16, 2, 2, True).on_device(device).kwargs
            empty_rows = dict(base)
            empty_rows["q"] = (base["q"][0][:0], None)
            for name in ("weights", "context_lens", "block_tables"):
                empty_rows[name] = base[name][:0]
            empty_width = dict(base)
            empty_width["max_model_len"] = 0
            empty_width["context_lens"] = torch.zeros_like(base["context_lens"])
            empty_width["block_tables"] = base["block_tables"][:, :0]
            empty_checks = (
                ("B0", empty_rows, (0, base["max_model_len"])),
                ("N0", empty_width, (6, 0)),
            )
            cache = base["kv_cache"]
            misaligned = torch.empty(
                cache.numel() + 1, dtype=torch.uint8, device=device
            )[1:].view(cache.shape)
            if misaligned.storage_offset() != 1 or not misaligned.is_contiguous():
                raise AssertionError("Misalignment fixture has wrong storage metadata")
            rejects = [("cache_offset1", {**base, "kv_cache": misaligned})]
        rejects.append(
            ("Q_float16", {**base, "q": (base["q"][0].to(torch.float16), None)})
        )
        for mode, enabled in (("plain", False), ("tle", True)):
            os.environ[SWITCHES[operator]] = "1" if enabled else "0"
            for name, kwargs, shape in empty_checks:
                progress(f"{operator}/{mode}/{name}: checking empty output")
                output, calls = observed_call(module, lambda: fn(**kwargs))
                if (
                    tuple(output.shape) != shape
                    or output.dtype != torch.float32
                    or output.device != base["q"][0].device
                    or calls
                ):
                    raise AssertionError(
                        f"{operator}/{mode}/{name}: invalid empty path"
                    )
                results.append(f"{operator}/{mode}/{name}: empty, no TLE launch")
            for name, kwargs in rejects:
                progress(f"{operator}/{mode}/{name}: checking rejection")

                def rejected_call():
                    try:
                        fn(**kwargs)
                    except NotImplementedError:
                        return
                    raise AssertionError(
                        f"{operator}/{name} must raise NotImplementedError"
                    )

                _, calls = observed_call(module, rejected_call)
                if calls:
                    raise AssertionError(
                        f"{operator}/{name}: rejected input launched TLE"
                    )
                results.append(f"{operator}/{mode}/{name}: NotImplementedError")
    return results


def selected_config(module, operator):
    kernel = getattr(module, f"_{operator}_mqa_kernel")
    return str(getattr(kernel, "best_config", "UNAVAILABLE"))


def validate_case(case, functions, args, device):
    fn, module = functions[case.operator]
    run = lambda: fn(**case.kwargs)
    outputs, checks = {}, {}
    for mode, enabled in (("plain", False), ("tle", True)):
        os.environ[SWITCHES[case.operator]] = "1" if enabled else "0"
        progress(f"{case.name}: {mode} launch (first call may compile/autotune)")
        output, calls = observed_call(module, run)
        synchronize(device)
        if (enabled and calls == 0) or (not enabled and calls != 0):
            raise AssertionError(f"{mode}: unexpected TLE launcher count {calls}")
        checks[mode] = check_output(output, case, args.rtol, args.atol)
        checks[mode]["tle_launcher_calls"] = calls
        checks[mode]["best_config"] = selected_config(module, case.operator)
        progress(f"{mode}: correctness PASS; config={checks[mode]['best_config']}")
        outputs[mode] = output
    checks["tle_vs_plain"] = check_output(
        outputs["tle"], replace(case, expected=outputs["plain"]), args.rtol, args.atol
    )
    if args.bench:
        timings = {}
        # All compilation, references, output checks and launcher wrapping above
        # are outside the timing windows. Only the public operator call is timed.
        for mode, enabled in (("plain", False), ("tle", True)):
            os.environ[SWITCHES[case.operator]] = "1" if enabled else "0"
            progress(f"{case.name}: timing {mode}")
            timings[mode] = benchmark(
                run, device, args.warmup, args.iterations, args.repeats
            )
            timings[mode]["best_config"] = selected_config(module, case.operator)
        plain = timings["plain"]["device_timeline_us_median"]
        tle = timings["tle"]["device_timeline_us_median"]
        timings["plain_over_tle"] = plain / tle
        timings["independently_autotuned"] = True
        checks["benchmark"] = timings
    return checks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--operator", choices=("both", "dense", "paged"), default="both"
    )
    parser.add_argument("--quant", choices=("both", "fp8", "fp4"), default="both")
    parser.add_argument(
        "--page-sizes",
        nargs="+",
        type=int,
        choices=(16, 64, 256),
        default=[16, 64, 256],
    )
    parser.add_argument("--quick", action="store_true", help="Omit H=1 cases")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--libdevice-path", type=Path)
    parser.add_argument(
        "--trace-timeout",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Diagnostic only: periodically dump Python stacks; 0 disables it",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.cpu_only and args.bench:
        parser.error("--cpu-only cannot produce kernel timings")
    if min(args.iterations, args.repeats) <= 0 or args.warmup < 0:
        parser.error("iterations/repeats must be positive and warmup nonnegative")
    if args.rtol < 0 or args.atol < 0:
        parser.error("tolerances must be nonnegative")
    if args.trace_timeout < 0:
        parser.error("trace-timeout must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.trace_timeout:
        faulthandler.enable()
        faulthandler.dump_traceback_later(args.trace_timeout, repeat=True)
        progress(f"Diagnostic stack dumps enabled every {args.trace_timeout}s")
    device = torch.device("cpu" if args.cpu_only else args.device)
    progress("Configuring Torch CPU threads and reference precision")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_float32_matmul_precision("highest")
    report = {
        "status": "RUNNING",
        "cpu_only": args.cpu_only,
        "kernel_validation_completed": False,
        "rtol": args.rtol,
        "atol": args.atol,
        "benchmark_scope": "same candidate algorithm, independently autotuned plain/TLE",
        "cases": [],
    }
    try:
        functions = {}
        if not args.cpu_only:
            functions, report["environment"] = load_operators(args, device)
        with torch.inference_mode():
            if not args.cpu_only:
                progress("Checking empty/unsupported inputs before kernel tests")
                report["host_contract_checks"] = validate_host_contract(
                    functions, device
                )
            progress("Generating fixtures and starting correctness checks")
            for fixture in fixtures(args):
                print(fixture.name, flush=True)
                row = {"name": fixture.name, "status": "RUNNING"}
                report["cases"].append(row)
                if args.cpu_only:
                    row["fixture_checks"] = check_output(
                        fixture.expected, fixture, 0, 0
                    )
                    row["status"] = "CPU_FIXTURE_REFERENCE_PASS"
                else:
                    row["checks"] = validate_case(
                        fixture.on_device(device), functions, args, device
                    )
                    row["status"] = "KERNEL_PASS"
        report["status"] = "CPU_FIXTURES_ONLY" if args.cpu_only else "KERNEL_PASS"
        report["kernel_validation_completed"] = not args.cpu_only
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["cases"] and report["cases"][-1]["status"] == "RUNNING":
            report["cases"][-1]["status"] = "FAILED"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        if args.trace_timeout:
            faulthandler.cancel_dump_traceback_later()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.cpu_only and report["status"] != "FAILED":
        print(
            f"CPU fixtures/references passed: {len(report['cases'])}; no kernels executed"
        )
    else:
        print(f"{report['status']}: {len(report['cases'])} cases")
    return int(report["status"] == "FAILED")


if __name__ == "__main__":
    raise SystemExit(main())
