#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check torch.compile/Gems valid logits against hand-computed exact answers.

python tools/check_paged_mqa_equivalence.py
python tools/check_paged_mqa_equivalence.py --cpu-only --output /tmp/mqa-fixtures.json

This checks next_n=1, H=64, D=128 with two active heads per request and one
channel per active head. The four requests cover different head pairs and all
four 32-channel groups. It does not cover every dtype/layout or numerical edge.
CPU-only verifies the fixtures/Torch reference, not either GPU implementation.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from benchmark_paged_mqa import (
    Inputs,
    compare,
    compiler_preflight,
    decode_q,
    make_gems_call,
    make_torch_call,
    pack_cache,
    reference,
    synchronize,
    unpack_cache,
)

HEAD_PAIRS = ((0, 1), (16, 17), (32, 33), (62, 63))
CHANNEL_PAIRS = ((0, 1), (32, 33), (64, 65), (96, 97))
BASE_VARIANTS = ("base", "permuted_pages", "negative_q", "zero_q")


def fixture(quant, page, variant, device):
    if quant not in ("fp8", "fp4"):
        raise ValueError(f"Unsupported query format: {quant}")
    if variant not in BASE_VARIANTS + ("nonunit_q_scale",):
        raise ValueError(f"Unknown fixture variant: {variant}")
    if variant == "nonunit_q_scale" and quant != "fp4":
        raise ValueError("nonunit_q_scale is an FP4 fixture")
    contexts = [1, page - 1, page, page + 1]
    batch, heads, dim = 4, 64, 128
    width = max(contexts) + 3
    pages = sum((length + page - 1) // page for length in contexts)
    rng = torch.Generator().manual_seed(7)
    physical_pages = torch.randperm(pages, generator=rng).to(torch.int32)
    tables = torch.zeros((batch, 2), dtype=torch.int32)
    cursor = 0
    for row, length in enumerate(contexts):
        count = (length + page - 1) // page
        tables[row, :count] = physical_pages[cursor : cursor + count]
        cursor += count

    q = torch.zeros((batch, 1, heads, dim), dtype=torch.float32)
    weights = torch.zeros((batch, heads), dtype=torch.float32)
    for row, ((h0, h1), (d0, d1)) in enumerate(zip(HEAD_PAIRS, CHANNEL_PAIRS)):
        q[row, 0, h0, d0] = 1
        q[row, 0, h1, d1] = -1 if variant == "negative_q" else 1
        weights[row, h0], weights[row, h1] = 2, -1
    if variant == "zero_q":
        q.zero_()
    if quant == "fp8":
        q_values, q_scale = q.to(torch.float8_e4m3fn), None
    else:
        q_values = torch.zeros((batch, 1, heads, dim // 2), dtype=torch.uint8)
        scale_codes = torch.full((batch, 1, heads, 4), 127, dtype=torch.int64)
        if variant != "zero_q":
            # E2M1 code 2 is 1; code 1 is 0.5. The latter times scale 2
            # preserves Q while checking that every packed scale byte is read.
            code = 1 if variant == "nonunit_q_scale" else 2
            for row, ((h0, h1), (d0, d1)) in enumerate(zip(HEAD_PAIRS, CHANNEL_PAIRS)):
                q_values[row, 0, h0, d0 // 2] = code << (4 * (d0 % 2))
                signed_code = code | (8 if variant == "negative_q" else 0)
                q_values[row, 0, h1, d1 // 2] = signed_code << (4 * (d1 % 2))
                if variant == "nonunit_q_scale":
                    scale_codes[row, 0, h0, d0 // 32] = 128
                    scale_codes[row, 0, h1, d1 // 32] = 128
        shifts = 8 * torch.arange(4, dtype=torch.int64)
        q_scale = (scale_codes << shifts).sum(-1).to(torch.int32)
        if variant == "nonunit_q_scale":
            assert (
                q_scale[3, 0, 62:64] < 0
            ).all(), "High scale byte must exercise signed int32"

    decoded_q = decode_q(q_values, q_scale)
    if not torch.equal(decoded_q, q[:, 0]):
        raise AssertionError("Packed query does not decode exactly to the hand-built Q")
    if not torch.equal(decoded_q.to(torch.bfloat16).float(), q[:, 0]):
        raise AssertionError("Query changes during native BF16 preparation")

    pattern = torch.tensor([[-1, 3], [2, 1], [0, -2], [1, 2], [3, 4], [-2, -3]])
    token = torch.arange(pages * page)
    keys = torch.zeros((pages * page, dim), dtype=torch.float32)
    for d0, d1 in CHANNEL_PAIRS:
        keys[:, d0 : d1 + 1] = pattern[token % len(pattern)].float()
    # Every key pattern experiences every scale; all values remain exact in BF16.
    scales = torch.tensor([0.5, 1.0, 2.0])[(token // len(pattern)) % 3]
    keys = keys.reshape(pages, page, dim)
    scales = scales.reshape(pages, page)
    # Ensure the final valid position detects accidentally excluding the current token.
    for row, length in enumerate(contexts):
        physical = int(tables[row, (length - 1) // page])
        slot = (length - 1) % page
        d0, d1 = CHANNEL_PAIRS[row]
        keys[physical, slot, d0 : d1 + 1] = torch.tensor([2.0, 1.0])
        scales[physical, slot] = 1

    # Hand-derived scalar answer, independent of all matmul/reference helpers.
    expected = torch.zeros((batch, width), dtype=torch.float32)
    for row, length in enumerate(contexts):
        d0, d1 = CHANNEL_PAIRS[row]
        for pos in range(length):
            physical, slot = int(tables[row, pos // page]), pos % page
            a, b = keys[physical, slot, d0 : d1 + 1].tolist()
            if variant == "negative_q":
                b = -b
            score = 0 if variant == "zero_q" else 2 * max(a, 0) - max(b, 0)
            expected[row, pos] = score * float(scales[physical, slot])

    k_fp8 = keys.to(torch.float8_e4m3fn)
    if not torch.equal(k_fp8.float(), keys):
        raise AssertionError("Hand-built K changes during FP8 conversion")
    native_keys = keys * scales[..., None]
    if not torch.equal(native_keys.to(torch.bfloat16).float(), native_keys):
        raise AssertionError("Scaled K changes during native BF16 preparation")
    if not torch.equal(weights.to(torch.bfloat16).float(), weights):
        raise AssertionError("Weights change during native BF16 preparation")
    cache = pack_cache(k_fp8, scales)
    if variant == "permuted_pages":
        # A cycle of length five is not its own inverse, unlike reversal.
        permutation = torch.roll(torch.arange(pages), shifts=1)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(pages)
        assert not torch.equal(
            permutation, inverse
        ), "Permutation must not be self-inverse"
        permuted_cache = cache[permutation].contiguous()
        remapped_tables = inverse[tables.long()].to(torch.int32)
        if not torch.equal(
            cache[tables.long()], permuted_cache[remapped_tables.long()]
        ):
            raise AssertionError("Page permutation changed logical cache bytes")
        cache, tables = permuted_cache, remapped_tables

    data = Inputs(
        quant=quant,
        q=q_values.to(device),
        q_scale=None if q_scale is None else q_scale.to(device),
        cache=cache.to(device),
        weights=weights.to(device),
        lengths=torch.tensor(contexts, device=device, dtype=torch.int32),
        tables=tables.to(device),
        contexts=contexts,
        width=width,
    )
    return data, expected.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--page-sizes",
        type=int,
        nargs="+",
        choices=(16, 32, 64, 128, 256),
        default=[64, 256],
    )
    parser.add_argument("--quant", choices=("fp8", "fp4", "both"), default="both")
    parser.add_argument("--op-file", type=Path)
    parser.add_argument("--libdevice-path", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("paged_mqa_equivalence.json")
    )
    args = parser.parse_args()
    os.environ["GEMS_VENDOR"] = "iluvatar"
    os.environ["FLAGGEMS_FP8_FP4_PAGED_MQA_LOGITS_TLE"] = "0"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    device = torch.device("cpu" if args.cpu_only else args.device)
    if not args.cpu_only:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "Use the working CoreX container, or --cpu-only for fixtures only"
            )
        torch.cuda.set_device(device)
        preflight = compiler_preflight("iluvatar", args.libdevice_path)
        print(
            "Compiler preflight: " + json.dumps(preflight, ensure_ascii=False),
            flush=True,
        )
    torch.set_float32_matmul_precision("highest")
    report = {
        "scope": "valid logits, next_n=1, H=64, D=128, exactly representable inputs",
        "active_head_pairs_by_request": HEAD_PAIRS,
        "active_channel_pairs_by_request": CHANNEL_PAIRS,
        "tolerances": {"rtol": 0, "atol": 0},
        "cpu_only": args.cpu_only,
        "performance_comparison": False,
        "results": [],
    }
    failed = False
    quants = ("fp8", "fp4") if args.quant == "both" else (args.quant,)
    with torch.inference_mode():
        for page in args.page_sizes:
            for quant in quants:
                variants = BASE_VARIANTS
                if quant == "fp4":
                    variants += ("nonunit_q_scale",)
                for variant in variants:
                    name = f"{quant} page={page} {variant}"
                    row = {
                        "quant": quant,
                        "page": page,
                        "variant": variant,
                        "status": "ERROR",
                    }
                    try:
                        synchronize(device)
                        data, expected = fixture(quant, page, variant, device)
                        q = decode_q(data.q, data.q_scale)
                        k, scales = unpack_cache(data.cache)
                        ref = reference(q, k, scales, data.weights, data, 64)
                        row["torch_vs_hand_answer"] = compare(ref, expected, data, 0, 0)
                        if not row["torch_vs_hand_answer"]["passed"]:
                            raise AssertionError(
                                "Input packing/reference disagrees with the hand answer"
                            )
                        if not args.cpu_only:
                            compiled = make_torch_call(data, 64, compiled=True)
                            compiled_out = compiled()
                            synchronize(device)
                            gems, gems_info = make_gems_call(data, args.op_file)
                            gems_out = gems()
                            synchronize(device)
                            row["compiled_implementation"] = (
                                "torch_compile_chunked_fp32"
                            )
                            row["gems_implementation"] = gems_info
                            row["torch_compile_vs_hand_answer"] = compare(
                                compiled_out, expected, data, 0, 0
                            )
                            row["gems_vs_hand_answer"] = compare(
                                gems_out, expected, data, 0, 0
                            )
                            row["torch_compile_vs_gems"] = compare(
                                compiled_out, gems_out, data, 0, 0
                            )
                            checks = (
                                "torch_compile_vs_hand_answer",
                                "gems_vs_hand_answer",
                                "torch_compile_vs_gems",
                            )
                            if not all(row[key]["passed"] for key in checks):
                                raise AssertionError(
                                    "torch.compile/Gems/hand answer disagree; "
                                    "inspect JSON error metrics"
                                )
                            del compiled, gems, compiled_out, gems_out
                        row["status"] = "PASS"
                        scope = (
                            "CPU fixture only"
                            if args.cpu_only
                            else "torch.compile = gems = hand answer"
                        )
                        print(f"PASS {name} ({scope})", flush=True)
                    except Exception as exc:
                        failed = True
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        row["traceback"] = traceback.format_exc()
                        print(f"ERROR {name}: {row['error']}", flush=True)
                    report["results"].append(row)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
                    )
    print(f"Saved {args.output.resolve()}")
    print(
        "This checks the listed cases, not all inputs, padding semantics, "
        "or same-format performance."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
