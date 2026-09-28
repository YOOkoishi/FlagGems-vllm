#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dense inputs and references for benchmark_mqa_suite.py (benchmark use only)."""

import math
from dataclasses import dataclass

import torch
from benchmark_paged_mqa import decode_q
from paged_mqa_distributed_inputs import _quantized_kv, make_queries, shard_plan


@dataclass
class DenseInputs:
    quant: str
    q: torch.Tensor
    q_scale: object
    k: torch.Tensor
    scales: torch.Tensor
    weights: torch.Tensor
    starts: torch.Tensor
    lengths: torch.Tensor
    contexts: list
    width: int


def make_dense_inputs(args, quant, rank, world, device):
    """All M queries share one K sequence; shard its N axis, never the M axis.

    This is the native dense API contract. Paged cases instead have a separate
    logical K sequence for each request, so timings of the two ops are not a
    dense-versus-paged comparison of identical work.
    """
    plan = shard_plan(args.context, args.page_size, world, rank)
    capacity, start, length = (plan[k] for k in ("capacity", "start", "length"))
    q, q_scale, weights = make_queries(args, quant)
    k = torch.zeros((capacity, 128), dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.ones(capacity, dtype=torch.float32)
    for first in range(0, length, 65536):
        last = min(first + 65536, length)
        positions = torch.arange(start + first, start + last, dtype=torch.int64)
        values, sf = _quantized_kv(torch.zeros_like(positions), positions, args.seed)
        k[first:last] = values
        scales[first:last] = sf
    return (
        DenseInputs(
            quant=quant,
            q=q.to(device),
            q_scale=None if q_scale is None else q_scale.to(device),
            k=k.to(device),
            scales=scales.to(device),
            weights=weights.to(device),
            starts=torch.zeros(args.batch, dtype=torch.int32, device=device),
            lengths=torch.full((args.batch,), length, dtype=torch.int32, device=device),
            contexts=[length] * args.batch,
            width=capacity,
        ),
        plan,
    )


def dense_chunk(q, k, scales, weights):
    dots = torch.matmul(q, k.T)
    return ((dots * scales[None, None, :]).clamp_min(0) * weights[:, :, None]).sum(1)


def make_dense_torch_call(data, chunk_tokens, compiled=True):
    chunk = dense_chunk
    if compiled:
        chunk = torch.compile(chunk, fullgraph=True, dynamic=False)
    length = max(data.contexts)

    def run():
        # FP4/FP8 decode, allocation, Python chunk loop and output copies count.
        q = decode_q(data.q, data.q_scale)
        k = data.k.float()
        result = torch.full((q.shape[0], data.width), float("-inf"), device=q.device)
        for start in range(0, length, chunk_tokens):
            end = min(start + chunk_tokens, length)
            result[:, start:end] = chunk(
                q, k[start:end], data.scales[start:end], data.weights
            )
        return result

    return run


def dense_reference(data, chunk_tokens):
    """Independent per-query FP32 mm, outside timing (not the batched baseline)."""
    q = decode_q(data.q, data.q_scale)
    result = torch.full((q.shape[0], data.width), float("-inf"), device=q.device)
    for row, length in enumerate(data.contexts):
        for start in range(0, length, chunk_tokens):
            end = min(start + chunk_tokens, length)
            dots = q[row] @ data.k[start:end].float().T
            scores = torch.relu(dots * data.scales[None, start:end])
            result[row, start:end] = (scores * data.weights[row, :, None]).sum(0)
    return result


def dense_sample_reference(args, quant, positions):
    """Regenerate global K coordinates on CPU, independently of shard assembly."""
    positions = torch.tensor(positions, dtype=torch.int64)
    q, scales, weights = make_queries(args, quant)
    q = decode_q(q, scales)
    k, sf = _quantized_kv(torch.zeros_like(positions), positions, args.seed)
    out = torch.empty((args.batch, len(positions)), dtype=torch.float32)
    for row in range(args.batch):
        scores = torch.relu((q[row] @ k.float().T) * sf[None, :])
        out[row] = (scores * weights[row, :, None]).sum(0)
    return out


def make_operator_call(module, operator, data):
    if operator == "paged":
        fn = module.fp8_fp4_paged_mqa_logits

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

    else:
        fn = module.fp8_fp4_mqa_logits
        q = data.q[:, 0]
        scales = None if data.q_scale is None else data.q_scale[:, 0]

        def run():
            return fn(
                q=(q, scales),
                kv=(data.k, data.scales),
                weights=data.weights,
                cu_seqlen_ks=data.starts,
                cu_seqlen_ke=data.lengths,
                clean_logits=True,
            )

    return run


def estimate_memory(args, operator, world, rank):
    """Conservative peak including reference, decode, collectives and scratch."""
    plan = shard_plan(args.context, args.page_size, world, rank)
    tokens = (
        max(1, args.batch * math.ceil(plan["length"] / args.page_size)) * args.page_size
        if operator == "paged"
        else plan["capacity"]
    )
    local = args.batch * plan["capacity"] * 4
    full = args.batch * args.context * 4
    chunk = (
        4 * args.batch * args.chunk_tokens * 128
        + 12 * args.batch * args.heads * args.chunk_tokens
    )
    working = int(1.15 * (tokens * 776 + world * local + full + chunk + local * 11))
    gpu = working + 2 * 1024**3
    host = tokens * 132 + min(tokens, 65536) * 128 * 40 + 1024**3
    if args.device == "cpu":
        # CPU mode is only a small communication/reference check.
        host = working + tokens * 132 + min(tokens, 65536) * 128 * 40 + 64 * 1024**2
    return {
        **plan,
        "local_physical_tokens": tokens,
        "packed_kv_bytes": tokens * 132,
        "local_logits_bytes": local,
        "global_logits_bytes_per_rank": full,
        "estimated_gpu_bytes": gpu,
        "estimated_host_bytes": host,
    }
