# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU input construction for context-sharded paged MQA experiments.

Logical K values depend only on global batch/token/channel coordinates and seed.
Changing world size changes storage and page tables, not the valid input values.
Q/weights are regenerated identically on each rank; callers may broadcast their
prepared tensors to verify or enforce that agreement before measurements.
"""

import math

import torch
from benchmark_paged_mqa import Inputs, decode_q, pack_cache

HEAD_DIM = 128
MAX_CHUNK_TOKENS = 65536
_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1


def shard_plan(context, page_size, world_size, rank):
    """Return a page-aligned contiguous shard, including an empty-rank plan."""
    if context < 0 or page_size <= 0 or world_size <= 0:
        raise ValueError("context must be nonnegative; page/world sizes positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    pages = (context + page_size - 1) // page_size
    capacity = ((pages + world_size - 1) // world_size) * page_size
    start = rank * capacity
    return {
        "capacity": capacity,
        "start": start,
        "length": max(0, min(context - start, capacity)),
    }


def _sizes(args):
    batch, heads = args.batch, args.heads
    if batch <= 0 or heads <= 0:
        raise ValueError("batch and heads must be positive")
    if args.context < 0 or args.context > torch.iinfo(torch.int64).max:
        raise ValueError("context must fit a nonnegative int64")
    if not 0 < args.page_size <= MAX_CHUNK_TOKENS:
        raise ValueError("page_size must be in [1, 65536]")
    return batch, heads, HEAD_DIM


def make_queries(args, quant):
    """Return CPU (packed Q, optional packed scales, FP32 head weights)."""
    batch, heads, dim = _sizes(args)
    rng = torch.Generator(device="cpu").manual_seed(args.seed & _MASK64)
    if quant == "fp8":
        q = torch.randn((batch, 1, heads, dim), generator=rng, device="cpu")
        q = q.to(torch.float8_e4m3fn)
        q_scale = None
    elif quant == "fp4":
        codes = torch.randint(
            0, 16, (batch, 1, heads, dim), generator=rng, device="cpu"
        )
        q = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
        scale_codes = torch.randint(
            124, 129, (batch, 1, heads, 4), generator=rng, device="cpu"
        )
        shifts = 8 * torch.arange(4, dtype=torch.int64, device="cpu")
        q_scale = (scale_codes << shifts).sum(-1).to(torch.int32)
    else:
        raise ValueError(f"Unsupported query format: {quant}")
    weights = torch.randn((batch, heads), generator=rng, device="cpu") / math.sqrt(
        heads
    )
    return q, q_scale, weights


def _mix32(value):
    # Mask before every multiplication. The multiplier is below 2**27, so
    # products of unsigned 32-bit values fit signed int64 without overflow.
    value = value & _MASK32
    value = ((value ^ (value >> 16)) * 0x45D9F3B) & _MASK32
    value = ((value ^ (value >> 16)) * 0x45D9F3B) & _MASK32
    return value ^ (value >> 16)


def _fold_coordinate(value, salt):
    low = value & _MASK32
    high = (value >> 32) & _MASK32
    return _mix32(low ^ salt) ^ _mix32(high ^ (salt ^ 0x9E3779B9))


def _kv_samples(batch_ids, positions, seed):
    """Generate one full D-vector for each global (batch, token) coordinate."""
    if batch_ids.ndim != 1 or positions.ndim != 1:
        raise ValueError("Global coordinates must be one-dimensional")
    if batch_ids.shape != positions.shape:
        raise ValueError("Global batch/token coordinate counts must agree")
    batch_ids = batch_ids.to(device="cpu", dtype=torch.int64)
    positions = positions.to(device="cpu", dtype=torch.int64)
    dimensions = torch.arange(HEAD_DIM, dtype=torch.int64, device="cpu")
    seed_bits = seed & _MASK64
    signed_seed = seed_bits if seed_bits < (1 << 63) else seed_bits - (1 << 64)
    seed_hash = _fold_coordinate(torch.tensor(signed_seed, device="cpu"), 0x1B56C4E9)
    batch_hash = _fold_coordinate(batch_ids, 0xA511E9B3)
    token_hash = _fold_coordinate(positions, 0x63D83595)
    dimension_hash = _fold_coordinate(dimensions, 0xB5297A4D)
    hashed = _mix32(
        batch_hash[:, None] ^ token_hash[:, None] ^ dimension_hash[None, :] ^ seed_hash
    )
    # A dyadic FP32 grid gives deterministic samples in the inclusive [-1, 1].
    return (hashed.remainder(65537).float() - 32768.0) / 32768.0


def _quantized_kv(batch_ids, positions, seed):
    values = _kv_samples(batch_ids, positions, seed)
    scales = values.abs().amax(-1).clamp_min(1e-4) / 448.0
    values.div_(scales[:, None])
    return values.to(torch.float8_e4m3fn), scales


def make_shard_inputs(args, quant, rank, world, device):
    """Return local Inputs and a plan; local logits have [B, capacity] shape.

    Empty shards still carry Q/weights, zero lengths, and one finite dummy page.
    Callers should skip their local kernel while participating in collectives.
    """
    batch, _, dim = _sizes(args)
    plan = shard_plan(args.context, args.page_size, world, rank)
    capacity, start, length = plan["capacity"], plan["start"], plan["length"]
    page = args.page_size
    if length > torch.iinfo(torch.int32).max:
        raise ValueError("Local context lengths must fit int32")
    q, q_scale, weights = make_queries(args, quant)
    pages_per_request = (length + page - 1) // page
    physical_pages = max(1, batch * pages_per_request)
    if physical_pages > torch.iinfo(torch.int32).max:
        raise ValueError("Local physical page IDs must fit int32")
    tables = torch.zeros((batch, capacity // page), dtype=torch.int32, device="cpu")
    if length == 0:
        cache = pack_cache(
            torch.zeros((1, page, dim), device="cpu").to(torch.float8_e4m3fn),
            torch.ones((1, page), dtype=torch.float32, device="cpu"),
        )
    else:
        # Storage may depend on rank; values always use global coordinates.
        mapping_seed = (args.seed ^ 0xD1B54A32D192ED03 ^ rank) & _MASK64
        mapping_rng = torch.Generator(device="cpu").manual_seed(mapping_seed)
        physical = torch.randperm(physical_pages, generator=mapping_rng, device="cpu")
        tables[:, :pages_per_request] = physical.reshape(batch, pages_per_request)
        cache = torch.empty(
            (physical_pages, page, 1, dim + 4), dtype=torch.uint8, device="cpu"
        )
        pages_per_chunk = max(1, MAX_CHUNK_TOKENS // page)
        padded_tokens_per_request = pages_per_request * page
        for first in range(0, physical_pages, pages_per_chunk):
            end = min(first + pages_per_chunk, physical_pages)
            flat = torch.arange(
                first * page, end * page, dtype=torch.int64, device="cpu"
            )
            batch_ids = flat // padded_tokens_per_request
            positions = start + flat % padded_tokens_per_request
            k, scales = _quantized_kv(batch_ids, positions, args.seed)
            packed = pack_cache(
                k.reshape(end - first, page, dim),
                scales.reshape(end - first, page),
            )
            cache[physical[first:end]] = packed
        del k, scales, packed, flat, batch_ids, positions, physical
    data = Inputs(
        quant=quant,
        q=q.to(device),
        q_scale=None if q_scale is None else q_scale.to(device),
        cache=cache.to(device),
        weights=weights.to(device),
        lengths=torch.full((batch,), length, dtype=torch.int32, device=device),
        tables=tables.to(device),
        contexts=[length] * batch,
        width=capacity,
    )
    return data, plan


def sample_reference(args, quant, positions):
    """Return CPU FP32 [B, len(positions)] scores from global logical tokens.

    This does not inspect a shard's cache or page table. It regenerates and
    quantizes complete token vectors before calculating the FP32 reference.
    """
    batch, _, _ = _sizes(args)
    positions = torch.as_tensor(positions, device="cpu")
    if positions.ndim != 1:
        raise ValueError("Reference positions must be one-dimensional")
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if positions.numel() and positions.dtype not in integer_dtypes:
        raise ValueError("Reference positions must be integer indices")
    positions = positions.to(torch.int64)
    if positions.numel() and (
        (positions < 0).any() or (positions >= args.context).any()
    ):
        raise ValueError("Reference positions must be within the global context")
    q, q_scale, weights = make_queries(args, quant)
    q = decode_q(q, q_scale)
    result = torch.empty((batch, positions.numel()), dtype=torch.float32, device="cpu")
    for row in range(batch):
        for first in range(0, positions.numel(), MAX_CHUNK_TOKENS):
            pos = positions[first : first + MAX_CHUNK_TOKENS]
            k, scales = _quantized_kv(torch.full_like(pos, row), pos, args.seed)
            dots = q[row] @ k.float().T
            scores = torch.relu(dots * scales[None, :])
            result[row, first : first + pos.numel()] = (
                scores * weights[row, :, None]
            ).sum(0)
    return result
