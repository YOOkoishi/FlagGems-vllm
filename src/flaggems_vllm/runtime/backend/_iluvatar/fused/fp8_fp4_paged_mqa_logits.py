# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Experimental Iluvatar MQA: direct paged loads, optional TLE shared staging.

The byte cache contains all K bytes followed by all FP32 scales *within each
page*. Its logical [pages, page_size, 1, D+4] shape is not an interleaved layout.
No cache unpacking, context expansion, or device-to-host max reduction is needed.
"""

import torch
import triton
import triton.language as tl

from .fp8_fp4_mqa_logits import _load_query, _require_tle, _stage_k, _tle_enabled


# Iluvatar-only candidates, following PR #855's inline autotune convention.
# Every config fully overwrites its output, including invalid context positions.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": n}, num_warps=4, num_stages=1) for n in (32, 64, 128)
    ],
    key=["ROWS", "N", "H", "D", "PAGE_SIZE", "IS_FP4", "USE_TLE"],
)
@triton.jit
def _paged_mqa_kernel(
    Q,
    QS,
    KV,
    W,
    CTX,
    BT,
    OUT,
    ROWS,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    NEXT_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGES,
    BT_WIDTH: tl.constexpr,
    stride_qb,
    stride_qi,
    stride_qh,
    stride_qd,
    stride_qsb,
    stride_qsi,
    stride_qsh,
    stride_kvb,
    stride_wm,
    stride_wh,
    stride_ctxb,
    stride_ctxi,
    stride_btb,
    stride_bti,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP4: tl.constexpr,
    CLEAN: tl.constexpr,
    USE_TLE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    tiles = tl.cdiv(N, BLOCK_N)
    row = pid // tiles
    start = (pid % tiles) * BLOCK_N
    b, query = row // NEXT_N, row % NEXT_N
    n = start + tl.arange(0, BLOCK_N)
    ctx = tl.load(CTX + b * stride_ctxb + query * stride_ctxi)
    fill: tl.constexpr = float("-inf") if CLEAN else 0.0
    if start >= ctx:
        tl.store(OUT + row * N + n, fill, mask=n < N)
        return

    logical_page = n // PAGE_SIZE
    in_context = (n < N) & (n < ctx) & (logical_page < BT_WIDTH)
    physical_page = tl.load(
        BT + b * stride_btb + logical_page * stride_bti,
        mask=in_context,
        other=-1,
    )
    valid = in_context & (physical_page >= 0) & (physical_page < PAGES)
    # Use a safe address even for lanes masked out of the load.
    physical_page = tl.where(valid, physical_page, 0).to(tl.int64)
    page_offset = n % PAGE_SIZE
    d = tl.arange(0, D)
    k_bytes = tl.load(
        KV
        + physical_page[:, None] * stride_kvb
        + page_offset[:, None] * D
        + d[None, :],
        mask=valid[:, None],
        other=0,
    )
    k = k_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    if USE_TLE:
        k = _stage_k(k)

    scale_ptr = (KV + physical_page * stride_kvb + PAGE_SIZE * D + page_offset * 4).to(
        tl.pointer_type(tl.float32)
    )
    scales = tl.load(scale_ptr, mask=valid, other=0.0)
    q = _load_query(
        Q,
        QS,
        b * stride_qb + query * stride_qi,
        b * stride_qsb + query * stride_qsi,
        stride_qh,
        stride_qd,
        stride_qsh,
        H,
        D,
        BLOCK_H,
        IS_FP4,
    )
    heads = tl.arange(0, BLOCK_H)
    weights = tl.load(W + row * stride_wm + heads * stride_wh, heads < H, 0.0)
    dots = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    scores = tl.maximum(dots * scales[None, :], 0.0)
    result = tl.sum(scores * weights[:, None], axis=0)
    tl.store(OUT + row * N + n, tl.where(valid, result, fill), mask=n < N)


def _launch_kernel(args, meta, use_tle):
    grid = lambda cfg: (meta["ROWS"] * triton.cdiv(meta["N"], cfg["BLOCK_N"]),)
    _paged_mqa_kernel[grid](*args, **meta, USE_TLE=use_tle)


def _launch_tle_kernel(args, meta):
    """Observable entry used only when the TLE specialization is requested."""
    _require_tle()
    _launch_kernel(args, meta, True)


def fp8_fp4_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
):
    """Direct paged FP8/FP4 MQA candidate for Iluvatar; forward only.

    Supports D=128, 1..64 heads, packed contiguous pages, strided Q/weights,
    and context lengths [B] (shared by each query) or [B,next_n]. Invalid output
    positions are -inf when clean_logits is true and zero otherwise.
    """
    if not isinstance(q, (tuple, list)) or len(q) != 2:
        raise ValueError("q must be (q_values, q_scale)")
    q_values, q_scale = q
    is_fp4 = q_scale is not None
    tensors = [q_values, kv_cache, weights, context_lens, block_tables]
    if is_fp4:
        tensors.append(q_scale)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("MQA inputs must be tensors")
    if any(tensor.layout != torch.strided for tensor in tensors):
        raise NotImplementedError("Only strided inputs are supported")
    if not isinstance(clean_logits, bool):
        raise TypeError("clean_logits must be bool")
    if q_values.ndim == 3:
        q_values = q_values.unsqueeze(1)  # Metadata-only view.
        if is_fp4 and q_scale.ndim == 2:
            q_scale = q_scale.unsqueeze(1)
    if q_values.ndim != 4:
        raise NotImplementedError("Iluvatar MQA expects Q [B,next_n,H,D]")
    batch, next_n, heads, qdim = q_values.shape
    if kv_cache.ndim != 4 or kv_cache.shape[2] != 1:
        raise NotImplementedError("Expected packed KV [pages,page_size,1,D+4]")
    dim = kv_cache.shape[3] - 4
    page_size = kv_cache.shape[1]
    if dim != 128 or not 1 <= heads <= 64:
        raise NotImplementedError("Iluvatar MQA candidate requires D=128, 1<=H<=64")
    if page_size not in (16, 32, 64, 128, 256):
        raise NotImplementedError("Supported page sizes are 16,32,64,128,256")
    if qdim != (dim // 2 if is_fp4 else dim):
        raise ValueError("Q dimension does not match the KV cache")
    if q_values.dtype != (torch.uint8 if is_fp4 else torch.float8_e4m3fn):
        raise NotImplementedError("Expected packed uint8 FP4 or float8_e4m3fn Q")
    if kv_cache.dtype != torch.uint8 or not kv_cache.is_contiguous():
        raise NotImplementedError("KV cache must be contiguous packed uint8 pages")
    if kv_cache.storage_offset() % 4:
        raise NotImplementedError("Packed KV scales require a 4-byte aligned cache")
    if not isinstance(max_model_len, int) or not 0 <= max_model_len < 2**31:
        raise ValueError("max_model_len must be a nonnegative int32 length")
    rows = batch * next_n
    if weights.shape != (rows, heads) or weights.dtype != torch.float32:
        raise ValueError("weights must be FP32 [B*next_n,H]")
    if block_tables.ndim != 2 or block_tables.shape[0] != batch:
        raise ValueError("block_tables must have shape [B,max_blocks]")
    if block_tables.dtype != torch.int32 or context_lens.dtype != torch.int32:
        raise NotImplementedError("Block tables and context lengths must be int32")
    if context_lens.shape == (batch,):
        ctx_strides = (context_lens.stride(0), 0)
    elif context_lens.shape == (batch, next_n):
        ctx_strides = context_lens.stride()
    else:
        raise ValueError("context_lens must have shape [B] or [B,next_n]")
    tensors = [q_values, kv_cache, weights, context_lens, block_tables]
    if is_fp4:
        if q_scale.shape != (batch, next_n, heads) or q_scale.dtype != torch.int32:
            raise ValueError("FP4 scales must be packed int32 [B,next_n,H]")
        tensors.append(q_scale)
    if q_values.device.type != "cuda" or any(
        tensor.device != q_values.device for tensor in tensors
    ):
        raise NotImplementedError("All inputs must be on the same Iluvatar device")
    logits = torch.empty(
        (rows, max_model_len), dtype=torch.float32, device=q_values.device
    )
    if rows == 0 or max_model_len == 0:
        return logits
    qs_strides = q_scale.stride() if is_fp4 else (0, 0, 0)
    args = (
        q_values,
        q_scale if is_fp4 else q_values,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        logits,
    )
    meta = dict(
        ROWS=rows,
        N=max_model_len,
        H=heads,
        D=dim,
        NEXT_N=next_n,
        PAGE_SIZE=page_size,
        PAGES=kv_cache.shape[0],
        BT_WIDTH=block_tables.shape[1],
        stride_qb=q_values.stride(0),
        stride_qi=q_values.stride(1),
        stride_qh=q_values.stride(2),
        stride_qd=q_values.stride(3),
        stride_qsb=qs_strides[0],
        stride_qsi=qs_strides[1],
        stride_qsh=qs_strides[2],
        stride_kvb=kv_cache.stride(0),
        stride_wm=weights.stride(0),
        stride_wh=weights.stride(1),
        stride_ctxb=ctx_strides[0],
        stride_ctxi=ctx_strides[1],
        stride_btb=block_tables.stride(0),
        stride_bti=block_tables.stride(1),
        BLOCK_H=max(16, triton.next_power_of_2(heads)),
        IS_FP4=is_fp4,
        CLEAN=clean_logits,
    )
    if _tle_enabled("FLAGGEMS_FP8_FP4_PAGED_MQA_LOGITS_TLE"):
        _launch_tle_kernel(args, meta)
    else:
        _launch_kernel(args, meta, False)
    return logits
