# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental Iluvatar MQA logits with a shared-memory TLE ablation."""

import os

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None

try:
    from triton.experimental.tle import is_primitive_supported
except ImportError:
    is_primitive_supported = None


def _tle_enabled(env_name):
    return os.environ.get(env_name, "1").lower() not in ("0", "false", "off", "no")


def _require_tle():
    """Check the primitives actually used by the Iluvatar candidate."""
    gpu = getattr(tle, "gpu", None)
    for name in ("alloc", "local_ptr"):
        if gpu is None or not hasattr(gpu, name):
            raise NotImplementedError(f"Iluvatar MQA requires TLE gpu.{name}")
        if is_primitive_supported is not None and not is_primitive_supported(
            "iluvatar", f"gpu.{name}"
        ):
            raise NotImplementedError(f"Iluvatar does not support TLE gpu.{name}")


@triton.jit
def _load_query(
    Q_ptr,
    Q_scale_ptr,
    q_base,
    qs_base,
    stride_qh,
    stride_qd,
    stride_qsh,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IS_FP4: tl.constexpr,
):
    """Load one query, preserving its heads as the dot product's M axis."""
    h = tl.arange(0, BLOCK_H).to(tl.int64)
    d = tl.arange(0, D)
    if IS_FP4:
        d2 = tl.arange(0, D // 2).to(tl.int64)
        packed = tl.load(
            Q_ptr + q_base + h[:, None] * stride_qh + d2[None, :] * stride_qd,
            mask=h[:, None] < H,
            other=0,
        )
        nibble = tl.reshape(tl.join(packed & 15, (packed >> 4) & 15), [BLOCK_H, D])
        n = nibble.to(tl.int32)
        sign = 1.0 - 2.0 * ((n >> 3) & 1).to(tl.float32)
        exponent = (n >> 1) & 3
        mantissa = (n & 1).to(tl.float32)
        magnitude = tl.where(
            exponent == 0,
            0.5 * mantissa,
            tl.where(
                exponent == 1,
                1.0 + 0.5 * mantissa,
                tl.where(exponent == 2, 2.0 + mantissa, 4.0 + 2.0 * mantissa),
            ),
        )
        packed_scale = tl.load(
            Q_scale_ptr + qs_base + h * stride_qsh, mask=h < H, other=0
        )
        scale_byte = (packed_scale[:, None] >> (8 * (d[None, :] // 32))) & 255
        scale = tl.exp2(scale_byte.to(tl.float32) - 127.0)
        query = (sign * magnitude * scale).to(tl.float16)
    else:
        query = tl.load(
            Q_ptr
            + q_base
            + h[:, None] * stride_qh
            + d[None, :].to(tl.int64) * stride_qd,
            mask=h[:, None] < H,
            other=0.0,
        ).to(tl.float16)
    return query


@triton.jit
def _stage_k(k):
    """Stage an FP16 tile synchronously; this does not promise async overlap."""
    shared = tle.gpu.alloc(k.shape, dtype=tl.float16, nv_mma_shared_layout=False)
    pointers = tle.gpu.local_ptr(shared)
    tl.store(pointers, k)
    tl.debug_barrier()
    return tl.load(pointers)


# This backend-local experiment follows the small inline candidate-set pattern
# used by vendor overrides. The NVIDIA configuration is deliberately separate.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": block_n}, num_warps=4, num_stages=1)
        for block_n in (32, 64, 128)
    ],
    key=["M", "N", "H", "D", "IS_FP4", "CLEAN_LOGITS", "USE_TLE"],
)
@triton.jit
def _dense_mqa_kernel(
    Q,
    QScale,
    K,
    KScale,
    W,
    Starts,
    Ends,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_qsm: tl.constexpr,
    stride_qsh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_start: tl.constexpr,
    stride_end: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP4: tl.constexpr,
    CLEAN_LOGITS: tl.constexpr,
    USE_TLE: tl.constexpr,
):
    tiles: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    pid = tl.program_id(0).to(tl.int64)
    row = pid // tiles
    n = (pid % tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, D)
    h = tl.arange(0, BLOCK_H)
    query = _load_query(
        Q,
        QScale,
        row * stride_qm,
        row * stride_qsm,
        stride_qh,
        stride_qd,
        stride_qsh,
        H,
        D,
        BLOCK_H,
        IS_FP4,
    )
    keys = tl.load(
        K + n[:, None] * stride_kn + d[None, :] * stride_kd,
        mask=n[:, None] < N,
        other=0.0,
    ).to(tl.float16)
    if USE_TLE:
        keys = _stage_k(keys)
    dots = tl.dot(query, tl.trans(keys), out_dtype=tl.float32)
    scales = tl.load(KScale + n * stride_ks, mask=n < N, other=0.0)
    weights = tl.load(W + row * stride_wm + h * stride_wh, mask=h < H, other=0.0)
    scores = tl.maximum(dots * scales[None, :], 0.0)
    logits = tl.sum(scores * weights[:, None], axis=0)
    if CLEAN_LOGITS:
        start = tl.load(Starts + row * stride_start)
        end = tl.load(Ends + row * stride_end)
        logits = tl.where((n >= start) & (n < end), logits, float("-inf"))
    tl.store(Out + row * N + n, logits, mask=n < N)


def _launch_tle_kernel(args, meta, M, N):
    _require_tle()
    grid = lambda config: (M * triton.cdiv(N, config["BLOCK_N"]),)
    _dense_mqa_kernel[grid](*args, **meta, USE_TLE=True)


def _launch_triton_kernel(args, meta, M, N):
    grid = lambda config: (M * triton.cdiv(N, config["BLOCK_N"]),)
    _dense_mqa_kernel[grid](*args, **meta, USE_TLE=False)


def fp8_fp4_mqa_logits(
    q: tuple,
    kv: tuple,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Experimental D=128, H<=64 Iluvatar dense logits, FP8 or packed MXFP4.

    Nonnegative input strides are supported without materializing copies.
    The TLE switch selects shared staging; disabling it uses the same schedule
    and FP16 dot operands. Each switch setting autotunes its tile independently.
    """
    if not isinstance(q, (tuple, list)) or len(q) != 2:
        raise ValueError("q must be (q_values, q_scale)")
    if not isinstance(kv, (tuple, list)) or len(kv) != 2:
        raise ValueError("kv must be (k_values, k_scales)")
    q_values, q_scale = q
    k_values, k_scales = kv
    tensors = [q_values, k_values, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke]
    if q_scale is not None:
        tensors.append(q_scale)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("MQA inputs must be tensors")
    if q_values.ndim != 3 or k_values.ndim != 2:
        raise ValueError("q_values must be [M,H,D or D/2] and k_values [N,D]")
    M, H, q_dim = q_values.shape
    N, D = k_values.shape
    is_fp4 = q_scale is not None
    if D != 128 or not 1 <= H <= 64:
        raise NotImplementedError("Iluvatar MQA candidate supports D=128 and 1<=H<=64")
    if q_dim != (D // 2 if is_fp4 else D):
        raise ValueError("query width does not match the key head dimension")
    if weights.shape != (M, H) or k_scales.shape != (N,):
        raise ValueError("weights must be [M,H] and k_scales [N]")
    if cu_seqlen_ks.shape != (M,) or cu_seqlen_ke.shape != (M,):
        raise ValueError("sequence starts and ends must be [M]")
    if is_fp4 and (q_scale.shape != (M, H) or q_scale.dtype != torch.int32):
        raise ValueError("MXFP4 q_scale must be [M,H] int32")
    expected_q_dtype = torch.uint8 if is_fp4 else torch.float8_e4m3fn
    if q_values.dtype != expected_q_dtype or k_values.dtype != torch.float8_e4m3fn:
        raise NotImplementedError("Expected FP8 E4M3FN keys and FP8 or packed uint8 Q")
    if weights.dtype != torch.float32 or k_scales.dtype != torch.float32:
        raise NotImplementedError("MQA weights and key scales must be float32")
    if any(
        t.dtype not in (torch.int32, torch.int64) for t in (cu_seqlen_ks, cu_seqlen_ke)
    ):
        raise NotImplementedError("Sequence starts and ends must be int32 or int64")
    if q_values.device.type != "cuda" or any(
        tensor.device != q_values.device for tensor in tensors
    ):
        raise ValueError("All MQA inputs must be on the same CUDA/CoreX device")
    if any(tensor.layout != torch.strided for tensor in tensors) or any(
        stride < 0 for tensor in tensors for stride in tensor.stride()
    ):
        raise NotImplementedError(
            "Only strided inputs with nonnegative strides are supported"
        )
    if not isinstance(clean_logits, bool):
        raise TypeError("clean_logits must be bool")
    logits = torch.empty((M, N), dtype=torch.float32, device=q_values.device)
    if M == 0 or N == 0:
        return logits

    args = (
        q_values,
        q_scale if is_fp4 else q_values,
        k_values,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
    )
    meta = {
        "M": M,
        "N": N,
        "H": H,
        "D": D,
        "stride_qm": q_values.stride(0),
        "stride_qh": q_values.stride(1),
        "stride_qd": q_values.stride(2),
        "stride_qsm": q_scale.stride(0) if is_fp4 else 0,
        "stride_qsh": q_scale.stride(1) if is_fp4 else 0,
        "stride_kn": k_values.stride(0),
        "stride_kd": k_values.stride(1),
        "stride_ks": k_scales.stride(0),
        "stride_wm": weights.stride(0),
        "stride_wh": weights.stride(1),
        "stride_start": cu_seqlen_ks.stride(0),
        "stride_end": cu_seqlen_ke.stride(0),
        "BLOCK_H": max(16, triton.next_power_of_2(H)),
        "IS_FP4": is_fp4,
        "CLEAN_LOGITS": clean_logits,
    }
    launcher = (
        _launch_tle_kernel
        if _tle_enabled("FLAGGEMS_FP8_FP4_MQA_LOGITS_TLE")
        else _launch_triton_kernel
    )
    launcher(args, meta, M, N)
    return logits
