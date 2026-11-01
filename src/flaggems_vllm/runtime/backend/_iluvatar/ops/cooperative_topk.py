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

"""Iluvatar GPU cooperative top-k specialization.

The public ABI and wide-row radix protocol are shared with the generic
implementation.  Iluvatar TLE backend stages the ordered key tile in shared
memory before the radix scans, which bounds the live register tile on DCU
Iluvatar architectures.  If the installed Triton build has no TLE support, dispatch
uses the shared implementation unchanged.
"""

import triton
import triton.language as tl

from flaggems_vllm.ops.cooperative_topk import (
    _cooperative_topk_impl,
    _key_greater,
    _ordered_key,
    _radix_digit,
)
from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False

HAS_TLE = bool(
    HAS_TLE
    and hasattr(tle, "gpu")
    and hasattr(tle.gpu, "alloc")
    and hasattr(tle.gpu, "local_ptr")
    and hasattr(tle.gpu, "smem")
)


if HAS_TLE:

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("cooperative_topk_small"),
        key=["WIDTH", "K", "ROWS"],
        strategy=["default", "default", "default"],
    )
    @triton.jit
    def _cooperative_topk_small_tle(
        Logits,
        Lengths,
        Output,
        WIDTH: tl.constexpr,
        STRIDE: tl.constexpr,
        K: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
        offsets = tl.arange(0, BLOCK)
        valid = offsets < length

        # Keep the ordered key tile in DCU shared memory instead of carrying
        # the complete tile through every radix pass in registers.
        shared_keys = tle.gpu.alloc(
            [BLOCK],
            dtype=tl.uint32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        shared_keys_ptr = tle.gpu.local_ptr(shared_keys, (0,))

        if length <= K:
            short_indices = tl.arange(0, K)
            tl.store(
                Output + row * K + short_indices,
                tl.where(short_indices < length, short_indices, -1),
            )
        else:
            values = tl.load(Logits + row * STRIDE + offsets, valid, other=0.0)
            keys = _ordered_key(values)
            tl.store(shared_keys_ptr + offsets, keys, mask=valid)
            tl.debug_barrier()
            keys = tl.load(shared_keys_ptr + offsets, mask=valid, other=0)
            prefix = tl.full((), 0, tl.uint32)
            remaining = tl.full((), K, tl.int32)
            candidates = valid
            for shift in tl.static_range(24, -1, -8):
                digits = ((keys >> shift) & 255).to(tl.int32)
                histogram = tl.histogram(digits, 256, mask=candidates)
                digit, remaining = _radix_digit(histogram, remaining)
                prefix = prefix | (digit << shift)
                candidates = candidates & (digits == digit)
            greater = valid & _key_greater(keys, prefix)
            equal = valid & (keys == prefix)
            tie_rank = tl.cumsum(equal.to(tl.int32))
            selected = greater | (equal & (tie_rank <= remaining))
            positions = tl.cumsum(selected.to(tl.int32)) - 1
            tl.store(Output + row * K + positions, offsets, selected)

else:
    _cooperative_topk_small_tle = None


def cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    """Run cooperative top-k with the Iluvatar TLE narrow-row specialization."""
    return _cooperative_topk_impl(
        logits,
        lengths,
        output,
        workspace,
        k,
        max_seq_len,
        small_kernel=_cooperative_topk_small_tle,
    )
