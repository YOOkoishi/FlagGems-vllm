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

"""Ascend TLE narrow-row path for cooperative_topk.

Ascend exposes TLE scratch through the DSA buffer API rather than the CUDA-like
``tle.gpu`` shared-memory helpers.  The wide-row path remains the shared radix
implementation, whose int32 scratch layout avoids unsupported uint16 stores on
Ascend A2/A3.
"""

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.cooperative_topk import (
    _cooperative_topk_impl,
    _key_greater,
    _ordered_key,
    _radix_digit,
)
from flaggems_vllm.utils import libentry, libtuner

try:
    import triton.experimental.tle as tle

    _HAS_TLE_DSA = all(
        hasattr(tle, name) for name in ("dsa",)
    ) and all(
        hasattr(tle.dsa, name) for name in ("to_buffer", "to_tensor")
    ) and hasattr(tle.dsa, "ascend")
except ImportError:
    tle = None
    _HAS_TLE_DSA = False


if _HAS_TLE_DSA:

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

        if length <= K:
            short_indices = tl.arange(0, K)
            tl.store(
                Output + row * K + short_indices,
                tl.where(short_indices < length, short_indices, -1),
            )
        else:
            values = tl.load(Logits + row * STRIDE + offsets, valid, other=0.0)

            # Materialize the tile in Ascend UB through the DSA TLE buffer
            # bridge. ``to_tensor`` returns a Triton value backed by the same
            # buffer, so the radix passes do not reload the row from GM.
            values_ub = tle.dsa.to_buffer(values, space=tle.dsa.ascend.UB)
            values = tle.dsa.to_tensor(values_ub)

            keys = _ordered_key(values)
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
    """Use Ascend DSA TLE staging for narrow rows when the API is available."""
    return _cooperative_topk_impl(
        logits,
        lengths,
        output,
        workspace,
        k,
        max_seq_len,
        small_kernel=_cooperative_topk_small_tle,
    )
