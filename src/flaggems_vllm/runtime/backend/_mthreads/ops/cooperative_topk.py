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

"""Moore Threads dispatch for the shared cooperative top-k kernels."""

from flaggems_vllm.ops.cooperative_topk import _cooperative_topk_impl


def cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    """Use one CTA per row for small batches up to 16384 columns."""
    # S5000 benefits from avoiding the multi-pass launches at these shapes;
    # larger batches retain the partitioned path to expose more parallel work.
    return _cooperative_topk_impl(
        logits,
        lengths,
        output,
        workspace,
        k,
        max_seq_len,
        small_max_width=16384,
        small_max_rows=16,
    )
