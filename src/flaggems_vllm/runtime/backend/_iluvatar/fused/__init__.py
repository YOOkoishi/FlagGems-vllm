# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import os

__all__ = []

# Keep unmeasured candidates opt-in until BI-V150 validation establishes their
# correctness and winning shapes. This is read when the package is imported.
if os.environ.get("FLAGGEMS_ILUVATAR_MQA_EXPERIMENTAL", "0").lower() in {
    "1",
    "true",
    "on",
    "yes",
}:
    from .fp8_fp4_mqa_logits import fp8_fp4_mqa_logits
    from .fp8_fp4_paged_mqa_logits import fp8_fp4_paged_mqa_logits

    __all__ = ["fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits"]
