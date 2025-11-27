# SPDX-License-Identifier: BSD-3-Clause
# This file includes material from the PyTorch XLA project (pytorch-tpu),
# licensed under the BSD 3-Clause License.
# Source: https://github.com/pytorch/xla/blob/master/torchax/torchax/ops/op_base.py
#
# Original copyright:
#   Copyright (c) 2023, pytorch-tpu
#   All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modifications by Amazon Web Services, Inc:
#   Copyright (c) 2025, Amazon Web Services, Inc.
#   - Simplified to include only essential helper functions

"""Helper utilities for JAX operation implementations."""

import jax
import jax.numpy as jnp
import numpy as np


def maybe_convert_constant_dtype(val: float | int | jax.Array | None, dtype: jnp.dtype | None):
    """Optionally converts scalar constant's dtype using `numpy`.

    Use in cases where you require a constant and can't handle a traced array.

    Args:
        val: Value to potentially convert
        dtype: Target dtype for conversion

    Returns:
        Converted value or original if no conversion needed
    """
    if val is not None and dtype is not None:
        if isinstance(val, jax.Array):
            # Extract scalar value from JAX array
            return maybe_convert_constant_dtype(val.item(), dtype)
        return np.array(val, dtype)
    return val
