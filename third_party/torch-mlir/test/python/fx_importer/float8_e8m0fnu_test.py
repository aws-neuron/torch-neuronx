# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

"""Tests for float8_e8m0fnu (OCP MX E8M0 scale dtype) support in the fx_importer.

float8_e8m0fnu is an 8-bit floating point format with:
  - 0 mantissa bits (no fractional part)
  - 8 exponent bits
  - No sign bit (unsigned, positive powers of two only)
  - No zero, infinity or NaN (except 0xFF = NaN)

It is primarily used as a block-scaling dtype in OCP MX formats (e.g. MXFP8).
"""

import torch
import torch.nn as nn
from torch.export import Dim

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


# ---------------------------------------------------------------------------
# Skip entire test file gracefully if this PyTorch build lacks e8m0fnu.
# ---------------------------------------------------------------------------
if not hasattr(torch, "float8_e8m0fnu"):
    print("float8_e8m0fnu not available in this PyTorch build; skipping tests.")
    raise SystemExit(0)


@run
# CHECK-LABEL: test_e8m0fnu_identity_2d
# CHECK: func.func @main(%[[ARG0:.+]]: !torch.vtensor<[4,8],f8E8M0FNU>) -> !torch.vtensor<[4,8],f8E8M0FNU>
# CHECK:   return %[[ARG0]] : !torch.vtensor<[4,8],f8E8M0FNU>
def test_e8m0fnu_identity_2d():
    """Identity function: verifies 2-D e8m0fnu tensors map to f8E8M0FNU in the IR."""

    class Identity(nn.Module):
        def forward(self, x):
            return x

    m = fx.export_and_import(Identity(), torch.ones(4, 8, dtype=torch.float8_e8m0fnu))
    print(m)


@run
# CHECK-LABEL: test_e8m0fnu_identity_1d
# CHECK: func.func @main(%[[ARG0:.+]]: !torch.vtensor<[32],f8E8M0FNU>) -> !torch.vtensor<[32],f8E8M0FNU>
# CHECK:   return %[[ARG0]] : !torch.vtensor<[32],f8E8M0FNU>
def test_e8m0fnu_identity_1d():
    """Identity function: verifies 1-D e8m0fnu tensors map correctly."""

    class Identity(nn.Module):
        def forward(self, x):
            return x

    m = fx.export_and_import(Identity(), torch.ones(32, dtype=torch.float8_e8m0fnu))
    print(m)


@run
# CHECK-LABEL: test_e8m0fnu_dynamic_batch
# CHECK: func.func @dyn_e8m0fnu(%[[ARG0:.+]]: !torch.vtensor<[?,8],f8E8M0FNU>) -> !torch.vtensor<[?,8],f8E8M0FNU>
# CHECK:   return %[[ARG0]] : !torch.vtensor<[?,8],f8E8M0FNU>
def test_e8m0fnu_dynamic_batch():
    """Dynamic first dimension: verifies '?' shape propagates to the IR."""

    class Identity(nn.Module):
        def forward(self, x):
            return x

    batch = Dim("batch", max=32)
    m = fx.export_and_import(
        Identity(),
        torch.ones(4, 8, dtype=torch.float8_e8m0fnu),
        dynamic_shapes={"x": {0: batch}},
        func_name="dyn_e8m0fnu",
    )
    print(m)


@run
# CHECK-LABEL: test_e8m0fnu_view
# CHECK: func.func @main(%[[ARG0:.+]]: !torch.vtensor<[4,8],f8E8M0FNU>) -> !torch.vtensor<[8,4],f8E8M0FNU>
# CHECK:   %[[INT8:.+]] = torch.constant.int 8
# CHECK:   %[[INT4:.+]] = torch.constant.int 4
# CHECK:   %[[LIST:.+]] = torch.prim.ListConstruct %[[INT8]], %[[INT4]]
# CHECK:   %[[VIEW:.+]] = torch.aten.view %[[ARG0]], %[[LIST]] : !torch.vtensor<[4,8],f8E8M0FNU>, !torch.list<int> -> !torch.vtensor<[8,4],f8E8M0FNU>
# CHECK:   return %[[VIEW]] : !torch.vtensor<[8,4],f8E8M0FNU>
def test_e8m0fnu_view():
    """View/reshape: verifies that aten.view preserves f8E8M0FNU element type."""

    class ViewMod(nn.Module):
        def forward(self, x):
            return x.view(8, 4)

    m = fx.export_and_import(ViewMod(), torch.ones(4, 8, dtype=torch.float8_e8m0fnu))
    print(m)


@run
# CHECK-LABEL: test_cast_f32_to_e8m0fnu
# CHECK: func.func @cast_to_e8m0fnu(%[[ARG0:.+]]: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],f8E8M0FNU>
# CHECK:   %[[DTYPE:.+]] = torch.constant.int 44
# CHECK:   %[[OUT:.+]] = torch.prims.convert_element_type %[[ARG0]], %[[DTYPE]] : !torch.vtensor<[4,8],f32>, !torch.int -> !torch.vtensor<[4,8],f8E8M0FNU>
# CHECK:   return %[[OUT]] : !torch.vtensor<[4,8],f8E8M0FNU>
def test_cast_f32_to_e8m0fnu():
    """Cast f32 -> e8m0fnu: dtype int 44 and prims.convert_element_type are expected."""

    class CastToE8M0(nn.Module):
        def forward(self, x):
            return x.to(torch.float8_e8m0fnu)

    m = fx.export_and_import(
        CastToE8M0(), torch.ones(4, 8), func_name="cast_to_e8m0fnu"
    )
    print(m)


@run
# CHECK-LABEL: test_cast_e8m0fnu_to_f32
# CHECK: func.func @cast_to_f32(%[[ARG0:.+]]: !torch.vtensor<[4,8],f8E8M0FNU>) -> !torch.vtensor<[4,8],f32>
# CHECK:   %[[DTYPE:.+]] = torch.constant.int 6
# CHECK:   %{{.+}} = torch.prims.convert_element_type %[[ARG0]], %[[DTYPE]] : !torch.vtensor<[4,8],f8E8M0FNU>, !torch.int -> !torch.vtensor<[4,8],f32>
# CHECK:   return %{{.+}} : !torch.vtensor<[4,8],f32>
def test_cast_e8m0fnu_to_f32():
    """Cast e8m0fnu -> f32: dtype int 6 (torch.float32) is expected."""

    class CastToF32(nn.Module):
        def forward(self, x):
            return x.to(torch.float32)

    m = fx.export_and_import(
        CastToF32(),
        torch.ones(4, 8, dtype=torch.float8_e8m0fnu),
        func_name="cast_to_f32",
    )
    print(m)


@run
# CHECK-LABEL: test_e8m0fnu_frozen_buffer
# CHECK: func.func @main(%{{.+}}: !torch.vtensor<[4],f8E8M0FNU>, %{{.+}}: !torch.vtensor<[4],f8E8M0FNU>) -> !torch.vtensor<[4],f8E8M0FNU>
# CHECK:   return %{{.+}} : !torch.vtensor<[4],f8E8M0FNU>
def test_e8m0fnu_frozen_buffer():
    """Module with a frozen e8m0fnu buffer: the buffer becomes a function argument
    (or literal) with element type f8E8M0FNU in the imported IR."""

    class WithScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "scale",
                torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float8_e8m0fnu),
            )

        def forward(self, x):
            # Return x unchanged; just verifying the scale buffer type in IR.
            return x

    m = fx.export_and_import(WithScale(), torch.ones(4, dtype=torch.float8_e8m0fnu))
    print(m)
