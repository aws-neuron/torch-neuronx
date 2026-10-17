# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

"""Tests for uint32 dtype support in the fx_importer."""

import torch
import torch.nn as nn

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_uint32_identity
# CHECK: func.func @main(%[[ARG0:.+]]: !torch.vtensor<[4,8],ui32>) -> !torch.vtensor<[4,8],ui32>
# CHECK:   return %[[ARG0]] : !torch.vtensor<[4,8],ui32>
def test_uint32_identity():
    """Identity: verifies uint32 tensors map to ui32 in the IR."""

    class Identity(nn.Module):
        def forward(self, x):
            return x

    m = fx.export_and_import(Identity(), torch.zeros(4, 8, dtype=torch.uint32))
    print(m)


@run
# CHECK-LABEL: test_cast_int32_to_uint32
# CHECK: func.func @cast_to_uint32(%[[ARG0:.+]]: !torch.vtensor<[4,8],si32>) -> !torch.vtensor<[4,8],ui32>
# CHECK:   %[[DTYPE:.+]] = torch.constant.int 29
# CHECK:   %{{.+}} = torch.prims.convert_element_type %[[ARG0]], %[[DTYPE]] : !torch.vtensor<[4,8],si32>, !torch.int -> !torch.vtensor<[4,8],ui32>
# CHECK:   return %{{.+}} : !torch.vtensor<[4,8],ui32>
def test_cast_int32_to_uint32():
    """Cast int32 -> uint32: dtype int 29 and prims.convert_element_type are expected."""

    class CastToUInt32(nn.Module):
        def forward(self, x):
            return x.to(torch.uint32)

    m = fx.export_and_import(
        CastToUInt32(), torch.zeros(4, 8, dtype=torch.int32), func_name="cast_to_uint32"
    )
    print(m)


@run
# CHECK-LABEL: test_cast_uint32_to_int64
# CHECK: func.func @cast_to_int64(%[[ARG0:.+]]: !torch.vtensor<[4,8],ui32>) -> !torch.vtensor<[4,8],si64>
# CHECK:   %[[DTYPE:.+]] = torch.constant.int 4
# CHECK:   %{{.+}} = torch.prims.convert_element_type %[[ARG0]], %[[DTYPE]] : !torch.vtensor<[4,8],ui32>, !torch.int -> !torch.vtensor<[4,8],si64>
# CHECK:   return %{{.+}} : !torch.vtensor<[4,8],si64>
def test_cast_uint32_to_int64():
    """Cast uint32 -> int64: dtype int 4 (torch.int64) is expected."""

    class CastToInt64(nn.Module):
        def forward(self, x):
            return x.to(torch.int64)

    m = fx.export_and_import(
        CastToInt64(),
        torch.zeros(4, 8, dtype=torch.uint32),
        func_name="cast_to_int64",
    )
    print(m)
