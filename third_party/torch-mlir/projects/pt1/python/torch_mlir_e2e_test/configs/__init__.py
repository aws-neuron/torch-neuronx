# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# Configs that need CMake-only C extensions are optional imports.
try:
    from .lazy_tensor_core import LazyTensorCoreTestConfig
except ImportError:
    pass
from .native_torch import NativeTorchTestConfig
try:
    from .onnx_backend import OnnxBackendTestConfig
except ImportError:
    pass
try:
    from .torchscript import TorchScriptTestConfig
except ImportError:
    pass
try:
    from .torchdynamo import TorchDynamoTestConfig
except ImportError:
    pass
try:
    from .jit_importer_backend import JITImporterTestConfig
except ImportError:
    pass
from .fx_importer_backend import FxImporterTestConfig
