"""
FX Importer hooks for custom handling of FX node to MLIR translation
"""

from typing import Any

import numpy as np
import torch
from torch_mlir import fx
from torch_mlir import ir as mlir_ir
from torch_mlir.extras.fx_importer import (
    TORCH_DTYPE_TO_MLIR_TYPE,
    TORCH_DTYPE_TO_NPY_TYPE,
    GraphNodeImporter,
    InputInfo,
    ml_dtypes,
)


class CustomFxImporterHooks(fx.FxImporterHooks):
    def resolve_literal(
        self, gni: GraphNodeImporter, literal: Any, info: InputInfo | None
    ) -> mlir_ir.Value | None:
        """Create an MLIR value for a literal found in the FX graph

        This function is called to translate a literal to an MLIR value, this particular
        instance overrides handling of torch.Tensor literals and leaves handling of any others
        to the default implementation by returning None.

        torch.Tensor literals are converted to a torch.vtensor_literal containing an MLIR
        DenseElementsAttr, the default implementation converts them to a DenseResourceElementsAttr
        that XLA's HLO conversion utilities do not handle.

        Args:
            gni: The singleton responsible for converting FX nodes into MLIR
            literal: The literal value to be converted
            info: Optional extra metadata about input nodes

        Returns:
            mlir.Value: The converted value
        """
        if isinstance(literal, torch.Tensor):
            tensor = literal

            vtensor_type = gni._cc.tensor_to_vtensor_type(tensor)
            py_attr_tracker = gni._cc._py_attr_tracker

            # Ref to _make_vtensor_literal_op in fx_importer.py in Torch-mlir
            mapping = py_attr_tracker.track(tensor)
            if mapping.is_empty:
                # check support for bfloat16
                assert not (
                    tensor.dtype == torch.bfloat16 and ml_dtypes is None
                ), "torch.bfloat16 requires the ml_dtypes package"
                # Resolve the attribute.
                npy_dtype = TORCH_DTYPE_TO_NPY_TYPE.get(tensor.dtype)
                assert (
                    npy_dtype is not None
                ), f"Can not create literal tensor for unsupported datatype: {tensor.dtype}"

                np_tensor = np.array(tensor.tolist()).astype(npy_dtype)

                npy_dtype = TORCH_DTYPE_TO_NPY_TYPE.get(tensor.dtype)
                assert npy_dtype is not None, f"Unsupported datatype: {tensor.dtype}"

                np_tensor = np.array(tensor.tolist()).astype(npy_dtype)
                element_type = TORCH_DTYPE_TO_MLIR_TYPE[tensor.dtype]()

                elements_attr = mlir_ir.DenseElementsAttr.get(
                    type=element_type, array=np_tensor, shape=np_tensor.shape
                )
                mapping.value = elements_attr
            else:
                elements_attr = mapping.value
            return mlir_ir.Operation.create(
                name="torch.vtensor.literal",
                results=[vtensor_type],
                attributes={"value": elements_attr},
            ).result

        # Use default handling for any other literal
        return None
