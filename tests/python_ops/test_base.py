import unittest
from unittest.mock import patch

import torch

from torch_neuronx.python_ops.base import (
    _build_args_kwargs_info,
    _format_tensor_info,
    _format_value_info,
)


class TestBaseFormatting(unittest.TestCase):
    """Test the modularized formatting functions in base.py"""

    def test_format_tensor_info(self):
        """Test _format_tensor_info function with normal and error cases"""
        tensor = torch.randn(2, 3, dtype=torch.float32, device="cpu")
        result = _format_tensor_info(tensor)

        expected_parts = [
            "Tensor(shape=torch.Size([2, 3])",
            "dtype=torch.float32",
            "device=cpu",
            "dispatch keys:",
        ]
        for part in expected_parts:
            self.assertIn(part, result)

    @patch("torch_neuronx.python_ops.base._get_dispatch_keys")
    def test_format_tensor_info_dispatch_error(self, mock_dispatch):
        """Test _format_tensor_info when dispatch keys fail"""
        mock_dispatch.side_effect = Exception("test error")
        tensor = torch.randn(2, 3)
        result = _format_tensor_info(tensor)

        self.assertIn("dispatch keys: unavailable", result)

    def test_format_value_info_integration(self):
        """Test _format_value_info with key types and _build_args_kwargs_info integration"""
        tensor = torch.randn(1, 2, dtype=torch.float32)
        mixed_list = [tensor, 42]
        args = (tensor, mixed_list)
        kwargs = {"out": tensor, "dim": 1}

        args_info, kwargs_info = _build_args_kwargs_info(args, kwargs)

        # Verify args formatting
        self.assertEqual(len(args_info), 2)
        self.assertTrue(
            args_info[0].startswith(
                "arg0: Tensor(shape=torch.Size([1, 2]), "
                "dtype=torch.float32, device=cpu, dispatch keys:"
            )
        )
        self.assertTrue(
            args_info[1].startswith(
                "arg1: list([0]: Tensor(shape=torch.Size([1, 2]), "
                "dtype=torch.float32, device=cpu, dispatch keys:"
            )
        )
        self.assertIn("[1]: int=42)", args_info[1])

        # Verify kwargs formatting
        self.assertEqual(len(kwargs_info), 2)
        out_info = next(info for info in kwargs_info if info.startswith("out="))
        dim_info = next(info for info in kwargs_info if info.startswith("dim="))

        self.assertTrue(
            out_info.startswith(
                "out=Tensor(shape=torch.Size([1, 2]), dtype=torch.float32, "
                "device=cpu, dispatch keys:"
            )
        )
        self.assertEqual(dim_info, "dim=int=1")
        self.assertTrue(args_info[0].startswith("arg0: Tensor(shape=torch.Size([1, 2])"))
        self.assertIn("[0]: Tensor(", args_info[1])
        self.assertIn("[1]: int=42", args_info[1])

        # Verify kwargs formatting
        out_info = next(info for info in kwargs_info if info.startswith("out="))
        self.assertTrue(out_info.startswith("out=Tensor(shape=torch.Size([1, 2])"))
        self.assertIn("dim=int=1", kwargs_info)


if __name__ == "__main__":
    unittest.main()
