import unittest

import torch

from torch_neuronx.python_ops.handlers.empty_tensor import BaseEmptyTensorHandler


class TestBaseEmptyTensorHandler(unittest.TestCase):
    def setUp(self):
        self.handler = BaseEmptyTensorHandler()

    def test_empty_tensor(self):
        self.assertTrue(self.handler.check_for_empty(torch.empty(0)))

    def test_non_empty_tensor(self):
        self.assertFalse(self.handler.check_for_empty(torch.randn(3, 4)))

    def test_empty_in_list(self):
        self.assertTrue(self.handler.check_for_empty([torch.empty(0, 5)]))

    def test_empty_in_tuple(self):
        self.assertTrue(self.handler.check_for_empty((torch.empty(2, 0),)))

    def test_nested_empty(self):
        self.assertTrue(self.handler.check_for_empty([[torch.empty(0)]]))

    def test_mixed_args_with_empty(self):
        self.assertTrue(self.handler.check_for_empty(torch.randn(2, 3), torch.empty(0)))

    def test_no_tensors(self):
        self.assertFalse(self.handler.check_for_empty(1, "str", [1, 2]))

    def test_empty_args(self):
        self.assertFalse(self.handler.check_for_empty())


if __name__ == "__main__":
    unittest.main()
