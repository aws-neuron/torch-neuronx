import pytest
import torch
import torch.nn.functional as F  # noqa N812


class TestNLLLoss:
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("weight", [None, torch.tensor([0.2, 0.3, 0.1, 0.2, 0.4])])
    @pytest.mark.parametrize("ignore_index", [-100, 1])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_nll_loss(self, reduction, weight, ignore_index, dtype):
        """Test nll_loss_forward and nll_loss_backward with different weights
        and reduction methods"""
        torch.manual_seed(123)
        input_tensor = F.log_softmax(torch.randn(5, 5, dtype=dtype), dim=1)
        input_tensor.requires_grad = True
        input_tensor_device = input_tensor.detach().clone().to("neuron")
        input_tensor_device.requires_grad = True
        target = torch.tensor([1, 0, 4, 2, 1]).to("neuron")
        weight_device = weight.to("neuron") if weight is not None else weight

        result = F.nll_loss(
            input_tensor_device,
            target,
            weight=weight_device,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        if reduction == "none":
            result = result.sum()
        result.backward()

        expected = F.nll_loss(
            input_tensor.to("cpu"),
            target.to("cpu"),
            weight=weight.to(dtype) if weight is not None else weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        if reduction == "none":
            expected = expected.sum()
        expected.backward()

        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(input_tensor_device.grad.cpu(), input_tensor.grad)
