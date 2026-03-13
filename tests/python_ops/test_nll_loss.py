import pytest
import torch
import torch.nn.functional as F  # noqa N812


class TestNLLLoss:
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("weight", [None, torch.tensor([0.2, 0.3, 0.1, 0.2, 0.4])])
    @pytest.mark.parametrize("ignore_index", [-100, 1])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("backward", [True, False])
    def test_nll_loss(self, reduction, weight, ignore_index, dtype, backward):
        """Test nll_loss_forward and nll_loss_backward with different weights
        and reduction methods"""
        torch.manual_seed(123)
        input_tensor = F.log_softmax(torch.randn(5, 5, dtype=dtype), dim=1)
        input_tensor.requires_grad = True
        input_tensor_device = input_tensor.detach().clone().to("neuron")
        if backward:
            input_tensor_device.requires_grad = True
        target = torch.tensor([1, 0, 4, 2, 1]).to("neuron")
        weight_device = weight.to("neuron") if weight is not None else weight

        result = F.nll_loss(
            input_tensor_device,
            target,
            # cast is needed to align test with CPU behavior
            weight=weight_device.to(dtype) if weight_device is not None else weight_device,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        if reduction == "none":
            result = result.sum()
        if backward:
            result.backward()

        expected = F.nll_loss(
            input_tensor.to("cpu"),
            target.to("cpu"),
            # correction needed because otherwise, CPU call will fail
            weight=weight.to(dtype) if weight is not None else weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        if reduction == "none":
            expected = expected.sum()
        if backward:
            expected.backward()

        torch.testing.assert_close(result.cpu(), expected)
        if backward:
            torch.testing.assert_close(input_tensor_device.grad.cpu(), input_tensor.grad)

    def test_nll_loss_1d_input_1d_target_invalid_size(self):
        """Test that 1D input with 1D target of size != 1 raises ValueError."""
        input_tensor = torch.randn(10, device="neuron")
        target = torch.randint(0, 10, (3,), dtype=torch.int64, device="neuron")
        with pytest.raises(ValueError, match="For 1D input, 1D target must have size 1"):
            F.nll_loss(input_tensor, target)
