import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestFusedAdamW:
    def _compare_optimizer_states(self, cpu_opt, neuron_opt):
        """Compare optimizer internal states between CPU and Neuron."""
        cpu_params = list(cpu_opt.param_groups[0]["params"])
        neuron_params = list(neuron_opt.param_groups[0]["params"])

        for cpu_param, neuron_param in zip(cpu_params, neuron_params, strict=False):
            cpu_state = cpu_opt.state.get(cpu_param, {})
            neuron_state = neuron_opt.state.get(neuron_param, {})

            # Compare state keys
            assert set(cpu_state.keys()) == set(neuron_state.keys()), (
                f"State keys mismatch: CPU={set(cpu_state.keys())}, "
                f"Neuron={set(neuron_state.keys())}"
            )

            # Compare state values
            for key in cpu_state:
                if key == "step":
                    # Step is a tensor, compare values
                    cpu_step = (
                        cpu_state[key].item() if hasattr(cpu_state[key], "item") else cpu_state[key]
                    )
                    neuron_step = (
                        neuron_state[key].cpu().item()
                        if hasattr(neuron_state[key], "item")
                        else neuron_state[key]
                    )
                    assert (
                        cpu_step == neuron_step
                    ), f"Step count mismatch: CPU={cpu_step}, Neuron={neuron_step}"
                else:
                    # Other state variables (exp_avg, exp_avg_sq, max_exp_avg_sq)
                    torch.testing.assert_close(
                        neuron_state[key].cpu(),
                        cpu_state[key],
                        rtol=1e-5 if cpu_state[key].dtype == torch.float32 else 1e-3,
                        atol=1e-6 if cpu_state[key].dtype == torch.float32 else 1e-3,
                        msg=f"State '{key}' mismatch",
                    )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_fused_adamw_basic(self, dtype):
        """Test AdamW fused accuracy against CPU."""
        torch.manual_seed(42)

        param_cpu = torch.randn(10, dtype=dtype, requires_grad=True)
        param_neuron = param_cpu.detach().clone().to("neuron").requires_grad_(True)

        optimizer_cpu = torch.optim.AdamW([param_cpu], lr=0.001, fused=False)
        optimizer_neuron = torch.optim.AdamW([param_neuron], lr=0.001, fused=True)

        for _ in range(3):
            loss_cpu = param_cpu.sum()
            loss_cpu.backward()
            optimizer_cpu.step()
            optimizer_cpu.zero_grad()

            loss_neuron = param_neuron.sum()
            loss_neuron.backward()

            with track_neuron_ops():
                optimizer_neuron.step()
                assert_op_runs_on_neuron("_fused_adamw_")
            optimizer_neuron.zero_grad()

        self._compare_optimizer_states(optimizer_cpu, optimizer_neuron)

    def test_fused_adamw_multiple_params(self):
        """Test AdamW with multiple parameters of different shapes."""
        torch.manual_seed(42)

        # Params with different shapes
        params_cpu = [
            torch.randn(10, requires_grad=True),
            torch.randn(5, 8, requires_grad=True),
            torch.randn(3, 4, 2, requires_grad=True),
            torch.randn(100, requires_grad=True),
        ]
        params_neuron = [p.detach().clone().to("neuron").requires_grad_(True) for p in params_cpu]

        optimizer_cpu = torch.optim.AdamW(params_cpu, lr=0.01, fused=False)
        optimizer_neuron = torch.optim.AdamW(params_neuron, lr=0.01, fused=True)

        for _ in range(2):
            loss_cpu = sum(p.sum() for p in params_cpu)
            loss_cpu.backward()
            optimizer_cpu.step()
            optimizer_cpu.zero_grad()

            loss_neuron = sum(p.sum() for p in params_neuron)
            loss_neuron.backward()

            with track_neuron_ops():
                optimizer_neuron.step()
                assert_op_runs_on_neuron("_fused_adamw_")
            optimizer_neuron.zero_grad()

        self._compare_optimizer_states(optimizer_cpu, optimizer_neuron)

    @pytest.mark.parametrize(
        "lr,betas,weight_decay,eps",
        [
            (0.001, (0.9, 0.999), 0.01, 1e-8),  # defaults
            (0.1, (0.8, 0.95), 0.01, 1e-6),  # high lr
            (0.01, (0.95, 0.999), 0.1, 1e-7),  # high weight decay
            (0.005, (0.85, 0.9), 0.0, 1e-5),  # no weight decay
        ],
    )
    def test_fused_adamw_different_hyperparams(self, lr, betas, weight_decay, eps):
        """Test AdamW with different hyperparameters."""
        torch.manual_seed(42)

        param_cpu = torch.randn(20, requires_grad=True)
        param_neuron = param_cpu.detach().clone().to("neuron").requires_grad_(True)

        optimizer_cpu = torch.optim.AdamW(
            [param_cpu], lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, fused=False
        )
        optimizer_neuron = torch.optim.AdamW(
            [param_neuron], lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, fused=True
        )

        for _ in range(3):
            loss_cpu = param_cpu.pow(2).sum()
            loss_cpu.backward()
            optimizer_cpu.step()
            optimizer_cpu.zero_grad()

            loss_neuron = param_neuron.pow(2).sum()
            loss_neuron.backward()

            with track_neuron_ops():
                optimizer_neuron.step()
                assert_op_runs_on_neuron("_fused_adamw_")
            optimizer_neuron.zero_grad()

        self._compare_optimizer_states(optimizer_cpu, optimizer_neuron)

    def test_fused_adamw_large_tensors(self):
        """Test AdamW with larger tensors."""
        torch.manual_seed(42)

        param_cpu = torch.randn(1000, 512, requires_grad=True)
        param_neuron = param_cpu.detach().clone().to("neuron").requires_grad_(True)

        optimizer_cpu = torch.optim.AdamW([param_cpu], lr=0.001, fused=False)
        optimizer_neuron = torch.optim.AdamW([param_neuron], lr=0.001, fused=True)

        loss_cpu = param_cpu.mean()
        loss_cpu.backward()
        optimizer_cpu.step()

        loss_neuron = param_neuron.mean()
        loss_neuron.backward()

        with track_neuron_ops():
            optimizer_neuron.step()
            assert_op_runs_on_neuron("_fused_adamw_")

        self._compare_optimizer_states(optimizer_cpu, optimizer_neuron)

    def test_fused_adamw_zero_grad(self):
        """Test AdamW with zero gradients."""
        torch.manual_seed(42)

        param_cpu = torch.randn(10, requires_grad=True)
        param_neuron = param_cpu.detach().clone().to("neuron").requires_grad_(True)

        optimizer_cpu = torch.optim.AdamW([param_cpu], lr=0.001, fused=False)
        optimizer_neuron = torch.optim.AdamW([param_neuron], lr=0.001, fused=True)

        # Set zero gradients
        param_cpu.grad = torch.zeros_like(param_cpu)
        param_neuron.grad = torch.zeros_like(param_neuron)

        optimizer_cpu.step()

        with track_neuron_ops():
            optimizer_neuron.step()
            assert_op_runs_on_neuron("_fused_adamw_")

        self._compare_optimizer_states(optimizer_cpu, optimizer_neuron)

    def test_fused_adamw_mixed_shapes(self):
        """Test AdamW with mixed parameter shapes including scalars and vectors."""
        torch.manual_seed(42)

        params_cpu = [
            torch.randn(1, requires_grad=True),
            torch.randn(50, requires_grad=True),  # 1D
            torch.randn(10, 10, requires_grad=True),  # 2D
            torch.randn(2, 3, 4, requires_grad=True),  # 3D
        ]
        params_neuron = [p.detach().clone().to("neuron").requires_grad_(True) for p in params_cpu]

        optimizer_cpu = torch.optim.AdamW(params_cpu, lr=0.005, weight_decay=0.1, fused=False)
        optimizer_neuron = torch.optim.AdamW(params_neuron, lr=0.005, weight_decay=0.1, fused=True)

        for step in range(2):
            # change loss each step
            if step == 0:
                loss_cpu = sum(p.abs().sum() for p in params_cpu)
                loss_neuron = sum(p.abs().sum() for p in params_neuron)
            else:
                loss_cpu = sum(p.pow(2).mean() for p in params_cpu)
                loss_neuron = sum(p.pow(2).mean() for p in params_neuron)

            loss_cpu.backward()
            optimizer_cpu.step()
            optimizer_cpu.zero_grad()

            loss_neuron.backward()

            with track_neuron_ops():
                optimizer_neuron.step()
                assert_op_runs_on_neuron("_fused_adamw_")
            optimizer_neuron.zero_grad()

        self._compare_optimizer_states(optimizer_cpu, optimizer_neuron)

    def test_fused_adamw_amsgrad(self):
        """Test AdamW with AMSGrad enabled."""
        torch.manual_seed(42)

        param_cpu = torch.randn(20, requires_grad=True)
        param_neuron = param_cpu.detach().clone().to("neuron").requires_grad_(True)

        optimizer_cpu = torch.optim.AdamW([param_cpu], lr=0.01, amsgrad=True, fused=False)
        optimizer_neuron = torch.optim.AdamW([param_neuron], lr=0.01, amsgrad=True, fused=True)

        for _ in range(3):
            loss_cpu = param_cpu.pow(2).sum()
            loss_cpu.backward()
            optimizer_cpu.step()
            optimizer_cpu.zero_grad()

            loss_neuron = param_neuron.pow(2).sum()
            loss_neuron.backward()

            with track_neuron_ops():
                optimizer_neuron.step()
                assert_op_runs_on_neuron("_fused_adamw_")
            optimizer_neuron.zero_grad()

        self._compare_optimizer_states(optimizer_cpu, optimizer_neuron)

    def test_fused_adamw_maximize(self):
        """Test AdamW with maximize=True."""
        torch.manual_seed(42)

        param_cpu = torch.randn(15, requires_grad=True)
        param_neuron = param_cpu.detach().clone().to("neuron").requires_grad_(True)

        optimizer_cpu = torch.optim.AdamW([param_cpu], lr=0.01, maximize=True, fused=False)
        optimizer_neuron = torch.optim.AdamW([param_neuron], lr=0.01, maximize=True, fused=True)

        for _ in range(2):
            loss_cpu = param_cpu.sum()
            loss_cpu.backward()
            optimizer_cpu.step()
            optimizer_cpu.zero_grad()

            loss_neuron = param_neuron.sum()
            loss_neuron.backward()

            with track_neuron_ops():
                optimizer_neuron.step()
                assert_op_runs_on_neuron("_fused_adamw_")
            optimizer_neuron.zero_grad()

        self._compare_optimizer_states(optimizer_cpu, optimizer_neuron)

    def test_fused_adamw_tensor_lr(self):
        """Test AdamW with tensor learning rate."""
        torch.manual_seed(42)

        param_cpu = torch.randn(20, requires_grad=True)
        param_neuron = param_cpu.detach().clone().to("neuron").requires_grad_(True)

        optimizer_cpu = torch.optim.AdamW([param_cpu], lr=torch.tensor(0.01), fused=False)
        optimizer_neuron = torch.optim.AdamW([param_neuron], lr=torch.tensor(0.01), fused=True)

        for _ in range(3):
            loss_cpu = param_cpu.pow(2).sum()
            loss_cpu.backward()
            optimizer_cpu.step()
            optimizer_cpu.zero_grad()

            loss_neuron = param_neuron.pow(2).sum()
            loss_neuron.backward()

            with track_neuron_ops():
                optimizer_neuron.step()
                assert_op_runs_on_neuron("_fused_adamw_")
            optimizer_neuron.zero_grad()

        self._compare_optimizer_states(optimizer_cpu, optimizer_neuron)

    def test_fused_adamw_buffer_donation(self):
        """Test that fused AdamW properly donates buffers for in-place updates."""
        torch.manual_seed(42)

        # Create parameters as leaf tensors on neuron device
        params = [
            torch.randn(50, device="neuron", requires_grad=True),
            torch.randn(10, 10, device="neuron", requires_grad=True),
        ]

        # Store original data pointers
        param_ptrs = [p.data_ptr() for p in params]

        optimizer = torch.optim.AdamW(params, lr=0.001, fused=True)

        # Run a few optimization steps
        for _ in range(3):
            loss = sum(p.sum() for p in params)
            loss.backward()

            with track_neuron_ops():
                optimizer.step()
                assert_op_runs_on_neuron("_fused_adamw_")
            optimizer.zero_grad()

        # Verify parameters were updated in-place (same pointers)
        for i, param in enumerate(params):
            assert (
                param.data_ptr() == param_ptrs[i]
            ), f"Parameter {i} should be updated in-place with buffer donation"

        # Verify optimizer state tensors also maintain their pointers
        for param in params:
            state = optimizer.state[param]
            exp_avg_ptr = state["exp_avg"].data_ptr()
            exp_avg_sq_ptr = state["exp_avg_sq"].data_ptr()

            # Run one more step
            loss = param.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # State tensors should also be updated in-place
            assert state["exp_avg"].data_ptr() == exp_avg_ptr, "exp_avg should be updated in-place"
            assert (
                state["exp_avg_sq"].data_ptr() == exp_avg_sq_ptr
            ), "exp_avg_sq should be updated in-place"

    def test_fused_adamw_amsgrad_buffer_donation(self):
        """Test buffer donation with AMSGrad enabled."""
        torch.manual_seed(42)

        param = torch.randn(30, device="neuron", requires_grad=True)
        param_ptr = param.data_ptr()

        optimizer = torch.optim.AdamW([param], lr=0.01, amsgrad=True, fused=True)

        # Run optimization steps
        for _ in range(3):
            loss = param.pow(2).sum()
            loss.backward()

            with track_neuron_ops():
                optimizer.step()
                assert_op_runs_on_neuron("_fused_adamw_")
            optimizer.zero_grad()

        # Verify parameter was updated in-place
        assert param.data_ptr() == param_ptr, "Parameter should be updated in-place"

        # Verify all optimizer state tensors maintain their pointers
        state = optimizer.state[param]
        exp_avg_ptr = state["exp_avg"].data_ptr()
        exp_avg_sq_ptr = state["exp_avg_sq"].data_ptr()
        max_exp_avg_sq_ptr = state["max_exp_avg_sq"].data_ptr()

        # Run one more step
        loss = param.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # All state tensors should be updated in-place
        assert state["exp_avg"].data_ptr() == exp_avg_ptr, "exp_avg should be updated in-place"
        assert (
            state["exp_avg_sq"].data_ptr() == exp_avg_sq_ptr
        ), "exp_avg_sq should be updated in-place"
        assert (
            state["max_exp_avg_sq"].data_ptr() == max_exp_avg_sq_ptr
        ), "max_exp_avg_sq should be updated in-place with AMSGrad"
