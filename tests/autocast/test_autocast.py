import collections
from itertools import product

import pytest
import torch
from torch.testing._internal.autocast_test_lists import (
    AutocastTestLists,
    TestAutocast,
)
from torch.utils.checkpoint import checkpoint_sequential

TEST_BF16 = True


class TestNeuronAutocast(TestAutocast):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.autocast_lists = AutocastTestLists(torch.device("neuron:0"))
        self.disable_tests_for_torch_fp16 = [
            "_convolution",
            "cudnn_convolution",
            "cudnn_convolution_transpose",
            "einsum",
            "lstm_cell",
            "gru_cell",
            "rnn_tanh_cell",
            "rnn_relu_cell",
        ]

    def tearDown(self):
        del self.autocast_lists
        super().tearDown()

    @pytest.mark.xfail(
        reason="Flaky due to precision differences between Neuron and CPU. "
        "PyTorch's TestAutocast uses torch.equal which requires exact match."
    )
    def test_autocast_torch_fp16(self):
        for op_with_args in self.autocast_lists.torch_fp16:
            op, args = op_with_args[0], op_with_args[1]
            if op in self.disable_tests_for_torch_fp16:
                continue
            self._run_autocast_outofplace(
                op, args, torch.float16, device="neuron", amp_dtype=torch.float16
            )

    @pytest.mark.xfail(
        reason="Flaky due to precision differences between Neuron and CPU. "
        "PyTorch's TestAutocast uses torch.equal which requires exact match."
    )
    def test_autocast_torch_bf16(self):
        for op_with_args in self.autocast_lists.torch_fp16:
            op, args = op_with_args[0], op_with_args[1]
            if op in self.disable_tests_for_torch_fp16:
                continue
            self._run_autocast_outofplace(op, args, torch.bfloat16, device="neuron")

    @pytest.mark.xfail(
        reason="[PT2.12] Neuron aten/wheel nightly failure (run 28991277106): likely profiler "
        "regression (profiler/memory/autocast). Xfailing to unblock nightly pending root-cause."
    )
    def test_autocast_torch_fp32(self):
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            if "rsqrt" in op:
                continue
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="neuron",
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    def test_autocast_torch_need_autocast_promote(self):
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(
                op, args, torch.float32, device="neuron", amp_dtype=torch.float16
            )

    def test_autocast_torch_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="neuron",
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    def test_autocast_nn_fp16(self):
        for op, args in self.autocast_lists.nn_fp16:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float16,
                device="neuron",
                module=torch._C._nn,
                amp_dtype=torch.float16,
            )

    @pytest.mark.xfail(
        reason="maximum recursion depth exceeded while getting the repr of an object,"
        "issue with copy_"
    )
    def test_autocast_nn_bf16(self):
        for op, args in self.autocast_lists.nn_fp16:
            self._run_autocast_outofplace(
                op, args, torch.bfloat16, device="neuron", module=torch._C._nn
            )

    def test_autocast_nn_fp32(self):
        for op, args in self.autocast_lists.nn_fp32:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="neuron",
                module=torch._C._nn,
                amp_dtype=torch.float16,
            )

    def test_autocast_linalg_fp16(self):
        for op, args in self.autocast_lists.linalg_fp16:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float16,
                device="neuron",
                module=torch._C._linalg,
                amp_dtype=torch.float16,
            )

    def test_autocast_methods_fp16(self):
        for op, args in self.autocast_lists.methods_fp16:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float16,
                device="neuron",
                module=None,
                amp_dtype=torch.float16,
            )

    def test_autocast_methods_fp32(self):
        for op, args in self.autocast_lists.methods_fp32:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="neuron",
                module=None,
                amp_dtype=torch.float16,
            )

    def test_autocast_methods_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="neuron",
                module=None,
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    def test_autocast_banned(self):
        with torch.autocast("neuron"):
            for op, args, module in self.autocast_lists.banned:
                with self.assertRaises(RuntimeError):
                    getattr(module, op)(*args)

    def test_autocast_ignored_types(self):
        with torch.autocast("neuron"):
            for ignore_type in (torch.double, torch.int32):
                a_ignore = torch.ones((8, 8), dtype=ignore_type, device="neuron")
                b_ignore = torch.ones((8, 8), dtype=ignore_type, device="neuron")

                # Tests if CastPolicy::fp16 ops ignore double and int
                # Currently, no ops belonging to this policy support integer inputs.
                if ignore_type is torch.double:
                    with torch.autocast("neuron", enabled=False):
                        type_no_autocast = torch.mm(a_ignore, b_ignore).dtype
                    self.assertTrue(torch.mm(a_ignore, b_ignore).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32 ops ignore double and int
                with torch.autocast("neuron", enabled=False):
                    type_no_autocast = torch.pow(a_ignore, 2.0).dtype
                self.assertTrue(torch.pow(a_ignore, 2.0).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32_set_opt_dtype ops ignore double and int
                with torch.autocast("neuron", enabled=False):
                    type_no_autocast = torch.sum(a_ignore).dtype
                self.assertTrue(torch.sum(a_ignore).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32_append_dtype ops ignore double and int
                # Currently, no ops belonging to this policy support integer inputs.
                if ignore_type is torch.double:
                    with torch.autocast("neuron", enabled=False):
                        type_no_autocast = torch.norm(a_ignore).dtype
                    self.assertTrue(torch.norm(a_ignore).dtype is type_no_autocast)

    @pytest.mark.xfail(
        reason="maximum recursion depth exceeded while getting the repr of an object, "
        "issue with copy_"
    )
    def test_autocast_custom_enabled(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.amp.custom_fwd(device_type="neuron")
            def forward(ctx, a, b):
                self.assertTrue(a.dtype is torch.float32)
                self.assertTrue(b.dtype is torch.float32)
                self.assertTrue(torch.is_autocast_enabled("neuron"))
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.amp.custom_bwd(device_type="neuron")
            def backward(ctx, grad):
                self.assertTrue(torch.is_autocast_enabled("neuron"))
                a, b = ctx.saved_tensors
                a_grad, b_grad = grad.mm(b.t()), a.t().mm(grad)
                self.assertTrue(a_grad.dtype is dtype and b_grad.dtype is dtype)
                return a_grad, b_grad

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="neuron", dtype=torch.float32, requires_grad=True)
        y = torch.randn((8, 8), device="neuron", dtype=torch.float32, requires_grad=True)

        dtypes = (torch.float16, torch.bfloat16) if TEST_BF16 else (torch.float16,)
        for dtype in dtypes:
            with torch.autocast(device_type="neuron", dtype=dtype):
                output = mymm(x, y)
                self.assertTrue(output.dtype is dtype)
                loss = output.sum()
            loss.backward()

    def test_autocast_custom_cast_inputs(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.amp.custom_fwd(device_type="neuron", cast_inputs=torch.float32)
            def forward(ctx, a, container, expect_type):
                b = container[1][0]
                self.assertTrue(a.dtype is expect_type)
                self.assertTrue(b.dtype is expect_type)
                self.assertFalse(torch.is_autocast_enabled("neuron"))
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.amp.custom_bwd(device_type="neuron")
            def backward(ctx, grad):
                self.assertFalse(torch.is_autocast_enabled("neuron"))
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), None, None

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="neuron", dtype=torch.float16, requires_grad=True)
        # Puts one input tensor in a nested container.  y's contained Tensor won't receive
        # a gradient,because torch.autograd.Function can't hand gradients back to
        # non-Tensor forward arguments. Sets requires_grad=False explicitly so we don't
        # lie about expecting a gradient.
        y = (
            0,
            {0: torch.randn((8, 8), device="neuron", dtype=torch.float16, requires_grad=False)},
        )

        with torch.autocast("neuron"):
            output = mymm(x, y, torch.float32)
            self.assertTrue(output.dtype is torch.float32)
            loss = output.sum()
        loss.backward()

        # Tests if custom_fwd becomes a no-op when mymm runs outside an autocast-enabled region.
        output = mymm(x, y, torch.float16)
        self.assertTrue(output.dtype is torch.float16)
        loss = output.sum()
        loss.backward()

    def test_autocast_cat_jit(self):
        # Reported at https://github.com/pytorch/pytorch/issues/38958

        class Model(torch.nn.Module):
            def forward(self):
                a = torch.randn(1)
                b = torch.randn(1)
                c = torch.cat((a, b), 0)
                d = torch.stack([c, c], 0)
                return d

        # The JIT here doesn't really matter, we just need to call
        # cat via the boxed API
        model = Model()
        model_jit_script = torch.jit.script(model)

        with torch.autocast("neuron", enabled=True):
            model()
            model_jit_script()

    def test_autocast_checkpointing(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        ).to("neuron")
        input = torch.rand((8, 8), device="neuron", dtype=torch.float16, requires_grad=True)
        for reentrant in (True, False):
            with torch.autocast("neuron"):
                output = checkpoint_sequential(model, 2, input, use_reentrant=reentrant)
            self.assertTrue(output.requires_grad)
            self.assertTrue(output.dtype is torch.float16)
            output.sum().backward()

    def test_autocast_cross_entropy_fp32(self):
        """Test that cross_entropy_loss is cast to fp32 during autocast."""
        input = torch.randn(3, 5, device="neuron", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, 5, (3,), device="neuron")

        with torch.autocast("neuron", dtype=torch.bfloat16):
            output = torch.nn.functional.cross_entropy(input, target)
            self.assertTrue(output.dtype is torch.float32)
