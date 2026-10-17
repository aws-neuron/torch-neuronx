"""Tests for non-standard bool tensor value handling on Neuron.

PyTorch bool tensors are stored as uint8. While values are
typically 0 or 1, the spec allows any non-zero value to mean
True. The CPU backend normalizes before arithmetic
(bool(3) -> 1.0), but a bug in the Neuron backend causes raw
byte values to leak through during bool -> numeric promotion.

See: BOOL_HANDLING_PROBLEM_STATEMENT.md
"""

import pytest
import torch


def _make_nonstandard_bool(values, device="cpu"):
    """Create a bool tensor with non-standard values."""
    raw = torch.tensor(values, dtype=torch.uint8)
    return raw.view(torch.bool).to(device)


def _expected_normalized(values):
    """Return the expected normalized bool tensor (any non-zero -> True)."""
    return torch.tensor(values, dtype=torch.uint8).view(torch.bool)


class TestNonStandardBoolValues:
    """Tests that non-standard bool values are handled correctly on Neuron."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("neuron", 0)
        # Raw uint8 values: 0 is False, anything else is True
        self.raw_values = [0, 1, 2, 0, 3, 0, 5, 0, 7]
        self.bool_neuron = _make_nonstandard_bool(self.raw_values, self.device)
        self.expected = _expected_normalized(self.raw_values)

    # ── Round-trip and identity ────────────────────────────────────────

    def test_round_trip(self):
        """Device transfer should normalize bool values to 0/1."""
        result = self.bool_neuron.cpu()
        assert torch.equal(result, self.expected)

    def test_clone(self):
        """Clone should preserve normalized values."""
        result = self.bool_neuron.clone().cpu()
        assert torch.equal(result, self.expected)

    # ── Arithmetic ops ─────────────────────────────────────────────────
    @pytest.mark.parametrize(
        "op",
        [
            torch.exp,
            torch.log2,
            pytest.param(
                torch.rsqrt,
                marks=pytest.mark.xfail(reason="unrelated rsqrt bug, tracked separately"),
            ),
        ],
        ids=["exp", "log2", "rsqrt"],
    )
    def test_unary_float_op(self, op):
        """Unary float ops should use 1.0 for True, not raw byte."""
        result = op(self.bool_neuron).cpu()
        expected = op(self.expected)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_mul_bool_bool(self):
        """True * True = 1, not raw * raw."""
        b1 = _make_nonstandard_bool([0, 2, 3], self.device)
        b2 = _make_nonstandard_bool([0, 3, 5], self.device)
        result = (b1 * b2).int().cpu()
        n1 = _expected_normalized([0, 2, 3])
        n2 = _expected_normalized([0, 3, 5])
        expected = (n1 * n2).int()
        assert torch.equal(result, expected)

    # ── Comparison ops ─────────────────────────────────────────────────

    def test_eq_true(self):
        """bool == True should compare normalized values."""
        result = (self.bool_neuron == True).cpu()  # noqa: E712
        expected = self.expected == True  # noqa: E712
        assert torch.equal(result, expected)

    def test_eq_between_nonstandard_bools(self):
        """Two non-standard bools that are both True should be equal."""
        a = _make_nonstandard_bool([2], self.device)
        b = _make_nonstandard_bool([5], self.device)
        assert torch.equal(a, b)

    def test_ne_false(self):
        """bool_tensor != False should use normalized semantics."""
        result = (self.bool_neuron != False).cpu()  # noqa: E712
        expected = self.expected != False  # noqa: E712
        assert torch.equal(result, expected)

    # ── Type conversions ───────────────────────────────────────────────

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.int32, torch.bfloat16],
        ids=["float32", "int32", "bfloat16"],
    )
    def test_to_dtype(self, dtype):
        """bool.to(dtype) should give 0/1 values, not raw bytes."""
        result = self.bool_neuron.to(dtype).cpu()
        expected = self.expected.to(dtype)
        assert torch.equal(result, expected)

    @pytest.mark.parametrize(
        "method, dtype",
        [
            ("int", torch.int32),
            ("long", torch.int64),
            ("float", torch.float32),
            ("double", torch.float64),
            ("half", torch.float16),
            ("bfloat16", torch.bfloat16),
        ],
        ids=["int", "long", "float", "double", "half", "bfloat16"],
    )
    def test_cast_method(self, method, dtype):
        """bool.int(), .float(), etc. should give 0/1, not raw bytes."""
        result = getattr(self.bool_neuron, method)().cpu()
        expected = getattr(self.expected, method)()
        assert torch.equal(result, expected)

    # ── Logical ops ────────────────────────────────────────────────────

    def test_advanced_indexing_preserves_bool(self):
        """Advanced indexing on a non-standard bool tensor should
        return normalized bool values (gather normalizes on Neuron)."""
        t = _make_nonstandard_bool([[0, 3, 7], [5, 0, 2]], self.device)
        result = t[[0]].cpu()
        expected = _expected_normalized([[0, 3, 7]])
        assert torch.equal(result, expected)

    def test_logical_and(self):
        """Logical ops should treat any non-zero as truthy."""
        other = _expected_normalized(self.raw_values).to(self.device)
        result = torch.logical_and(self.bool_neuron, other).cpu()
        expected = torch.logical_and(self.expected, _expected_normalized(self.raw_values))
        assert torch.equal(result, expected)

    def test_logical_or(self):
        """Logical OR should treat any non-zero as truthy."""
        # Invert the pattern: swap True/False
        inv_values = [1 if v == 0 else 0 for v in self.raw_values]
        other = _expected_normalized(inv_values).to(self.device)
        result = torch.logical_or(self.bool_neuron, other).cpu()
        expected = torch.logical_or(self.expected, _expected_normalized(inv_values))
        assert torch.equal(result, expected)

    # ── On-device creation (torch.where + view) ───────────────────────

    def test_where_view_bool_assert_close(self):
        """Create non-standard bytes on-device via torch.where,
        view as bool, then assert_close against original."""
        x = torch.tensor(
            [True, False, True, True, False],
            device=self.device,
        )
        true_vals = torch.randint(2, 255, x.shape, dtype=torch.uint8, device=self.device)
        false_vals = torch.zeros((), dtype=torch.uint8, device=self.device)
        x_int = torch.where(x, true_vals, false_vals)
        ret = x_int.view(torch.bool)
        torch.testing.assert_close(ret, x)

    def test_assert_close_nonstandard_vs_standard(self):
        """assert_close should pass when comparing non-standard
        bool tensor against a standard one."""
        # raw_values[:4] = [0, 1, 2, 0] -> [False, True, True, False]
        standard = torch.tensor(
            [False, True, True, False],
            device=self.device,
        )
        torch.testing.assert_close(self.bool_neuron[:4], standard)

    # ── On-device bool creation without CPU transfer ──────────────────

    def test_randint_view_bool_on_device(self):
        """Non-standard bytes created entirely on-device should
        still compare correctly after normalization."""
        raw = torch.randint(2, 255, (10,), dtype=torch.uint8, device=self.device)
        viewed = raw.view(torch.bool)
        all_true = torch.ones(10, dtype=torch.bool, device=self.device)
        torch.testing.assert_close(viewed, all_true)

    def test_zeros_view_bool_on_device(self):
        """Zero bytes viewed as bool should all be False."""
        raw = torch.zeros(5, dtype=torch.uint8, device=self.device)
        viewed = raw.view(torch.bool)
        all_false = torch.zeros(5, dtype=torch.bool, device=self.device)
        torch.testing.assert_close(viewed, all_false)

    # ── Arithmetic with on-device non-standard bools ──────────────────

    @pytest.mark.parametrize(
        "op_fn, vals_a, vals_b",
        [(torch.Tensor.add, [0, 3, 7], [5, 0, 2])],
        ids=["add"],
    )
    def test_binary_int_op_nonstandard_bools(self, op_fn, vals_a, vals_b):
        """Binary int ops on non-standard bools should use 0/1."""
        a = _make_nonstandard_bool(vals_a, self.device)
        b = _make_nonstandard_bool(vals_b, self.device)
        result = op_fn(a.int(), b.int()).cpu()
        ea = _expected_normalized(vals_a).int()
        eb = _expected_normalized(vals_b).int()
        assert torch.equal(result, op_fn(ea, eb))

    # ── 2D tensor ─────────────────────────────────────────────────────

    def test_2d_nonstandard_bool(self):
        """Non-standard bool normalization works for 2D tensors."""
        raw = torch.tensor([[0, 2, 3], [4, 0, 6]], dtype=torch.uint8)
        t = raw.to(self.device).view(torch.bool)
        result = t.float().cpu()
        expected = raw.view(torch.bool).float()
        assert torch.equal(result, expected)

    # ── Edge cases ────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "raw_values, expected_vals",
        [
            ([0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0]),
            ([2, 3, 127, 128, 255], [1.0, 1.0, 1.0, 1.0, 1.0]),
            ([200], [1.0]),
        ],
        ids=["all_zeros", "all_nonstandard_true", "single_element"],
    )
    def test_edge_case_float_values(self, raw_values, expected_vals):
        """Non-standard bool edge cases should normalize correctly."""
        t = _make_nonstandard_bool(raw_values, self.device)
        result = t.float().cpu()
        assert torch.equal(result, torch.tensor(expected_vals))

    def test_empty_tensor(self):
        """Empty bool tensor should not crash."""
        t = torch.tensor([], dtype=torch.bool).to(self.device)
        result = t.float().cpu()
        assert result.numel() == 0
