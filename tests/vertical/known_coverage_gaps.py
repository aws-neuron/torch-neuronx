"""Preexisting vertical op gaps on main. The merge-queue coverage gate reads this.

The gate runs the coverage suites with NEURON_COVERAGE_GATE=1, so each op test
asserts: a supported op that reports a real failure status fails; an op on this
allowlist reports xfail instead. An op that already fails on main is not a PR
regression, so it belongs here. To clear an entry, fix the op and delete its
line in the same PR.

Scope, per suite (see the _gate_check helpers in the two coverage suites):
  * eager  -- only a "registered" op gates (a PrivateUse1 kernel we ship). A
    composite op that falls back is a missing primitive, not a shipped kernel;
    the eager unimplemented-ops skip registry owns that set. So the eager list
    stays small.
  * compile -- a registered or composite op, or an op the backend lowers through
    a decomposition (compile_supported_ops), gates. The compile path has no
    in-repo unimplemented-ops registry to defer to, so the preexisting gaps live
    here. A compile timeout is watchdog/load dependent, so it never gates and
    needs no entry here.

The value is a short human reason. The gate matches on the op name only, so the
value is documentation, not a status. Seeded from main run 34895577413.
"""

from __future__ import annotations

# Registered eager kernels that still fall back. All produce a data-dependent
# output shape, so the kernel cannot run on a static Neuron graph.
EAGER_GAPS: dict[str, str] = {
    "aten::_grouped_mm": "dynamic group sizes; falls back",
    "aten::cross_entropy_loss": "composite loss; a primitive falls back",
    "aten::histc": "data-dependent bin counts",
    "aten::masked_scatter": "data-dependent scatter; falls back",
    "aten::masked_scatter.out": "data-dependent scatter; falls back",
    "aten::masked_scatter_": "data-dependent scatter; falls back",
    "aten::masked_select": "data-dependent output shape",
    "aten::masked_select.out": "data-dependent output shape",
    "aten::nonzero": "data-dependent output shape",
    "aten::nonzero.out": "data-dependent output shape",
    "aten::repeat_interleave.Tensor": "data-dependent output shape",
    "aten::repeat_interleave.Tensor_out": "data-dependent output shape",
    "aten::scalar_tensor.out": "scalar construction; falls back",
    # Potential regressions: these passed on the suite's imported shapes but fall
    # back once the suite drops to upstream shapes. Marked xfail pending root-cause.
    "aten::gather": "potential regression; falls back under upstream shapes",
    "aten::ones_like": "potential regression; falls back under upstream shapes",
    # Falls back; pending fix.
    "aten::_fft_r2c.out": "falls back; pending fix",
    # Deliberate CPU bounce, not a lost lowering. int64 indices route through
    # CPU int32 to fix a HostCC NaN and an out-of-bounds read.
    "aten::embedding": "int64 indices routed via CPU int32; HostCC NaN and OOB workaround",
}

# Ops the torch.compile backend cannot lower yet. Grouped by cause.
COMPILE_GAPS: dict[str, str] = {
    # No backend lowering.
    "aten::_grouped_mm": "direct aten op call not supported; graph break covers torch._grouped_mm",
    "aten::_conj_physical": "no compile lowering",
    "aten::_conj_physical.out": "no compile lowering",
    "aten::_fused_adam_.tensor_lr": "fused optimizer; no lowering",
    "aten::_fused_adamw_.tensor_lr": "fused optimizer; no lowering",
    "aten::_log_softmax": "no compile lowering",
    "aten::_native_batch_norm_legit_no_training.out": "batch-norm out variant; no lowering",
    "aten::_thnn_fused_gru_cell_backward": "fused GRU backward; no lowering",
    "aten::_upsample_nearest_exact2d_backward": "no compile lowering",
    "aten::add.Scalar_out": "scalar out variant; no lowering",
    "aten::addmm.dtype_out": "dtype out variant; no lowering",
    "aten::argsort": "no compile lowering",
    "aten::baddbmm.dtype_out": "dtype out variant; no lowering",
    "aten::cross_entropy_loss": "no compile lowering",
    "aten::cumprod_backward": "no compile lowering",
    "aten::cumsum": "no compile lowering",
    "aten::embedding_dense_backward.out": "out variant; no lowering",
    "aten::eye.out": "no compile lowering",
    "aten::histc": "data-dependent bins; no lowering",
    "aten::linalg_slogdet.out": "no linalg lowering",
    "aten::multilabel_margin_loss_forward.output": "no compile lowering",
    "aten::nanmedian": "no compile lowering",
    "aten::nanmedian.out": "no compile lowering",
    "aten::native_layer_norm_backward.out": "out variant; no lowering",
    "aten::nll_loss2d_forward.output": "out variant; no lowering",
    "aten::pixel_unshuffle": "no compile lowering",
    "aten::poisson_nll_loss": "no compile lowering",
    "aten::linalg_svd": "no linalg lowering",
    "aten::linalg_svd.U": "no linalg lowering",
    "aten::linalg_svdvals": "no linalg lowering",
    "aten::linalg_svdvals.out": "no linalg lowering",
    "aten::nan_to_num_": "no compile lowering",
    "aten::native_batch_norm.out": "batch-norm out variant; no lowering",
    "aten::ormqr.out": "no linalg lowering",
    "aten::polar": "no compile lowering",
    "aten::put": "no compile lowering",
    "aten::special_laguerre_polynomial_l.n_scalar_out": "no compile lowering",
    "aten::special_laguerre_polynomial_l.x_scalar_out": "no compile lowering",
    "aten::special_log_ndtr": "no compile lowering",
    "aten::special_spherical_bessel_j0": "no compile lowering",
    "aten::sub.Scalar_out": "scalar out variant; no lowering",
    "aten::tile": "no compile lowering",
    # Graph break: data-dependent shape or an unsupported dynamo construct.
    "aten::_fused_adam_": "fused optimizer; graph break",
    "aten::_fused_adamw_": "fused optimizer; graph break",
    "aten::_native_batch_norm_legit.no_stats_out": "graph break",
    "aten::_unique2.out": "data-dependent output shape; graph break",
    "aten::batch_norm_elemt.out": "graph break",
    "aten::bincount.out": "data-dependent output shape; graph break",
    "aten::masked_select": "data-dependent output shape; graph break",
    "aten::masked_select.out": "data-dependent output shape; graph break",
    "aten::matmul_backward.out": "graph break",
    "aten::nonzero": "data-dependent output shape; graph break",
    "aten::nonzero.out": "data-dependent output shape; graph break",
    "aten::sym_storage_offset": "symbolic-shape query; graph break",
    "aten::unique_consecutive.out": "data-dependent output shape; graph break",
    # neuronx-cc rejects the lowering (COMPILATION FAILED).
    "aten::addbmm": "neuronx-cc compile failure",
    "aten::baddbmm": "neuronx-cc compile failure",
    "aten::baddbmm.out": "neuronx-cc compile failure",
    "aten::eye": "neuronx-cc compile failure",
    "aten::flip": "neuronx-cc compile failure (max() on empty sequence)",
    "aten::flip.out": "neuronx-cc compile failure (max() on empty sequence)",
    "aten::gather": "neuronx-cc compile failure; flaky",
    "aten::repeat": "neuronx-cc compile failure (max() on empty sequence)",
    "aten::repeat.out": "neuronx-cc compile failure (max() on empty sequence)",
    "aten::stack": "neuronx-cc compile failure",
    "aten::stack.out": "neuronx-cc compile failure",
    # The out variant has no meta, so the fake-tensor trace raises TypeError.
    "aten::_weight_norm_interface.out": "out variant; no meta",
    # AOT retains the complex output copy.
    "aten::_fft_r2c.out": "complex decomposition rejects the retained complex copy_",
    "aten::polar.out": "complex decomposition rejects the retained complex copy_",
}
