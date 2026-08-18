"""Neuron blocklist — files and directories to fully ignore during collection.

Uses SkipGroup format for consistency with skip/xfail registries.
The conftest converts these into collect_ignore_glob patterns automatically:
  - Files (ending in .py):  entry → **/entry
  - Directories (no .py):   entry → **/entry/**
"""

from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

BLOCKLIST: list[SkipGroup] = [
    # =============================================================================
    # Distributed — files and subdirectories
    # =============================================================================
    SkipGroup(
        reason=(
            "CUDA/NCCL hardware-specific tests — raw NCCL API, CUDA P2P IPC, "
            "CuPy __cuda_array_interface__, NVLink symmetric memory, CUDA "
            "graphs, and NCCL flight recorder trace format"
        ),
        tests=[
            "distributed/test_nccl.py",
            "distributed/test_p2p_ipc.py",
            "distributed/test_cupy_as_tensor.py",
            "distributed/test_symmetric_memory.py",
            "dynamo/test_cudagraphs.py",
            "dynamo/test_cudagraphs_expandable_segments.py",
            "distributed/flight_recorder",
            "test_cuda.py",
            "test_cuda_compatibility.py",
            "test_cuda_expandable_segments.py",
            "test_cuda_multigpu.py",
            "test_cuda_nvml_based_avail.py",
            "test_cuda_primary_ctx.py",
            "test_cuda_sanitizer.py",
            "test_cuda_trace.py",
            "test_matmul_cuda.py",
            "test_scaled_matmul_cuda.py",
            "test_jiterator.py",  # CUDA jiterator
            "test_numba_integration.py",  # CUDA/Numba
            "test_kernel_launch_checks.py",  # CUDA kernel launch
        ],
    ),
    SkipGroup(
        reason=(
            "Backend-specific tests locked to NCCL, Gloo, or UCC — test "
            "backend-specific wrapper behavior, protocols, or explicitly "
            "exit if the backend is unavailable"
        ),
        tests=[
            "distributed/test_c10d_ucc.py",
            "distributed/test_c10d_gloo.py",
            "distributed/test_c10d_spawn_gloo.py",
            "distributed/test_c10d_spawn_ucc.py",
            "distributed/test_ce_colls.py",
            "distributed/test_nvshmem.py",
            "distributed/test_nvshmem_triton.py",
            "distributed/test_pg_wrapper.py",
            "distributed/test_shmem_triton.py",
            # Have a PrivateUse1 Variant of this file
            # test/distributed/test_c10d_ops_privateuse1.py
            "distributed/test_c10d_ops_nccl.py",
        ],
    ),
    SkipGroup(
        reason=(
            "Inductor+Triton distributed tests — require Triton code generation "
            "and GPU-specific inductor passes (HAS_GPU, run_and_get_triton_code)"
        ),
        tests=[
            "distributed/test_inductor_collectives.py",
            "distributed/test_overlap_bucketing_unit.py",
            "distributed/test_aten_comm_compute_reordering.py",
            "distributed/test_compute_comm_reordering.py",
        ],
    ),
    SkipGroup(
        reason=(
            "Legacy nn.DataParallel with hardcoded CUDA device indices — "
            "deprecated in favor of DDP/FSDP, not worth making device-agnostic"
        ),
        tests=[
            "distributed/test_data_parallel.py",
        ],
    ),
    SkipGroup(
        reason="CPU-only infrastructure — launchers, elastic, RPC, tutorials",
        tests=[
            "distributed/launcher",
            "distributed/_tools",
            "distributed/local_tensor_tutorial_examples",
            "distributed/elastic",
            "distributed/bin",
            "distributed/rpc",
            # torch.futures.Future (RPC-like) — not supported on Neuron
            "test_futures.py",
            "distributed/test_distributed_spawn.py",
            "nn/attention/test_open_registry.py",
        ],
    ),
    SkipGroup(
        reason=(
            "CPU-only tests — fake/mock process groups, base modules with no "
            "test classes, pure utilities with no device involvement"
        ),
        tests=[
            "distributed/_pycute",
            "distributed/nn/jit",
            "distributed/optim/test_named_optimizer.py",
            "distributed/optim/test_apply_optimizer_in_backward.py",
            "distributed/test_local_tensor.py",
            "distributed/test_c10d_spawn.py",
            "distributed/test_c10d_pypg.py",
            "distributed/test_processgroup_jit.py",
            "distributed/argparse_util_test.py",
            "distributed/_test_template.py",
            "distributed/test_launcher.py",
            "distributed/test_run.py",
            "distributed/test_store.py",
        ],
    ),
    SkipGroup(
        reason=(
            "Requires optional dependencies not installed (aiohttp, tabulate) "
            "and tests only CPU debug HTTP server — no device involvement"
        ),
        tests=[
            "distributed/test_debug.py",
        ],
    ),
    SkipGroup(
        reason=(
            "MultiThreadedTestCase spawns threads sharing one device — "
            "incompatible with Neuron's one-device-per-process runtime model"
        ),
        tests=[
            "distributed/test_multi_threaded_pg.py",
        ],
    ),
    SkipGroup(
        reason=("Legacy ShardedTensor API deprecated by PyTorch in favor of DTensor"),
        tests=[
            "distributed/_shard",
        ],
    ),
    # =========================================================================
    # Root-level directories — not applicable to PrivateUse1 accelerators
    # =========================================================================
    SkipGroup(
        reason=(
            "TorchInductor compiler backend — generates Triton/CUDA/CPU"
            " kernels, not applicable to Neuron"
        ),
        tests=[
            "inductor",
            "inductor_expected_failures",
            "inductor_skips",
        ],
    ),
    SkipGroup(
        reason="C++ test sources — require CMake build and separate toolchain, not Python tests",
        tests=[
            "cpp",
            "cpp_api_parity",
            "cpp_extensions",
            "custom_backend",
            "test_cpp_api_parity.py",
            "test_cpp_extensions_aot.py",
            "test_cpp_extensions_jit.py",
            "test_cpp_extensions_mtia_backend.py",
            "test_cpp_extensions_stream_and_event.py",
        ],
    ),
    SkipGroup(
        reason=(
            "TorchScript/JIT tests — legacy JIT compiler, not used by PrivateUse1 eager backends"
        ),
        tests=[
            "jit",
            "jit_hooks",
            "lazy",
            "test_jit.py",
            "test_jit_autocast.py",
            "test_jit_fuser.py",
            "test_jit_fuser_legacy.py",
            "test_jit_fuser_te.py",
            "test_jit_legacy.py",
            "test_jit_profiling.py",
            "test_jit_simple.py",
            "test_jit_string.py",
            "test_jit_disabled.py",
            "test_jit_llga_fuser.py",  # Intel MKLDNN JIT fuser
            "test_ops_jit.py",  # JIT-specific op tests
            "test_static_runtime.py",  # TorchScript static runtime
            "test_tensorexpr.py",  # NNC/TensorExpr JIT compiler
            "test_tensorexpr_pybind.py",
            "test_dynamic_shapes.py",  # JIT dynamic shapes
        ],
    ),
    SkipGroup(
        reason=(
            "Mobile/edge deployment tests — targets iOS/Android"
            " lite interpreter, not server accelerators"
        ),
        tests=[
            "mobile",
            "package",
            "test_bundled_images.py",
            "test_bundled_inputs.py",
            "test_metal.py",  # Apple Metal GPU
            "test_vulkan.py",  # Vulkan GPU
            "test_nnapi.py",  # Android NNAPI
            "test_mobile_optimizer.py",
            "test_set_default_mobile_cpu_allocator.py",
            "test_show_pickle.py",  # TorchScript model pickle inspection
            "test_package.py",  # torch.package deployment
        ],
    ),
    SkipGroup(
        reason="ONNX export tests — tests graph export format, not device execution",
        tests=[
            "onnx",
        ],
    ),
    SkipGroup(
        reason=(
            "Quantization tests — requires FBGEMM/QNNPACK/CUDA"
            " quantized kernels not available on Neuron"
        ),
        tests=[
            "quantization",
            "test_quantization.py",
            "ao",
            "test_ao_sparsity.py",
            "test_sparse_csr.py",
            "test_sparse_semi_structured.py",
        ],
    ),
    SkipGroup(
        reason=(
            "Strobelight — Meta-internal profiling CLI tool,"
            " not available outside Meta infrastructure"
        ),
        tests=[
            "strobelight",
        ],
    ),
    SkipGroup(
        reason=(
            "Other hardware-specific backends (Intel XPU/MKL, Apple MPS, XNNPACK)"
            " — not applicable to Neuron"
        ),
        tests=[
            "xpu",
            "backends",  # backends/xeon — Intel Xeon CPU launcher
            "test_mps.py",
            "test_mkl_verbose.py",
            "test_mkldnn_fusion.py",
            "test_mkldnn_verbose.py",
            "test_xnnpack_integration.py",
            "test_xpu.py",
            "test_xpu_expandable_segments.py",
            "test_itt.py",  # Intel Instrumentation and Tracing
            "test_mkldnn.py",  # Intel MKLDNN ops
        ],
    ),
    SkipGroup(
        reason=(
            "torch.export tests — graph export infrastructure,"
            " mostly runs on CPU and does not exercise device execution"
        ),
        tests=[
            "export",
            # Export to core-aten — ExportedProgram flow not onboarded
            "test_model_exports_to_core_aten.py",
        ],
    ),
    SkipGroup(
        reason=(
            "torch.fx tests — graph IR transformations"
            " (CSE, DCE, subgraph rewriting), mostly runs on CPU"
            " and does not exercise device execution"
        ),
        tests=[
            "fx",
            "test_fx_experimental.py",
            "test_fx_graph_print.py",
            "test_fx_passes.py",
            "test_fx_reinplace_pass.py",
        ],
    ),
    SkipGroup(
        reason="Static analysis, metadata, and support directories — no device execution involved",
        tests=[
            "typing",
            "error_messages",
            "expect",
            "test_img",
            "scripts",
            "benchmark_utils",
            "forward_backward_compatibility",
            "compiled_autograd_skips",
            "dynamo_expected_failures",
            "dynamo_skips",
            "test_logging.py",  # logging infrastructure, no device tests
            "test_pytree.py",  # pure Python tree utilities
            "test_weak.py",  # weak reference utilities
            "test_overrides.py",  # __torch_function__ protocol, CPU-only
            "test_license.py",  # license file checks
            "test_namedtuple_return_api.py",  # YAML schema validation
            "test_ci_sanity_check_fail.py",  # CI infrastructure
            "test_import_stats.py",  # CI infrastructure
            "test_determination.py",  # CI test determination
            "test_pruning_op.py",  # hypothesis-based, no device
            "python_native",  # DSL native ops registry, CPU-only
            "run_test.py",  # test runner script, not a test file
        ],
    ),
    SkipGroup(
        reason=(
            "optim/ directory contains test_optim.py which collides with the"
            " root-level test_optim.py on module name 'test_optim', causing"
            " both to fail collection. Root-level test_optim.py is the"
            " canonical file used by the test suite."
        ),
        tests=[
            "optim",
        ],
    ),
    SkipGroup(
        reason="CPython 3.13+ APIs (suppress_immortalization, requires_working_threading)",
        tests=[
            "dynamo/cpython",
        ],
    ),
    SkipGroup(
        reason=(
            "CUDA-only NVIDIA Flash Attention kernel tests — fa3 depends on"
            " flash_attn_interface (NVIDIA FA3) and fa4 depends on"
            " flash_attn.cute.interface (NVIDIA FA4), neither available on Neuron"
        ),
        tests=[
            "nn/attention/test_fa3.py",
            "nn/attention/test_fa4.py",
        ],
    ),
    SkipGroup(
        reason=(
            "Device-specific test classes and methods names"
            " (e.g. TestAutocastCPU/GPU/MPS), no privateuse1 variant"
        ),
        tests=[
            "test_autocast.py",
        ],
    ),
    SkipGroup(
        reason=(
            "JitTestCase classes default to CPU; device-type classes are GPU-only"
            " (e.g. cublaslt context, cudaLaunchKernel profiler events, triton"
            " kernels) — not portable to Neuron/privateuse1"
        ),
        tests=[
            "test_fx.py",
        ],
    ),
    # =========================================================================
    # PENDING ONBOARDING and UNSUPPORTED FEATURES
    # =========================================================================
    SkipGroup(
        reason="TODO: Directories/files pending onboarding, ensure to remove when onboarded.",
        tests=[
            "distributed/test_composability.py",
        ],
    ),
    SkipGroup(
        reason="Not supported in Beta — to be onboarded in a future release",
        tests=[
            "complex_tensor",
            "test_complex.py",
        ],
    ),
    SkipGroup(
        reason="Not supported in Beta — to be onboarded in a future release",
        tests=[
            "functorch",
            "custom_operator",
            "distributions",
            "higher_order_ops",
            "distributed/algorithms/quantization",
            # an experimental next-gen distributed API (the dist2 namespace)
            "distributed/test_dist2.py",
            # Masked / named / nested tensors
            "test_masked.py",
            "test_maskedtensor.py",
            "test_namedtensor.py",
            "test_nestedtensor.py",
            # DataPipe API — iterable dataset infrastructure
            "test_datapipe.py",
            # vmap / functorch-related per-sample gradient computation
            "test_expanded_weights.py",
            # Functorch functionalization pass
            "test_functionalization.py",
            "test_functionalization_of_rng_ops.py",
            # Legacy vmap — replaced by functorch; functorch itself not onboarded
            "test_legacy_vmap.py",
            # Sparse tensors not supported on Neuron
            "test_sparse.py",
            # Variable-length attention
            "test_varlen_attention.py",
            # Unbacked symints (dynamic shapes) — not supported on Neuron
            "test_ops_unbacked.py",
            # FFT / spectral ops not supported on Neuron
            "test_spectral_ops.py",
            # Segment reductions (torch._segment_reduce) not supported
            "test_segment_reductions.py",
            # DLPack interop — requires device-specific DLPack support
            "test_dlpack.py",
        ],
    ),
    SkipGroup(
        reason="NumPy compatibility tests — not relevant to Neuron device execution",
        tests=[
            "torch_np",
        ],
    ),
    SkipGroup(
        reason=(
            "Tests PyTorch's PrivateUse1 registration API itself, not Neuron."
            " Each file registers a toy backend at import time which conflicts"
            " with Neuron — only one PrivateUse1 backend allowed per process."
        ),
        tests=[
            "test_privateuseone_python_backend.py",  # registers 'npy' backend
            "test_rename_privateuse1_to_existing_device.py",  # renames to 'maia' backend
        ],
    ),
    SkipGroup(
        reason=("Tests majorly use backend=fake not relevant to Neuron"),
        tests=["distributed/tensor/debug/test_debug_mode.py"],
    ),
]
