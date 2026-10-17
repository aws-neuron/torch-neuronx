.. meta::
   :description: Environment variables used to configure torch-neuronx behavior
   :date-modified: 2026-01-26
   :keywords: neuron, pytorch, environment variables, configuration

Environment Variables
=====================

This document provides an overview of the different environment variables available to configure ``torch_neuronx`` behavior.

``neuron`` backend for ``torch.compile``
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 8 15 52

   * - Variable
     - Type
     - Default
     - Description
   * - ``TORCH_NEURONX_DEBUG_DIR``
     - str
     - ``/tmp/neuron_backend_<uuid>``
     - Directory where compilation artifacts (FX graphs, StableHLO IR, NEFF files) are saved.
   * - ``TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS``
     - bool
     - ``False``
     - When ``1``, keeps compilation artifacts after execution. Useful for debugging via IR inspection.
   * - ``TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION``
     - bool
     - ``True``
     - When ``1``, disables CPU fallback on Neuron execution failure. Fails fast instead.
   * - ``TORCH_NEURONX_RETAIN_DEVICE_MODE``
     - bool
     - ``False``
     - When ``1``, outputs retain original device placement instead of Neuron device.
   * - ``TORCH_NEURONX_DYNAMO_TENSOR_DUMPS``
     - str
     - ``""``
     - Configures input/output tensor dumping during execution. Format: semicolon-delimited ``key=value`` pairs. Keys: ``inputs``, ``outputs``. Values: ``all`` (dump for all ranks) or comma-separated rank integers. Example: ``inputs=all;outputs=0,1``.

Artifact Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both ``torch.compile`` and eager execution use the same unified job-based artifact hierarchy.
The working directory used by all neuronx-cc calls is preserved under ``TORCH_NEURONX_DEBUG_DIR``::

    {TORCH_NEURONX_DEBUG_DIR}/
    ├── torch_neuronx/
    │   └── rank{rank}/
    │       └── {job_id}/
    │           ├── fx_graphs/{model}_{ts}.fx.txt
    │           ├── transformed_fx/{model}_{ts}.transformed.fx.txt
    │           ├── raw/{model}_{ts}.torch_raw.mlir
    │           ├── stablehlo/{model}_{ts}.stablehlo.mlir
    │           ├── torch_mlir_error/{model}_{ts}.mlir
    │           ├── inputs/inp-{rank_3digit}-{count}.p
    │           └── outputs/out-{rank_3digit}-{count}.p
    └── neuronx-cc/
        └── {hash}/
            ├── log-neuron-cc.txt
            ├── module.mlir
            ├── module.neff
            ├── metadata.txt
            └── compiler_args.txt

Input/output dump files (``inp-*.p``, ``out-*.p``) are only written when ``TORCH_NEURONX_DYNAMO_TENSOR_DUMPS`` is set.
Each file is a pickled dict mapping ``"input0"``, ``"input1"``, ... (or ``"output0"``, ...) to numpy arrays.

When ``TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1``, all directories and their contents are preserved for debugging and analysis.

Compiler
--------

See `Neuron Compiler CLI Reference <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html>`_ for available flags.

General
-------

These variables affect eager execution (non-compiled operations).

.. list-table::
   :header-rows: 1
   :widths: 25 8 12 55

   * - Variable
     - Type
     - Default
     - Description
   * - ``NEURON_LAUNCH_BLOCKING``
     - bool
     - ``False``
     - When ``1``, forces synchronous execution. Useful for debugging.
   * - ``TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS``
     - bool
     - ``False``
     - When ``1``, only falls back to CPU for unimplemented ops, not execution errors.
   * - ``TORCH_NEURONX_ENABLE_STACK_TRACE``
     - bool
     - ``False``
     - When ``1``, captures Python stack traces per operation. Adds overhead.
   * - ``TORCH_NEURONX_DISABLE_NEFF_CACHE``
     - bool
     - ``False``
     - When ``1``, disables NEFF caching. Forces recompilation every time.
   * - ``TORCH_NEURONX_LOG_LEVEL``
     - int
     - ``0``
     - When ``3``, enable debug logs.
   * - ``TORCH_NEURONX_MODULE_AUTO_COMPILE``
     - bool
     - ``False``
     - When ``1``, enables bounded AutoCompile discovery for eager modules.
   * - ``TORCH_NEURONX_MODULE_AUTO_COMPILE_DEBUG``
     - bool
     - ``False``
     - When ``1``, prints AutoCompile diagnostics to stderr.

Module AutoCompile
~~~~~~~~~~~~~~~~~~

``Module.to``, ``Module.to_empty``, and ``Module.neuron`` seed bounded
module-tree discovery. A module that starts directly on Neuron needs a later
tree seed. Without that seed, the module stays on the exact eager path.

AutoCompile uses one static Dynamo signature with the Neuron backend. Eligible
leaf modules are compile regions. Parent modules and checkpoint owners stay
eager. Distributed lifecycle hooks also stay eager. An eligible leaf compiles
inside the active checkpoint or SAC mode. The compiled partition applies the
active SAC policy.

An inference candidate needs a real Neuron forward executor. A training
candidate needs real Neuron forward and backward executors.

A graph break, compiler fallback, unprofitable graph, guard miss,
checkpoint-kind change, or SAC-policy change makes the candidate terminal
eager. AutoCompile returns the current result once. It does not replay the
module call.

A retained or higher-order eager graph makes the candidate terminal eager.
After promotion, higher-order autograd uses the captured differentiable eager
backward. Normal calls remain compiled.

Neuron Runtime
--------------

- See `Neuron Runtime Configurable Parameters <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-configurable-parameters.html>`_ for all runtime variables.
- See `Logical NeuronCore Configuration <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html>`_ for LNC details.

Distributed
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 8 12 55

   * - Variable
     - Type
     - Default
     - Description
   * - ``NEURON_USE_SPLIT_ALLGATHER``
     - bool
     - ``True``
     - Enables split-based allgather for better memory efficiency.
   * - ``NEURON_USE_SPLIT_REDUCE_SCATTER``
     - bool
     - ``True``
     - Enables split-based reduce_scatter implementation.
   * - ``COLLECTIVE_BUCKETSIZE_IN_MB``
     - int
     - ``512``
     - Maximum bucket size (MB) for collective operations.
   * - ``COLLECTIVE_MIN_BUCKETS``
     - int
     - ``0``
     - Minimum buckets for collectives. When > 1, forces splitting small tensors.

Metrics
-------

.. list-table::
   :header-rows: 1
   :widths: 25 8 12 55

   * - Variable
     - Type
     - Default
     - Description
   * - ``TORCH_NEURONX_ENABLED_METRIC_TABLES``
     - str
     - ``""``
     - Comma-separated metric tables to enable for CSV export.
   * - ``TORCH_NEURONX_METRICS_DIR``
     - str
     - ``/tmp``
     - Directory for metric CSV files.
