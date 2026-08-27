"""Compiler-artifact dump harness for torch-neuronx.

Import this module FIRST (before torch / torch_neuronx) to route every
neuronx-cc compilation's artifacts (fx graphs, raw + stablehlo MLIR, and the
neuronx-cc workdir with module.mlir = the HLO and module.neff = the NEFF) into
one directory, and to keep them after the process exits.

    import artifact_dump                       # sets env, MUST be before torch_neuronx
    artdir = artifact_dump.ARTDIR
    import torch, torch_neuronx
    ...run the failing op...
    artifact_dump.save_io(artdir, inputs={...}, outputs={...})

Env knobs (read at import):
  ARTDIR  - output directory. Default ./neuron_repro_artifacts. Point it at a
            path you can retrieve from the host afterwards (e.g. shared storage).

Why these env vars (verified against torch_neuronx/kernels/cache_utils.py):
  TORCH_NEURONX_DEBUG_DIR                    -> base dir for all artifacts
  TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1 -> do not delete after run
  TORCH_NEURONX_NEFF_CACHE_DIR               -> isolate this run's NEFF cache
The neuronx-cc `--dump=` CLI flag is REJECTED by recent compiler builds; do not
use it. NEURONX_DUMP_TO is not consumed by this repo.
"""

import os

ARTDIR = os.environ.get("ARTDIR", os.path.abspath("neuron_repro_artifacts"))
DUMP = os.path.join(ARTDIR, "compiler_dump")
NKI = os.path.join(ARTDIR, "nki_artifacts")
os.makedirs(DUMP, exist_ok=True)
os.makedirs(NKI, exist_ok=True)

os.environ["TORCH_NEURONX_DEBUG_DIR"] = DUMP
os.environ["TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS"] = "1"
os.environ.setdefault("TORCH_NEURONX_NEFF_CACHE_DIR", os.path.join(DUMP, "neff_cache"))
# Collect NKI artifacts into one dir. The compiler intake REQUIRES these when the HLO
# references external NKI files (an `AwsNeuronCustomNativeKernel` whose backend_config
# points at a `klir_binary` under an nki-intermediate-cache). Setting the cache URL
# makes NKI dump all artifacts here so they can be uploaded alongside the HLO.
os.environ.setdefault("NKI_COMPILE_CACHE_URL", NKI)


def save_io(artdir, inputs=None, outputs=None):
    """Persist input/output tensors next to the compiler dump (torch.save).

    Save enough to re-run: ALL inputs, and for outputs BOTH the (wrong) Neuron
    result AND the CPU reference — the compiler team needs the reference to see
    what "wrong" means and validate_hlo.py diffs against it. e.g.:
        save_io(artdir,
                inputs={"grad_output": g_cpu, "indices": idx_cpu},
                outputs={"neuron": neuron_out.cpu(), "cpu_ref": cpu_ref})
    """
    import torch

    for name, val in (inputs or {}).items():
        torch.save(val, os.path.join(artdir, f"in_{name}.pt"))
    for name, val in (outputs or {}).items():
        torch.save(val, os.path.join(artdir, f"out_{name}.pt"))


def list_modules(artdir=None):
    """Return [(path, mlir_text)] for every dumped neuronx-cc module.mlir."""
    import glob

    ncc = os.path.join(artdir or ARTDIR, "compiler_dump", "neuronx-cc")
    out = []
    for f in glob.glob(os.path.join(ncc, "*", "module.mlir")):
        with open(f) as fh:
            out.append((f, fh.read()))
    return out
