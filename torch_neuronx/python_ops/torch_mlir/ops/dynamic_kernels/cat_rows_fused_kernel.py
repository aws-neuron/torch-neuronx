"""Fused dim-0 concat kernel: one NEFF moves K single-row sources into one result.

Each source is one contiguous HBM row of H elements, so the whole source moves in one
HBM->HBM DMA. There is no SBUF stage and no free-dim segmentation. Those exist in
``strided_copy_kernel`` because a DMA permits only ONE dynamic (``scalar_offset``)
operand, and a row-walking copy needs a dynamic offset at both ends. With one row per
source both ends are static, so K DMAs do the whole concat.

* ``nki.jit`` rejects ``*args``: the lowering binds arguments by keyword. So the
  K-operand signature is generated per K, and the kernel goes into this module's
  globals with its source in ``linecache`` (NKI resolves a kernel by module-qualified
  name and reads its source). The generated source carries its own imports, because
  this module never calls the nki API itself.
* LNC sharding splits the free dim, like ``cast_kernel``. Every core then moves an
  equal share of each source instead of the whole thing, which halves the HBM traffic
  at LNC=2. The composite path cannot do this.
"""

import linecache

_KERNEL_CACHE: dict[int, object] = {}


def _kernel_source(k: int, name: str) -> str:
    params = ", ".join(f"s{i}" for i in range(k))
    body = [
        # The kernel executes in this module's globals, so it imports what it needs.
        "import nki",
        "import nki.isa as nisa",
        "import nki.language as nl",
        "",
        "@nki.jit()",
        f"def {name}({params}):",
        '    """K single-row [1, H] sources -> one [K, H] result on HBM."""',
        "    h = s0.shape[1]",
        f"    out = nl.ndarray(({k}, h), s0.dtype, nl.shared_hbm)",
        "    num_progs = nl.num_programs(0)",
        "    prog_id = nl.program_id(0)",
        "    if h % num_progs == 0:",
        "        span = h // num_progs",
        "        base = prog_id * span",
        "    else:",
        "        span = h",
        "        base = 0",
    ]
    for i in range(k):
        body += [
            "    nisa.dma_copy(",
            f"        dst=out.ap(pattern=[[h, 1], [1, span]], offset={i} * h + base),",
            f"        src=s{i}.ap(pattern=[[h, 1], [1, span]], offset=base),",
            "    )",
        ]
    body.append("    return out")
    return "\n".join(body) + "\n"


def cat_rows_fused_kernel(k: int):
    """The fused kernel for arity ``k``, built once per k."""
    kern = _KERNEL_CACHE.get(k)
    if kern is None:
        name = f"cat_rows_fused_kernel_k{k}"
        src = _kernel_source(k, name)
        filename = f"<{__name__}.{name}>"
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        exec(compile(src, filename, "exec"), globals())
        kern = globals()[name]
        _KERNEL_CACHE[k] = kern
    return kern
