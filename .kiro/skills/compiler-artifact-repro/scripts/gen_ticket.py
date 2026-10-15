"""Generate a ready-to-file ticket description (`ticket.md`) inside the repro
artifact folder, so an engineer only has to review it and cut the ticket to the
right team.

The technical body is filled from the run (versions, failing tests, the inline
offending HLO, the evidence, the repro commands). The ROUTING block is left as
fill-in placeholders on purpose: which team / tracker / upload location the
ticket goes to is team-internal and must not be hard-coded in this public repo
(the engineer fills it at filing time).

Call it from your repro script:

    import gen_ticket
    gen_ticket.write_ticket(
        artdir,
        title="stablehlo.power(int64,1) miscompiles for some 2D shapes — neuronx-cc <ver>",
        summary="power(x,1) must be identity but corrupts the 64-bit result for "
                "tile shapes n=12/15/16/19/20 …",
        versions={"neuronx-cc": "<ver>", "driver": "<ver>", "runtime-lib": "<ver>",
                  "torch-neuronx": "<commit>", "target": "trn2 --lnc 2"},
        ci_url="https://github.com/.../actions/runs/<id>/job/<id>",
        failing_tests=["tests/.../test_foo.py::TestBar::test_baz"],
        offending_hlo="<artdir>/compiler_dump/neuronx-cc/<hash>/module.mlir",
        compiler_flags="neuronx-cc compile module.mlir --framework XLA --target trn2 …",
        evidence=["max abs diff = exactly 1.0 (off-by-one, not fp noise)",
                  "standalone HLO reproduces (compile_graph + execute_compiled_graph)",
                  "deterministic across 3 runs"],
        repro_cmds=["python -m pytest <nodeid> -o addopts=''",
                    "python validate_hlo.py <module.mlir> --input in_x.pt --ref out_cpu_ref.pt …"],
        ask="Please investigate the int64 stablehlo.power lowering …",
    )

Or from the CLI with a JSON spec: `python gen_ticket.py --artdir <dir> --spec spec.json`.
"""

import json
import os


def _artifact_list(artdir):
    """A markdown bullet list of the files present in the repro folder."""
    lines = []
    for root, _dirs, files in os.walk(artdir):
        for f in sorted(files):
            if f == "ticket.md":
                continue
            rel = os.path.relpath(os.path.join(root, f), artdir)
            lines.append(f"- `{rel}`")
    return "\n".join(lines) or "- (none yet)"


def _offending_hlo_block(path):
    if not path or not os.path.exists(path):
        return "_(attach the offending `module.mlir` — not found at generation time)_"
    with open(path) as fh:
        text = fh.read().strip()
    return f"```mlir\n{text}\n```"


def write_ticket(
    artdir,
    *,
    title,
    summary,
    versions=None,
    ci_url=None,
    failing_tests=None,
    compiler_flags=None,
    case=None,
    offending_hlo=None,
    evidence=None,
    repro_cmds=None,
    ask=None,
    category=None,  # Error | Compile-time | Perf | Accuracy
    regression=None,  # "yes" / "no" / "unknown"
    last_good_compiler=None,  # last known-good neuronx-cc version/commit (ask the user;
    # source: the team's Weekly Ops Health dashboard). Lets the
    # compiler team bisect the failing<-passing range.
    customer_impact=None,
    environment=None,  # pasted output of the apt/pip/product_name commands (reference.md §D)
    extra=None,
):
    """Render ARTDIR/ticket.md. Returns its path. The compiler-intake fields
    (category / regression / customer_impact / environment) match reference.md §D —
    fill them so the ticket is not bounced for missing info. The neuronx-cc version
    goes in `versions` (we RCA to a specific compiler version; no version-set needed)."""
    versions = versions or {}
    failing_tests = failing_tests or []
    evidence = evidence or []
    repro_cmds = repro_cmds or []
    extra = extra or {}

    nl = "\n"
    env_rows = nl.join(f"| {k} | {v} |" for k, v in versions.items()) or "| | |"
    tests_md = nl.join(f"- `{t}`" for t in failing_tests) or "- (list failing node ids)"
    evidence_md = (
        nl.join(f"- {e}" for e in evidence) or "- (why this is the compiler, not the framework)"
    )
    repro_md = (
        nl.join("```" + nl + c + nl + "```" for c in repro_cmds)
        or "```" + nl + "(commands)" + nl + "```"
    )
    extra_md = "".join(nl + "## " + h + nl + nl + b + nl for h, b in extra.items())
    flags_md = (
        nl
        + "Compiler flags (from `compiler.txt`):"
        + nl
        + "```"
        + nl
        + (compiler_flags or "")
        + nl
        + "```"
        + nl
        if compiler_flags
        else ""
    )
    case_md = (nl + "## The case" + nl + case + nl) if case else ""
    ci_md = (nl + "CI: " + ci_url + nl) if ci_url else ""
    ask_md = ask or "Please root-cause the miscompile in this HLO at the version above."
    cat_line = category or "<Error | Compile-time | Perf | Accuracy>"
    reg_line = regression or "<yes: last-good compiler commit/version | no | unknown>"
    lastgood_line = (
        last_good_compiler
        or "<last known-good neuronx-cc version/commit — from the Weekly Ops Health dashboard>"
    )
    env_block = environment or (
        "# paste: apt list --installed | grep neuron  (or yum ...);\n"
        "# pip3 list | grep neuron;  cat /sys/devices/virtual/dmi/id/product_name"
    )
    impact_line = customer_impact or "<what this blocks + priority>"
    ci_intake = f"""## Category
{cat_line}  |  Regression: {reg_line}

## Last known-good compiler (for bisection)
{lastgood_line}

## Customer impact
{impact_line}

## Environment
```
{env_block}
```
"""

    body = f"""<!-- ROUTING — fill in before filing; kept out of the repo on purpose.
  Tracker / CTI (category / type / item) : <fill>
  Resolver group / team                  : <fill>   (e.g. the compiler or kernel team)
  Severity                               : <fill>   (SEV_3 for a reproducible correctness bug)
  Bug category                           : <fill>   (Error | Compile-time | Perf | Accuracy)
  Regression?                            : <fill>   (if yes: last-good compiler commit/version)
  Artifact upload location               : <fill>   (team intake bucket for CI/hardware runs —
                                                     small files only: HLO/NEFF/NKI/inputs)
-->

# {title}

## Summary
{summary}

## Environment (reproduced, matches failing CI)
| | |
|---|---|
{env_rows}
{ci_md}
Failing tests:
{tests_md}
{flags_md}{case_md}
## Offending HLO (the exact module handed to neuronx-cc)
{_offending_hlo_block(offending_hlo)}

## Evidence it is the compiler
{evidence_md}

## Artifacts (in this folder)
{_artifact_list(artdir)}

{ci_intake}
## How to reproduce
{repro_md}
{extra_md}
## Ask
{ask_md}
"""
    out = os.path.join(artdir, "ticket.md")
    with open(out, "w") as f:
        f.write(body)
    return out


def _main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--artdir", required=True)
    ap.add_argument("--spec", required=True, help="JSON file with the write_ticket kwargs")
    args = ap.parse_args()
    with open(args.spec) as fh:
        spec = json.load(fh)
    path = write_ticket(args.artdir, **spec)
    print(f"wrote {path}")


if __name__ == "__main__":
    _main()
