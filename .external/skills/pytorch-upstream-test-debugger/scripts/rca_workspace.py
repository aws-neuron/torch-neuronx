#!/usr/bin/env python3
"""Scaffold a local RCA workspace for one selected failure category.

Happy path (run from the torch-neuronx repository root):
    python3 .external/skills/pytorch-upstream-test-debugger/scripts/rca_workspace.py \
        category-tests.txt agent_space/<category-slug> \
        --category index_copy_backward

Failure example: a missing, empty, or unsafe tests file exits 2 and writes
nothing. Generated test and reproduction files require PyTorch and Neuron
hardware only when executed. This generator uses the Python standard library.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path, PurePosixPath

from redact import redact_text
from report_signatures import MAX_IDENTIFIER_CHARS, MAX_NODEID_CHARS, validate_nodeid
from safe_output import check_output_names, install_files, open_output_dir

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DOTTED_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_PATH_PART_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")
_FORBIDDEN_TEXT_RE = re.compile(r"['\"\x00-\x1f\x7f]")
_OP_RE = re.compile(
    r"^test_(?:fn_grad|fn_gradgrad|inplace_grad|inplace_gradgrad)_"
    r"(?P<op>[A-Za-z_][A-Za-z0-9_]*)_neuron_[a-z0-9]+$"
)


class UnsafeInputError(ValueError):
    """Raised when a node ID can create unsafe generated content."""


def _validate_plain_text(label: str, text: str, *, max_chars: int = MAX_IDENTIFIER_CHARS) -> None:
    if not text:
        raise UnsafeInputError(f"{label} is empty")
    if len(text) > max_chars:
        raise UnsafeInputError(f"{label} exceeds {max_chars} characters")
    if _FORBIDDEN_TEXT_RE.search(text):
        raise UnsafeInputError(f"{label} contains a quote or control character")


def _validate_identifier(label: str, text: str) -> None:
    _validate_plain_text(label, text)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise UnsafeInputError(f"{label} is not a valid identifier: {text!r}")


def _validate_dotted_identifier(label: str, text: str) -> None:
    _validate_plain_text(label, text)
    if not _DOTTED_IDENTIFIER_RE.fullmatch(text):
        raise UnsafeInputError(f"{label} is not a valid dotted identifier: {text!r}")


def _validate_test_path(text: str) -> str:
    _validate_plain_text("test file", text, max_chars=MAX_NODEID_CHARS)
    if "\\" in text or "//" in text:
        raise UnsafeInputError("test file must use a normalized relative POSIX path")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise UnsafeInputError("test file must be a safe relative path")
    if path.suffix != ".py" or any(not _PATH_PART_RE.fullmatch(part) for part in path.parts):
        raise UnsafeInputError("test file must be a safe relative Python test path")
    return text


def op_from_method(method_name: str) -> str | None:
    match = _OP_RE.match(method_name)
    return match.group("op") if match else None


def torch_op_from_method_op(op: str) -> str:
    """Translate common OpInfo method encoding into a torch attribute path."""
    _validate_identifier("operation", op)
    for namespace in ("linalg", "fft", "special"):
        prefix = f"{namespace}_"
        if op.startswith(prefix):
            result = f"{namespace}.{op[len(prefix):]}"
            _validate_dotted_identifier("torch operation", result)
            return result
    _validate_dotted_identifier("torch operation", op)
    return op


def _without_parameter_suffix(selector: str) -> str:
    """Remove one balanced trailing pytest parameter suffix."""
    if not selector.endswith("]"):
        if "[" in selector or "]" in selector:
            raise UnsafeInputError("node ID has malformed parameters")
        return selector

    depth = 0
    opening = None
    for index in range(len(selector) - 1, -1, -1):
        char = selector[index]
        if char == "]":
            depth += 1
        elif char == "[":
            depth -= 1
            if depth < 0:
                raise UnsafeInputError("node ID has malformed parameters")
            if depth == 0:
                opening = index
                break
    if opening is None or opening == len(selector) - 1:
        raise UnsafeInputError("node ID has malformed parameters")
    core = selector[:opening]
    if "[" in core or "]" in core:
        raise UnsafeInputError("node ID has malformed parameters")
    return core


def parse_nodeid(nodeid: str) -> tuple[str, str, str]:
    """Return validated file, class, and method fields from a pytest node ID."""
    _validate_plain_text("node ID", nodeid, max_chars=MAX_NODEID_CHARS)
    test_file_text, separator, selector = nodeid.partition("::")
    if not separator or not selector:
        raise UnsafeInputError("node ID must contain a test file and method")

    test_file = _validate_test_path(test_file_text)
    selector = _without_parameter_suffix(selector)
    parts = selector.split("::")
    if any(not part for part in parts):
        raise UnsafeInputError("node ID contains an empty class or method")
    method_name = parts[-1]
    _validate_identifier("test method", method_name)

    class_parts = parts[:-1]
    for class_part in class_parts:
        _validate_identifier("test class", class_part)
    test_class = ".".join(class_parts) if class_parts else "ModuleLevel"
    _validate_dotted_identifier("test class", test_class)
    return test_file, test_class, method_name


UNIT_TEST_TEMPLATE = '''"""Minimal RCA unit-test seeds for one selected failure category."""
from __future__ import annotations

import pytest
import torch


CATEGORY = {category}
TEST_CLASS = {test_class}
DEVICE = "neuron"


def _randn(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype, device=DEVICE, requires_grad=True)


{test_functions}
'''


SEED_FN = """\
def test_{safe_op}_grad():
    {description}
    # TODO: copy the failing OpInfo sample shape, dtype, args, and kwargs.
    x = _randn(4, 4)
    y = torch.{op}(x)  # Adjust arguments when the operation requires them.
    y.sum().backward()
    assert x.grad is not None

"""


REPRO_STUB = '''"""Minimal off-framework reproduction for one failure category."""
from __future__ import annotations

import traceback

import torch


CATEGORY = {category}
TEST_CLASS = {test_class}
DEVICE = "neuron"


def case(name, fn):
    try:
        fn()
        print(f"OK    {{name}}")
    except Exception:
        print(f"FAIL  {{name}}")
        traceback.print_exc()


def main():
    # TODO: add one case per failing shape/dtype combination.
    print("TODO: fill in cases that exercise the selected signature.")


if __name__ == "__main__":
    main()
'''


PR_BODY_TEMPLATE = """\
AI-assisted content disclosure: The quoted pull request draft below was generated with <AI \
tool name>.

## Human commentary

<!-- REQUIRED: Explain why the quoted change is relevant. The agent must not fill this section. -->

## Human confirmation

- [ ] I read and understand every changed file.
- [ ] I take responsibility for these changes.

## AI-generated pull request body

> ## Problem
>
> <Tests and the shared error signature. State that xfails are tracked failures when applicable.>
>
> ## Related Issue
>
> Fixes #<public issue number>
>
> ## Root cause
>
> <Name the product source symbol and explain the incorrect behavior.>
>
> ## Fix
>
> <Describe the smallest source change and why it fixes this defect.>
>
> ## Verification
>
> Baseline:
>
> ```bash
> <exact baseline command>
> ```
>
> Current:
>
> ```bash
> <exact verification command>
> ```
>
> Results: <fixed count> fixed, <same-signature count> same-signature failures, \
<different-signature count> different-signature failures, <regression count> regressions.
>
> ## Follow-ups
>
> <Separate bugs only. Omit this section when none exist.>
>
> ---
>
> <copy and complete the repository pull request template, or the bundled fallback>
"""


def slugify(text: str) -> str:
    _validate_plain_text("category", text)
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip()).strip("_")
    if slug and slug[0].isdigit():
        slug = f"category_{slug}"
    _validate_identifier("category slug", slug or "unnamed")
    return slug or "unnamed"


def _error(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 2


_SIGNATURE_HEADING_RE = re.compile(r"## (?P<signature>.+?)(?:  \((?P<count>[0-9]+) tests\))?\Z")
_HEADER_RE = re.compile(
    r"# (?:Selected (?P<selected>[0-9]+) tests from (?P<failures>[0-9]+) failures|"
    r"Pinned: nodeids=(?P<nodeids>[0-9]+) error_regexes=(?P<regexes>[0-9]+)|"
    r"Signatures matched: (?P<groups>[0-9]+))\Z"
)


def _sanitize_signatures(text: str) -> str:
    """Parse the selector sidecar and render only validated, redacted fields."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        header = _HEADER_RE.fullmatch(line)
        if header:
            lines.append(line)
            continue
        heading = _SIGNATURE_HEADING_RE.fullmatch(line)
        if heading:
            signature = redact_text(heading.group("signature"))
            count = heading.group("count")
            suffix = f"  ({int(count)} tests)" if count is not None else ""
            lines.append(f"## {signature}{suffix}")
            continue
        if line.startswith("example:"):
            nodeid = validate_nodeid(
                line.removeprefix("example:").strip(), label="signature example"
            )
            lines.append(f"    example: {redact_text(nodeid)}")
            continue
        if line.startswith("message:"):
            message = line.removeprefix("message:").strip()
            if len(message) > 500 or any(ord(char) < 32 or ord(char) == 127 for char in message):
                raise UnsafeInputError("signature message is invalid")
            lines.append(f"    message: {redact_text(message)}")
            continue
        raise UnsafeInputError("signature sidecar contains an unsupported line")
    return "\n".join(lines).rstrip() + "\n"


def _entry_info(fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _same_file_with_entry(source: Path, fd: int, name: str) -> bool:
    try:
        source_info = os.stat(source, follow_symlinks=False)
        destination_info = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except OSError:
        return False
    return (source_info.st_dev, source_info.st_ino) == (
        destination_info.st_dev,
        destination_info.st_ino,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tests_file", type=Path, help="one selected pytest nodeid per line")
    parser.add_argument("rca_root", type=Path, help="workspace directory to create or populate")
    parser.add_argument("--category", required=True, help="short failure-category name")
    parser.add_argument(
        "--signatures",
        type=Path,
        help="optional signature sidecar from select_category_tests.py",
    )
    parser.add_argument("--max-seeds", type=int, default=5, help="positive seed limit (default: 5)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate existing generated files; never overwrite notes.md",
    )
    args = parser.parse_args()

    if not args.tests_file.is_file():
        return _error(f"tests file does not exist: {args.tests_file.name}")
    if args.signatures is not None and not args.signatures.is_file():
        return _error(f"signature file does not exist: {args.signatures.name}")
    if args.max_seeds < 1:
        return _error("--max-seeds must be at least 1")

    try:
        tests_text = args.tests_file.read_text()
        nodeids = [line.strip() for line in tests_text.splitlines() if line.strip()]
        signatures_text = args.signatures.read_text() if args.signatures is not None else None
    except OSError as exc:
        detail = exc.strerror or "I/O error"
        return _error(f"cannot read input file: {detail}")
    if not nodeids:
        return _error(f"tests file contains no nodeids: {args.tests_file.name}")

    try:
        category_slug = slugify(args.category)
        parsed_nodeids = [parse_nodeid(nodeid) for nodeid in nodeids]
        if signatures_text is not None:
            signatures_text = _sanitize_signatures(signatures_text)
    except (UnsafeInputError, ValueError) as exc:
        return _error(f"unsafe input: {exc}")

    test_class = parsed_nodeids[0][1]
    workspace = args.rca_root

    seeded: list[str] = []
    used_ops: set[str] = set()
    for _, _, method in parsed_nodeids:
        if len(seeded) >= args.max_seeds:
            break
        encoded_op = op_from_method(method)
        if not encoded_op or encoded_op in used_ops:
            continue
        used_ops.add(encoded_op)
        try:
            torch_op = torch_op_from_method_op(encoded_op)
        except UnsafeInputError as exc:
            return _error(f"unsafe input: {exc}")
        description = repr(f"Seed for {method}; selected signature: {category_slug}.")
        seeded.append(
            SEED_FN.format(
                safe_op=encoded_op,
                op=torch_op,
                description=description,
            )
        )
    if not seeded:
        seeded = ["# No recognized gradient-op nodeids; add a focused test manually.\n"]

    unit_test_text = UNIT_TEST_TEMPLATE.format(
        category=repr(category_slug),
        test_class=repr(test_class),
        test_functions="\n".join(seeded),
    )
    repro_text = REPRO_STUB.format(
        category=repr(category_slug),
        test_class=repr(test_class),
    )
    try:
        ast.parse(unit_test_text, filename="unit_test_template.py")
        ast.parse(repro_text, filename="repro.py")
    except SyntaxError as exc:
        return _error(f"generated Python did not parse: {exc.msg}")

    readme_lines = [
        f"# RCA workspace: {test_class} / {category_slug}",
        "",
        f"- {len(nodeids)} selected failing or xfailing tests",
        f"- Selection input: `{args.tests_file.name}`",
        "",
        "## Workflow",
        "",
        "1. Confirm `tests.txt` and, when present, `signatures.txt` describe one defect.",
        "2. Replace the placeholder in `repro.py`; prove it fails before the source fix.",
        "3. Reuse a focused upstream test as the fence, or refine `unit_test_template.py`.",
        "4. Implement the source fix and record hypotheses and evidence in `notes.md`.",
        "5. Rerun the complete selection and compare it with the baseline report.",
        "6. Sanitize the local reproduction and handoff text before sharing either one.",
        "",
        "## Files",
        "",
        "- `tests.txt` - selected pytest nodeids",
        "- `signatures.txt` - optional signature breakdown",
        "- `unit_test_template.py` - generated regression-test seeds",
        "- `repro.py` - local off-framework reproduction stub",
        "- `notes.md` - RCA log",
        "- `pr_body.md` - contained pull request body scaffold",
    ]

    notes_text = (
        f"# RCA notes - {test_class} / {category_slug}\n\n"
        "## Hypothesis\n\n## What I tried\n\n## Root cause\n\n"
        "## Fix\n\n## Verification\n\n"
    )
    files = {
        "tests.txt": tests_text.encode(),
        "README.md": ("\n".join(readme_lines) + "\n").encode(),
        "unit_test_template.py": unit_test_text.encode(),
        "repro.py": repro_text.encode(),
        "pr_body.md": PR_BODY_TEMPLATE.encode(),
    }
    if signatures_text is not None:
        files["signatures.txt"] = signatures_text.encode()

    fd = -1
    try:
        fd, identity = open_output_dir(workspace)
        same_sources: set[str] = set()
        if _same_file_with_entry(args.tests_file, fd, "tests.txt"):
            same_sources.add("tests.txt")
        if args.signatures is not None and _same_file_with_entry(
            args.signatures, fd, "signatures.txt"
        ):
            same_sources.add("signatures.txt")
        for name in same_sources:
            files.pop(name, None)

        notes_info = _entry_info(fd, "notes.md")
        check_output_names(fd, identity, [*files, "notes.md"])
        if notes_info is None:
            files["notes.md"] = notes_text.encode()

        require_absent = {"notes.md"} if notes_info is None else set()
        if not args.force:
            require_absent.update(files)
        install_files(fd, identity, files, require_absent=require_absent)
    except (OSError, TypeError, ValueError) as exc:
        detail = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
        if detail == "output directory must not be a symlink":
            detail = "RCA workspace must not be a symlink"
        if isinstance(exc, FileExistsError) and not args.force:
            detail = f"{detail}; use --force"
        return _error(f"cannot create RCA workspace: {detail}")
    finally:
        if fd >= 0:
            os.close(fd)

    print(f"# scaffolded RCA workspace ({len(nodeids)} tests, {len(seeded)} seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
