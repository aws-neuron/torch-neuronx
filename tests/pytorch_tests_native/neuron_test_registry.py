"""Neuron skip/xfail registry for upstream PyTorch tests.

Single source of truth for all tests that need to be skipped or xfailed
when running the upstream PyTorch test suite on Neuron. Applied automatically
via conftest.py using pytest_collection_modifyitems — no patches to upstream code.

Organization:
    Each SkipGroup represents ONE root cause. When the root cause is fixed,
    delete the entire group and all its tests are re-enabled.

    Three levels of granularity per group:
    - folders:  skip entire directories (e.g., "distributed/")
    - tests:    contiguous component match against nodeid split by :: and /
                Supports any combination: "test_method", "TestClass",
                "test_sparse.py", "test_sparse.py::TestClass",
                "distributed::test_comm.py::TestComm", etc.
    - patterns:    glob patterns against class and test name only
"""

import fnmatch
from dataclasses import dataclass, field

import pytest


def _split_nodeid(nodeid: str) -> list[str]:
    """Split a nodeid into components (path segments + class + test)."""
    return nodeid.replace("/", "::").split("::")


@dataclass
class SkipGroup:
    """A group of tests to skip/xfail, tied to a single root cause."""

    reason: str
    tests: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)
    # Only used with `patterns` (glob matching). Ignored for `tests` and `folders`.
    # "test_method" = glob against the test method name (last :: component)
    # "test_class"  = glob against the test class name (second :: component)
    match_target: str = "test_method"

    def __post_init__(self):
        # Pre-compute: split tests into simple names (set lookup) vs qualified (list match)
        self._simple_tests: set[str] = set()
        # Index qualified entries by first component for fast filtering
        self._qualified_by_first: dict[str, list[list[str]]] = {}
        # Entries that contain a glob wildcard — bypass both exact-match shortcuts
        # and always run through _is_contiguous_match.
        self._glob_tests: list[list[str]] = []
        for t in self.tests:
            has_glob = any(c in t for c in ("*", "?", "["))
            if "::" not in t and "/" not in t:
                if has_glob:
                    self._glob_tests.append(_split_nodeid(t))
                else:
                    self._simple_tests.add(t)
            else:
                parts = _split_nodeid(t)
                if has_glob:
                    self._glob_tests.append(parts)
                else:
                    self._qualified_by_first.setdefault(parts[0], []).append(parts)


# =============================================================================
# SKIP: Tests that should not run with Neuron backend
# Each group = one root cause. Fix the cause, delete the group.
# =============================================================================

SKIP_GROUPS: list[SkipGroup] = []

XFAIL_GROUPS: list[SkipGroup] = []


# Pull in all registries from skip_registry folder
def _load_skip_registry():
    from tests.pytorch_tests_native import skip_registry

    SKIP_GROUPS.extend(skip_registry.SKIP_GROUPS)
    XFAIL_GROUPS.extend(skip_registry.XFAIL_GROUPS)


_load_skip_registry()


# =============================================================================
# Matching logic (used by conftest.py)
# =============================================================================


def _is_contiguous_match(skip_entry: list, pytest_nodeid: list) -> bool:
    """Check if skip_entry appears as a contiguous subsequence in pytest_nodeid.

    Parts containing ``*`` or ``?`` are matched with fnmatch glob rather than
    exact equality. Glob entries are routed through ``_glob_tests`` in
    ``__post_init__`` so they bypass both exact-match shortcuts in ``_match``
    (the ``_simple_tests`` set and ``_qualified_by_first`` dict lookup).
    This means wildcards work correctly in any position — first, middle, last,
    or as a bare single-component entry.
    """

    def _parts_match(entry_parts: list, nodeid_slice: list) -> bool:
        return all(
            fnmatch.fnmatchcase(n, e) for e, n in zip(entry_parts, nodeid_slice, strict=False)
        )

    return any(
        _parts_match(skip_entry, pytest_nodeid[i : i + len(skip_entry)])
        for i in range(len(pytest_nodeid) - len(skip_entry) + 1)
    )


def _match(
    group: SkipGroup,
    filepath: str,
    parts: list[str],
    parts_set: set[str],
    func_name: str,
) -> bool:
    """Check if a test matches any entry in a group.

    Matching order (first match wins):
    1. folders — checks if "test/{folder}" is in the file path
    2. tests   — splits nodeid into components by / and ::, then
                 checks if the entry is a contiguous subsequence.
                 Supports any combination: "test_name", "TestClass",
                 "test_file.py", "test_file.py::TestClass", etc.
    3. patterns — glob match against class and test name only

    Note: pytest nodeids are always fully qualified paths like
    "tests/pytorch_tests_native/test/distributed/file.py::TestClass::test_method"
    so they will always be longer than or equal to any skip entry we define.
    """
    if any(f"test/{f}" in filepath for f in group.folders):
        return True
    if group._simple_tests & parts_set:
        return True
    # Only check qualified entries whose first component exists in the nodeid
    for first, entries in group._qualified_by_first.items():
        if first in parts_set:
            for e in entries:
                if _is_contiguous_match(e, parts):
                    return True
    # Glob entries bypass the exact-match shortcuts above
    for e in group._glob_tests:
        if _is_contiguous_match(e, parts):
            return True
    if group.match_target == "test_class":
        class_name = parts[-2] if len(parts) >= 2 else func_name
        return any(fnmatch.fnmatch(class_name, p) for p in group.patterns)
    return any(fnmatch.fnmatch(func_name, p) for p in group.patterns)


def _group_matches(group: SkipGroup, nodeid: str) -> bool:
    """Check if a test matches any entry in a group.

    Convenience wrapper that splits the nodeid. Used by unit tests.
    Production path uses apply_markers which splits once.
    """
    filepath = nodeid.split("::")[0]
    parts = _split_nodeid(nodeid)
    return _match(group, filepath, parts, set(parts), parts[-1])


def apply_markers(item: pytest.Item) -> None:
    """Apply skip/xfail markers to a pytest item based on the registry."""
    nodeid = item.nodeid
    filepath = nodeid.split("::")[0]
    parts = _split_nodeid(nodeid)
    parts_set = set(parts)
    func_name = parts[-1]

    for group in SKIP_GROUPS:
        if _match(group, filepath, parts, parts_set, func_name):
            item.add_marker(pytest.mark.skip(reason=group.reason))
            return

    for group in XFAIL_GROUPS:
        if _match(group, filepath, parts, parts_set, func_name):
            item.add_marker(pytest.mark.xfail(reason=group.reason, strict=False))
            return
