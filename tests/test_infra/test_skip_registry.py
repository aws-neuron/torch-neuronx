"""Tests for neuron_test_registry matching logic."""

from tests.pytorch_tests_native.neuron_test_registry import (
    SkipGroup,
    _group_matches,
)

FSDP_NODEID = (
    "tests/pytorch_tests_native/test/distributed/" "fsdp/test_fsdp.py::TestFSDP::test_something"
)


# -- Folder matching --


def test_folder_match():
    g = SkipGroup(reason="x", folders=["distributed/"])
    assert _group_matches(g, FSDP_NODEID)


def test_folder_no_match():
    g = SkipGroup(reason="x", folders=["inductor/"])
    assert not _group_matches(g, FSDP_NODEID)


# -- Tests field: exact component matching --


def test_match_by_test_name():
    g = SkipGroup(reason="x", tests=["test_add"])
    assert _group_matches(g, "test/test_ops.py::TestOps::test_add")


def test_no_prefix_match():
    """test_add must NOT match test_add_scalar."""
    g = SkipGroup(reason="x", tests=["test_add"])
    assert not _group_matches(g, "test/test_ops.py::TestOps::test_add_scalar")


def test_match_by_class():
    g = SkipGroup(reason="x", tests=["TestSparse"])
    assert _group_matches(g, "test/test_sparse.py::TestSparse::test_coo")


def test_class_no_prefix_match():
    """TestLinear must NOT match TestLinearAlgebra."""
    g = SkipGroup(reason="x", tests=["TestLinear"])
    assert not _group_matches(g, "test/test_linalg.py::TestLinearAlgebra::test_svd")


def test_match_by_filename():
    g = SkipGroup(reason="x", tests=["test_sparse.py"])
    assert _group_matches(g, "test/test_sparse.py::TestSparse::test_coo")


def test_filename_no_prefix_match():
    g = SkipGroup(reason="x", tests=["test_sparse.py"])
    assert not _group_matches(g, "test/test_sparse_csr.py::TestCSR::test_csr")


def test_filename_without_extension_no_match():
    """test_fsdp (no .py) is not a component, so no match."""
    g = SkipGroup(reason="x", tests=["test_fsdp"])
    assert not _group_matches(g, FSDP_NODEID)


# -- Multi-component matching (file::class, class::test) --


def test_match_file_and_class():
    g = SkipGroup(reason="x", tests=["test_sparse.py::TestSparse"])
    assert _group_matches(g, "test/test_sparse.py::TestSparse::test_coo")


def test_file_and_class_no_cross_file():
    g = SkipGroup(reason="x", tests=["test_sparse.py::TestSparse"])
    assert not _group_matches(g, "test/test_linalg.py::TestSparse::test_coo")


def test_match_class_and_test():
    g = SkipGroup(reason="x", tests=["TestOps::test_add"])
    assert _group_matches(g, "test/test_ops.py::TestOps::test_add")
    assert not _group_matches(g, "test/test_ops.py::TestOther::test_add")


def test_fully_qualified():
    g = SkipGroup(reason="x", tests=["test_fsdp.py::TestFSDP::test_something"])
    assert _group_matches(g, FSDP_NODEID)


def test_fully_qualified_wrong_test():
    g = SkipGroup(reason="x", tests=["test_fsdp.py::TestFSDP::test_other"])
    assert not _group_matches(g, FSDP_NODEID)


def test_non_adjacent_components_no_match():
    """distributed and test_fsdp.py are not adjacent (fsdp dir between)."""
    g = SkipGroup(reason="x", tests=["distributed::test_fsdp.py"])
    assert not _group_matches(g, FSDP_NODEID)


# -- Pattern matching (glob on test function name) --


def test_pattern_match():
    g = SkipGroup(reason="x", patterns=["*_sparse_*"])
    assert _group_matches(g, "test/test_ops.py::TestOps::test_sparse_coo_neuron")


def test_pattern_no_match():
    g = SkipGroup(reason="x", patterns=["*_sparse_*"])
    assert not _group_matches(g, "test/test_ops.py::TestOps::test_add_neuron")


def test_pattern_matches_only_function_name():
    """Pattern should match against function name, not the path."""
    g = SkipGroup(reason="x", patterns=["test_ops*"])
    assert not _group_matches(g, "test/test_ops.py::TestOps::test_add")


def test_pattern_does_not_match_class_name():
    """Glob only checks function name, not class."""
    g = SkipGroup(reason="x", patterns=["*FSDP*"])
    assert not _group_matches(g, FSDP_NODEID)


# -- Glob wildcards in tests entries --


def test_glob_in_middle_component_matches():
    """Wildcard in class component matches both PT2.12 and PT2.13 class names."""
    g = SkipGroup(reason="x", tests=["test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_foo"])
    assert _group_matches(g, "test/test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_foo")
    assert _group_matches(g, "test/test_binary_ufuncs.py::TestBinaryUfuncsDeviceNEURON::test_foo")


def test_glob_in_last_component_matches():
    """Wildcard in method component matches method with and without suffix."""
    g = SkipGroup(reason="x", tests=["TestAccelerator*::test_memory_stats*"])
    assert _group_matches(g, "test/test_accelerator.py::TestAccelerator::test_memory_stats")
    assert _group_matches(
        g, "test/test_accelerator.py::TestAcceleratorNEURON::test_memory_stats_neuron"
    )


def test_glob_in_first_component_matches():
    """Wildcard in first (class) component is NOT silently skipped."""
    g = SkipGroup(reason="x", tests=["*CtxManagerTests::test_foo*"])
    assert _group_matches(g, "test/dynamo/test_ctx_manager.py::CtxManagerTests::test_foo")
    assert _group_matches(
        g,
        "test/dynamo/test_nested_graph_breaks_wrapped.py"
        "::NestedGraphBreaksCtxManagerTests::test_foo_nested_graph_breaks",
    )


def test_glob_bare_single_component_matches():
    """Bare glob entry (no :: or /) is NOT silently ignored."""
    g = SkipGroup(reason="x", tests=["test_foo*"])
    assert _group_matches(g, "test/test_ops.py::TestOps::test_foo_bar")
    assert not _group_matches(g, "test/test_ops.py::TestOps::test_bar")


def test_glob_does_not_match_different_file():
    """File-scoped glob does not leak to other files."""
    g = SkipGroup(reason="x", tests=["test_binary_ufuncs.py::TestBinaryUfuncs*NEURON::test_foo"])
    assert not _group_matches(g, "test/test_other.py::TestBinaryUfuncsDeviceNEURON::test_foo")


def test_glob_is_case_sensitive():
    """fnmatchcase: TestFooNEURON does NOT match testfooneuron."""
    g = SkipGroup(reason="x", tests=["TestBinaryUfuncs*NEURON::test_foo"])
    assert _group_matches(g, "test/test_binary_ufuncs.py::TestBinaryUfuncsNEURON::test_foo")
    assert not _group_matches(g, "test/test_binary_ufuncs.py::testbinaryufuncsneuron::test_foo")


def test_glob_bracket_pattern():
    """[seq] bracket globs are supported."""
    g = SkipGroup(reason="x", tests=["test_[abc]oo"])
    assert _group_matches(g, "test/test_ops.py::TestOps::test_boo")
    assert not _group_matches(g, "test/test_ops.py::TestOps::test_doo")


def test_glob_literal_still_matches():
    """Literals still work correctly — fnmatchcase reduces to equality for plain strings."""
    g = SkipGroup(reason="x", tests=["CtxManagerTests::test_foo"])
    assert _group_matches(g, "test/dynamo/test_ctx_manager.py::CtxManagerTests::test_foo")
    assert not _group_matches(g, "test/dynamo/test_ctx_manager.py::CtxManagerTests::test_bar")


# -- Empty group matches nothing --


def test_empty_group():
    g = SkipGroup(reason="x")
    assert not _group_matches(g, "test/test_ops.py::TestOps::test_add")


# -- Multiple entries in one group --


def test_multiple_tests():
    g = SkipGroup(reason="x", tests=["test_add", "test_mul"])
    assert _group_matches(g, "test/test_ops.py::TestOps::test_add")
    assert _group_matches(g, "test/test_ops.py::TestOps::test_mul")
    assert not _group_matches(g, "test/test_ops.py::TestOps::test_sub")


# -- __post_init__ pre-computation --


def test_post_init_simple_tests():
    """Simple (unqualified) entries go into _simple_tests set."""
    g = SkipGroup(reason="x", tests=["test_add", "TestSparse", "test_ops.py"])
    assert g._simple_tests == {"test_add", "TestSparse", "test_ops.py"}
    assert g._qualified_by_first == {}


def test_post_init_qualified_tests():
    """Qualified entries (with :: or /) go into _qualified_by_first."""
    g = SkipGroup(reason="x", tests=["test_ops.py::TestOps", "distributed/fsdp"])
    assert g._simple_tests == set()
    assert "test_ops.py" in g._qualified_by_first
    assert g._qualified_by_first["test_ops.py"] == [["test_ops.py", "TestOps"]]
    assert "distributed" in g._qualified_by_first
    assert g._qualified_by_first["distributed"] == [["distributed", "fsdp"]]


def test_post_init_mixed():
    """Mixed simple and qualified entries are split correctly."""
    g = SkipGroup(
        reason="x",
        tests=["test_add", "test_ops.py::TestOps", "TestSparse"],
    )
    assert g._simple_tests == {"test_add", "TestSparse"}
    assert len(g._qualified_by_first) == 1
    assert g._qualified_by_first["test_ops.py"] == [["test_ops.py", "TestOps"]]


# -- Mixed group with all field types --


def test_mixed_group_folder_wins():
    """Folder match takes priority over tests and patterns."""
    g = SkipGroup(
        reason="x",
        folders=["distributed/"],
        tests=["test_other"],
        patterns=["*never*"],
    )
    assert _group_matches(g, FSDP_NODEID)


def test_mixed_group_simple_test_wins():
    """Simple test match works when folder doesn't match."""
    g = SkipGroup(
        reason="x",
        folders=["inductor/"],
        tests=["test_something"],
        patterns=["*never*"],
    )
    assert _group_matches(g, FSDP_NODEID)


def test_mixed_group_qualified_test_wins():
    """Qualified test match works when folder and simple don't match."""
    g = SkipGroup(
        reason="x",
        folders=["inductor/"],
        tests=["test_fsdp.py::TestFSDP"],
        patterns=["*never*"],
    )
    assert _group_matches(g, FSDP_NODEID)


def test_mixed_group_pattern_wins():
    """Pattern match works when folder and tests don't match."""
    g = SkipGroup(
        reason="x",
        folders=["inductor/"],
        tests=["test_other"],
        patterns=["*something*"],
    )
    assert _group_matches(g, FSDP_NODEID)


def test_mixed_group_nothing_matches():
    """No match when all fields miss."""
    g = SkipGroup(
        reason="x",
        folders=["inductor/"],
        tests=["test_other"],
        patterns=["*never*"],
    )
    assert not _group_matches(g, FSDP_NODEID)


# -- Priority: skip checked before xfail --


def test_skip_before_xfail():
    """If a test matches both skip and xfail, skip wins."""
    from unittest.mock import MagicMock

    from tests.pytorch_tests_native.neuron_test_registry import (
        SKIP_GROUPS,
        XFAIL_GROUPS,
        apply_markers,
    )

    orig_skip, orig_xfail = SKIP_GROUPS[:], XFAIL_GROUPS[:]
    SKIP_GROUPS.clear()
    XFAIL_GROUPS.clear()
    SKIP_GROUPS.append(SkipGroup(reason="skip reason", tests=["test_foo"]))
    XFAIL_GROUPS.append(SkipGroup(reason="xfail reason", tests=["test_foo"]))

    try:
        item = MagicMock()
        item.nodeid = "test/test_ops.py::TestOps::test_foo"
        apply_markers(item)

        item.add_marker.assert_called_once()
        marker = item.add_marker.call_args[0][0]
        assert marker.name == "skip"
        assert marker.kwargs["reason"] == "skip reason"
    finally:
        SKIP_GROUPS.clear()
        SKIP_GROUPS.extend(orig_skip)
        XFAIL_GROUPS.clear()
        XFAIL_GROUPS.extend(orig_xfail)


# -- apply_markers with xfail --


def test_xfail_applied():
    """Xfail marker applied when no skip matches."""
    from unittest.mock import MagicMock

    from tests.pytorch_tests_native.neuron_test_registry import (
        SKIP_GROUPS,
        XFAIL_GROUPS,
        apply_markers,
    )

    orig_skip, orig_xfail = SKIP_GROUPS[:], XFAIL_GROUPS[:]
    SKIP_GROUPS.clear()
    XFAIL_GROUPS.clear()
    XFAIL_GROUPS.append(SkipGroup(reason="xfail reason", tests=["test_foo"]))

    try:
        item = MagicMock()
        item.nodeid = "test/test_ops.py::TestOps::test_foo"
        apply_markers(item)

        item.add_marker.assert_called_once()
        marker = item.add_marker.call_args[0][0]
        assert marker.name == "xfail"
        assert marker.kwargs["reason"] == "xfail reason"
    finally:
        SKIP_GROUPS.clear()
        SKIP_GROUPS.extend(orig_skip)
        XFAIL_GROUPS.clear()
        XFAIL_GROUPS.extend(orig_xfail)


def test_no_marker_when_no_match():
    """No marker applied when nothing matches."""
    from unittest.mock import MagicMock

    from tests.pytorch_tests_native.neuron_test_registry import (
        SKIP_GROUPS,
        XFAIL_GROUPS,
        apply_markers,
    )

    orig_skip, orig_xfail = SKIP_GROUPS[:], XFAIL_GROUPS[:]
    SKIP_GROUPS.clear()
    XFAIL_GROUPS.clear()
    SKIP_GROUPS.append(SkipGroup(reason="x", tests=["test_other"]))

    try:
        item = MagicMock()
        item.nodeid = "test/test_ops.py::TestOps::test_foo"
        apply_markers(item)

        item.add_marker.assert_not_called()
    finally:
        SKIP_GROUPS.clear()
        SKIP_GROUPS.extend(orig_skip)
        XFAIL_GROUPS.clear()
        XFAIL_GROUPS.extend(orig_xfail)


# -- Auto-discovery --


def test_auto_discovery_loads_registries():
    """skip_registry/__init__.py discovers *_registry.py files."""
    from tests.pytorch_tests_native import skip_registry

    # At least one registry file should have been loaded
    assert len(skip_registry.SKIP_GROUPS) + len(skip_registry.XFAIL_GROUPS) > 0


def test_loaded_groups_are_skip_groups():
    """All loaded entries are SkipGroup instances."""
    from tests.pytorch_tests_native import skip_registry

    for g in skip_registry.SKIP_GROUPS:
        assert isinstance(g, SkipGroup)
    for g in skip_registry.XFAIL_GROUPS:
        assert isinstance(g, SkipGroup)


# -- Duplicate detection --


def test_no_test_in_both_skip_and_xfail():
    """A test should not appear in both SKIP and XFAIL registries."""
    from tests.pytorch_tests_native.neuron_test_registry import (
        SKIP_GROUPS,
        XFAIL_GROUPS,
    )

    skip_tests = set()
    for g in SKIP_GROUPS:
        skip_tests.update(g.tests)

    xfail_tests = set()
    for g in XFAIL_GROUPS:
        xfail_tests.update(g.tests)

    duplicates = skip_tests & xfail_tests
    assert (
        not duplicates
    ), f"Tests in both SKIP and XFAIL (xfail is dead code): {sorted(duplicates)}"


# -- Data-quality invariants --


def test_every_group_has_reason():
    """No group may have an empty reason."""
    from tests.pytorch_tests_native.neuron_test_registry import (
        SKIP_GROUPS,
        XFAIL_GROUPS,
    )

    for g in SKIP_GROUPS + XFAIL_GROUPS:
        assert g.reason, f"group missing reason: {g}"


def test_every_group_has_targets():
    """A group with no tests, patterns, and folders selects nothing."""
    from tests.pytorch_tests_native.neuron_test_registry import (
        SKIP_GROUPS,
        XFAIL_GROUPS,
    )

    for g in SKIP_GROUPS + XFAIL_GROUPS:
        assert g.tests or g.patterns or g.folders, f"group selects nothing: {g.reason!r}"


def test_no_duplicate_tests_in_same_group():
    """Catches copy-paste mistakes in a single SkipGroup's tests list."""
    from tests.pytorch_tests_native.neuron_test_registry import (
        SKIP_GROUPS,
        XFAIL_GROUPS,
    )

    for g in SKIP_GROUPS + XFAIL_GROUPS:
        assert len(g.tests) == len(set(g.tests)), f"duplicate tests in group: {g.reason!r}"


def test_public_constants_are_lists_of_strings():
    """Every exported xfail/skipped_tests constant must be list[str]."""
    from tests.pytorch_tests_native.skip_registry import skipped_tests, xfail_tests

    for pkg in (xfail_tests, skipped_tests):
        for name in dir(pkg):
            if name.startswith("_") or not name.isupper():
                continue
            val = getattr(pkg, name)
            assert isinstance(val, list), f"{pkg.__name__}.{name} is not a list"
            assert all(
                isinstance(t, str) for t in val
            ), f"{pkg.__name__}.{name} contains non-string items"


def test_init_reexports_match_leaf_definitions():
    """Every name imported by xfail_tests/skipped_tests __init__.py must be defined
    in its leaf file. Catches partial-sync issues where __init__.py and a leaf
    drift out of agreement.
    """
    import ast
    from pathlib import Path

    import tests.pytorch_tests_native.skip_registry as reg_pkg

    pkg_root = Path(reg_pkg.__file__).parent
    errors: list[str] = []
    for pkg_dir in ["xfail_tests", "skipped_tests"]:
        pkg = pkg_root / pkg_dir
        init = pkg / "__init__.py"
        for node in ast.parse(init.read_text()).body:
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            leaf = pkg / (node.module.replace(".", "/") + ".py")
            if not leaf.exists():
                errors.append(f"{init} imports from missing leaf: {node.module}")
                continue
            defined = set()
            for n in ast.parse(leaf.read_text()).body:
                if isinstance(n, ast.Assign):
                    for tgt in n.targets:
                        if isinstance(tgt, ast.Name) and tgt.id.isupper():
                            defined.add(tgt.id)
            for alias in node.names:
                if alias.name not in defined:
                    errors.append(
                        f"{init}:{node.lineno} imports {alias.name!r} from "
                        f"{node.module} but that file defines {sorted(defined)}"
                    )
    assert not errors, "\n".join(errors)
