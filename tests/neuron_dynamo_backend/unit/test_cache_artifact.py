"""
Unit tests for cache_artifact.py precompile caching support.

Tests NeuronOutputCode serialization, monkey-patches, and the
_apply_precompile_patches gating logic.
"""

import dataclasses
import pickle
from unittest.mock import patch

import pytest
import torch
from packaging.version import Version

from torch_neuronx.neuron_dynamo_backend.cache_artifact import (
    NeuronOutputCode,
    _install_builtins_dict_share,
    _install_per_rank_package_dir,
    _install_process_group_reducer,
    _install_transport_reducer,
    _rebuild_process_group,
    _rebuild_unresolved,
    _resolve_package_element,
    _transport_types,
    _UnresolvedObject,
)


class TestNeuronOutputCode:
    """Tests for NeuronOutputCode dataclass."""

    def _make_output_code(self, **kwargs):
        defaults = {
            "graph_name": "test_model",
            "cache_key": "abc123",
            "hlo_cache_key": "abc123",
            "has_collectives": False,
            "retain_device": False,
        }
        defaults.update(kwargs)
        return NeuronOutputCode(**defaults)

    def test_basic_creation(self):
        oc = self._make_output_code()
        assert oc.graph_name == "test_model"
        assert oc.cache_key == "abc123"
        assert oc.hlo_cache_key == "abc123"
        assert oc._boxed_call is True
        assert oc._executor is None

    def test_picklable(self):
        """NeuronOutputCode should survive a pickle round-trip."""
        oc = self._make_output_code(
            graph_name="my_model",
            cache_key="key123",
            hlo_cache_key="hlo456",
            has_collectives=True,
        )
        data = pickle.dumps(oc)
        restored = pickle.loads(data)
        assert restored.graph_name == "my_model"
        assert restored.cache_key == "key123"
        assert restored.hlo_cache_key == "hlo456"
        assert restored.has_collectives is True
        assert restored._executor is None
        assert restored._boxed_call is True

    def test_prepare_for_serialization_clears_executor(self):
        oc = self._make_output_code()
        oc._executor = object()
        oc.prepare_for_serialization()
        assert oc._executor is None

    def test_pickle_after_prepare_for_serialization(self):
        oc = self._make_output_code()
        oc._executor = object()
        oc.prepare_for_serialization()
        data = pickle.dumps(oc)
        restored = pickle.loads(data)
        assert restored._executor is None
        assert restored.graph_name == "test_model"

    def test_post_compile_is_noop(self):
        oc = self._make_output_code()
        oc.post_compile(None, None, None)

    def test_set_triton_bundle_is_noop(self):
        oc = self._make_output_code()
        oc.set_triton_bundle(None)

    def test_boxed_call_attribute(self):
        """OutputCode must have _boxed_call=True for AOTAutograd."""
        oc = self._make_output_code()
        assert hasattr(oc, "_boxed_call")
        assert oc._boxed_call is True

    def test_ensure_neff_loaded_raises_on_missing_hlo(self):
        """_ensure_neff_loaded should raise RuntimeError if HLO cache misses."""
        oc = self._make_output_code(cache_key="nonexistent_key", hlo_bytes=b"")
        with (
            patch("torch_neuronx.python_ops.compilation.hlo_cache.get", return_value=None),
            pytest.raises(RuntimeError, match="HLO cache miss"),
        ):
            oc._ensure_neff_loaded()

    def test_ensure_neff_loaded_skips_if_no_key(self):
        """_ensure_neff_loaded should be a no-op if both cache_key and hlo_cache_key are empty."""
        oc = self._make_output_code(cache_key="", hlo_cache_key="")
        with patch("torch_neuronx.python_ops.compilation.hlo_cache.get") as mock_hlo_get:
            oc._ensure_neff_loaded()
            mock_hlo_get.assert_not_called()

    def test_hlo_bytes_survives_pickle_roundtrip(self):
        """Embedded hlo_bytes should survive a pickle round-trip."""
        oc = self._make_output_code(hlo_bytes=b"fake_hlo_data")
        data = pickle.dumps(oc)
        restored = pickle.loads(data)
        assert restored.hlo_bytes == b"fake_hlo_data"

    def test_prepare_for_serialization_preserves_hlo_bytes(self):
        """prepare_for_serialization clears _executor but preserves hlo_bytes."""
        oc = self._make_output_code(hlo_bytes=b"keep_me")
        oc._executor = object()
        oc.prepare_for_serialization()
        assert oc._executor is None
        assert oc.hlo_bytes == b"keep_me"

    def test_ensure_neff_loaded_uses_embedded_bytes_first(self):
        """_ensure_neff_loaded should use embedded hlo_bytes without hitting HLO cache."""
        oc = self._make_output_code(cache_key="key123", hlo_bytes=b"embedded")
        with (
            patch("torch_neuronx._C.compile_graph") as mock_compile,
            patch("torch_neuronx.python_ops.compilation.hlo_cache.get") as mock_hlo_get,
        ):
            oc._ensure_neff_loaded()
            mock_compile.assert_called_once_with("key123", b"embedded", False, wlo=False)
            mock_hlo_get.assert_not_called()

    def test_ensure_neff_loaded_falls_back_to_hlo_cache(self):
        """_ensure_neff_loaded should fall back to HLO cache when hlo_bytes is empty.

        `hlo_cache.get` returns a `CachedHLO` dataclass (not the old 7-tuple), so the
        fallback must read `entry.hlo_bytes`, not `entry[0]` — the dataclass is not
        subscriptable.
        """
        from torch_neuronx.python_ops.compilation.hlo_cache import CachedHLO

        oc = self._make_output_code(cache_key="key123", hlo_bytes=b"")
        with (
            patch(
                "torch_neuronx.python_ops.compilation.hlo_cache.get",
                return_value=CachedHLO(hlo_bytes=b"from_cache"),
            ),
            patch("torch_neuronx._C.compile_graph") as mock_compile,
        ):
            oc._ensure_neff_loaded()
            mock_compile.assert_called_once_with("key123", b"from_cache", False, wlo=False)

    def test_ensure_neff_loaded_raises_when_both_miss(self):
        """_ensure_neff_loaded should raise when hlo_bytes is empty and HLO cache misses."""
        oc = self._make_output_code(cache_key="missing", hlo_bytes=b"")
        with (
            patch("torch_neuronx.python_ops.compilation.hlo_cache.get", return_value=None),
            pytest.raises(RuntimeError, match="HLO cache miss"),
        ):
            oc._ensure_neff_loaded()


class TestNeuronOutputCodeWLO:
    """WLO async-sidecar path: the wrapped_neff.hlo fetch/parse is deferred from compile() to
    executor-build time (NeuronOutputCode._get_executor), keyed off the "|wlo" cache-key suffix."""

    def _make_output_code(self, **kwargs):
        defaults = {
            "graph_name": "m",
            "has_collectives": False,
            "hlo_bytes": b"embedded",
        }
        defaults.update(kwargs)
        return NeuronOutputCode(**defaults)

    def test_ensure_neff_loaded_resubmits_wlo_for_wlo_key(self):
        """A "|wlo" key must re-submit on the WLO path so a persistent cache hit repopulates the
        wrapped_neff.hlo sidecar (CompilationCache only loads it when the kernel requires it).
        Otherwise a cross-process warm load would run the relaid NEFF with no transforms."""
        oc = self._make_output_code(cache_key="graph:hash|wlo")
        with patch("torch_neuronx._C.compile_graph") as mock_compile:
            oc._ensure_neff_loaded()
        mock_compile.assert_called_once_with("graph:hash|wlo", b"embedded", False, wlo=True)

    def test_ensure_neff_loaded_no_wlo_for_plain_key(self):
        """A non-WLO key must re-submit on the plain path (wlo=False)."""
        oc = self._make_output_code(cache_key="graph:hash")
        with patch("torch_neuronx._C.compile_graph") as mock_compile:
            oc._ensure_neff_loaded()
        mock_compile.assert_called_once_with("graph:hash", b"embedded", False, wlo=False)

    def test_get_executor_fetches_and_parses_sidecar_lazily_for_wlo(self):
        """For a "|wlo" key, _get_executor fetches the sidecar via get_wrapped_neff_hlo, parses it,
        and passes the transforms to the Executor — all lazily, after _ensure_neff_loaded."""
        oc = self._make_output_code(cache_key="graph:hash|wlo")
        transforms = [{"input_index": 0, "reshape": [4, 3], "transpose": [1, 0]}]
        with (
            patch("torch_neuronx._C.compile_graph"),
            patch("torch_neuronx._C.get_wrapped_neff_hlo", return_value=b"sidecar") as mock_get,
            patch(
                "torch_neuronx.neuron_dynamo_backend.compile._parse_wrapped_neff_hlo",
                return_value=transforms,
            ) as mock_parse,
            patch("torch_neuronx.neuron_dynamo_backend.executor.Executor") as mock_executor,
        ):
            oc._get_executor()

        mock_get.assert_called_once_with("graph:hash|wlo")
        mock_parse.assert_called_once_with(b"sidecar")
        assert mock_executor.call_args.kwargs["wlo_transforms"] == transforms

    def test_get_executor_no_sidecar_for_non_wlo(self):
        """For a non-WLO key, _get_executor never fetches a sidecar and builds the Executor with no
        transforms."""
        oc = self._make_output_code(cache_key="graph:hash")
        with (
            patch("torch_neuronx._C.compile_graph"),
            patch("torch_neuronx._C.get_wrapped_neff_hlo") as mock_get,
            patch("torch_neuronx.neuron_dynamo_backend.executor.Executor") as mock_executor,
        ):
            oc._get_executor()

        mock_get.assert_not_called()
        assert mock_executor.call_args.kwargs["wlo_transforms"] == []


class TestLoadGuardsStateTracingFix:
    """Tests for the load_guards_state patch from _apply_precompile_patches()."""

    # PT 2.14 removed ImportSource.__post_init__ (which called install_guard),
    # so unpickling no longer requires a TracingContext.
    _PT_214_PLUS = Version(torch.__version__.split("+")[0]) >= Version("2.14")

    @staticmethod
    def _pickle_import_source():
        # ImportSource.__post_init__ calls install_guard, which needs a trace,
        # so build it inside one to get bytes whose unpickle re-runs __post_init__.
        # PT 2.14 removed __post_init__, but wrapping in a trace is harmless.
        from torch._dynamo.source import ImportSource
        from torch._guards import TracingContext, tracing

        with tracing(TracingContext(fake_mode=None)):
            src = ImportSource(module_name="torch")
        return pickle.dumps(src)

    def test_unpatched_load_guards_state_raises(self):
        """Without the patch, unpickling an ImportSource with no trace raises on PT<=2.13.
        On PT 2.14+, __post_init__ was removed upstream so it succeeds without a trace."""
        import torch._dynamo.package as pkg
        from torch._dynamo.source import ImportSource
        from torch._guards import TracingContext

        payload = self._pickle_import_source()
        assert TracingContext.try_get() is None

        if self._PT_214_PLUS:
            # PT 2.14: ImportSource.__post_init__ removed; no trace needed
            result = pkg.load_guards_state(payload)
            assert isinstance(result, ImportSource)
            assert result.module_name == "torch"
        else:
            # PT 2.13: __post_init__ calls install_guard() which needs a trace
            with pytest.raises(RuntimeError, match="must be called within an ongoing trace"):
                pkg.load_guards_state(payload)

    def test_unpickle_importsource_no_trace_does_not_raise(self):
        """The patch lets the same unpickle succeed with no active trace."""
        import torch._dynamo.package as pkg
        from torch._dynamo.source import ImportSource
        from torch._guards import TracingContext

        from torch_neuronx.neuron_dynamo_backend.cache_artifact import (
            _apply_precompile_patches,
        )

        payload = self._pickle_import_source()
        orig = pkg.load_guards_state
        try:
            _apply_precompile_patches()
            assert TracingContext.try_get() is None
            result = pkg.load_guards_state(payload)
            assert isinstance(result, ImportSource)
            assert result.module_name == "torch"
            assert TracingContext.try_get() is None
        finally:
            pkg.load_guards_state = orig


class TestResolvePackageElement:
    """Per-rank package directory element from the environment."""

    def test_cores(self, monkeypatch):
        monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8,9")
        assert _resolve_package_element() == "cores_8_9"

    def test_cores_with_spaces(self, monkeypatch):
        """A space in the value must not leak into the directory name."""
        monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8, 9")
        assert _resolve_package_element() == "cores_8_9"

    def test_cores_range(self, monkeypatch):
        monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0-3")
        assert _resolve_package_element() == "cores_0_3"

    def test_rank_fallback(self, monkeypatch):
        monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
        monkeypatch.setenv("RANK", "3")
        with patch("torch.distributed.is_initialized", return_value=False):
            assert _resolve_package_element() == "rank_3"

    def test_none_when_no_source(self, monkeypatch):
        monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
        monkeypatch.delenv("RANK", raising=False)
        with patch("torch.distributed.is_initialized", return_value=False):
            assert _resolve_package_element() is None

    def test_dist_rank_when_initialized(self, monkeypatch):
        monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=7),
        ):
            assert _resolve_package_element() == "rank_7"


class TestUnresolvedObject:
    def test_getattr_returns_standin(self):
        obj = _UnresolvedObject("root")
        child = obj.some_attr
        assert isinstance(child, _UnresolvedObject)
        assert "root.some_attr" in repr(child)


class TestTransportTypes:
    def test_includes_core_transport_types(self):
        import mmap
        import socket
        import threading

        types = _transport_types()
        assert isinstance(types, tuple)
        assert socket.socket in types
        assert mmap.mmap in types
        assert type(threading.Lock()) in types


class TestInstallPerRankPackageDir:
    def test_appends_element_and_is_idempotent(self, monkeypatch):
        from torch._dynamo.package import DiskDynamoCache

        monkeypatch.setattr(DiskDynamoCache, "path_prefix", lambda self: "/base")
        monkeypatch.setattr(
            "torch_neuronx.neuron_dynamo_backend.cache_artifact._resolve_package_element",
            lambda: "rank_5",
        )
        inst = DiskDynamoCache.__new__(DiskDynamoCache)

        _install_per_rank_package_dir()
        assert inst.path_prefix() == "/base/rank_5"

        # A second install must not wrap the wrapper again.
        _install_per_rank_package_dir()
        assert inst.path_prefix() == "/base/rank_5"

    def test_keeps_shared_dir_when_no_rank(self, monkeypatch):
        from torch._dynamo.package import DiskDynamoCache

        monkeypatch.setattr(DiskDynamoCache, "path_prefix", lambda self: "/base")
        monkeypatch.setattr(
            "torch_neuronx.neuron_dynamo_backend.cache_artifact._resolve_package_element",
            lambda: None,
        )
        inst = DiskDynamoCache.__new__(DiskDynamoCache)

        _install_per_rank_package_dir()
        assert inst.path_prefix() == "/base"


class TestRebuildProcessGroup:
    def test_returns_live_group(self):
        sentinel = object()
        with patch(
            "torch.distributed.distributed_c10d._resolve_process_group",
            return_value=sentinel,
        ):
            assert _rebuild_process_group("grp") is sentinel

    def test_falls_back_to_standin_on_error(self):
        with patch(
            "torch.distributed.distributed_c10d._resolve_process_group",
            side_effect=RuntimeError("no such group"),
        ):
            result = _rebuild_process_group("grp")
        assert isinstance(result, _UnresolvedObject)


class TestInstallTransportReducer:
    def test_socket_reduces_to_standin(self, monkeypatch):
        import socket

        from torch._dynamo.guards import GuardsStatePickler

        # Record the current override for automatic restore, then patch.
        monkeypatch.setattr(
            GuardsStatePickler, "reducer_override", GuardsStatePickler.reducer_override
        )
        _install_transport_reducer()
        s = socket.socket()
        try:
            # A transport object hits the isinstance branch, which never touches
            # self, so None is a safe stand-in for the pickler.
            fn, args = GuardsStatePickler.reducer_override(None, s)
        finally:
            s.close()
        assert fn is _rebuild_unresolved
        assert isinstance(fn(*args), _UnresolvedObject)


class TestInstallProcessGroupReducer:
    def test_drops_process_group_and_is_idempotent(self, monkeypatch):
        import torch._dynamo.guards as dynamo_guards

        try:
            from torch._C._distributed_c10d import ProcessGroup
        except ImportError:
            pytest.skip("ProcessGroup not available")
        if getattr(dynamo_guards, "_get_unsupported_types", None) is None:
            pytest.skip("no _get_unsupported_types in this torch")
        if ProcessGroup not in dynamo_guards._get_unsupported_types():
            pytest.skip("ProcessGroup not in the unsupported list")

        monkeypatch.setattr(
            dynamo_guards, "_get_unsupported_types", dynamo_guards._get_unsupported_types
        )
        _install_process_group_reducer()
        assert ProcessGroup not in dynamo_guards._get_unsupported_types()
        # Idempotent: a second install must not chain another wrapper.
        _install_process_group_reducer()
        assert ProcessGroup not in dynamo_guards._get_unsupported_types()


class TestInstallBuiltinsDictShare:
    def test_reuses_existing_builtins_name(self, monkeypatch):
        import builtins
        from types import SimpleNamespace

        from torch._dynamo.output_graph import OutputGraph, get_builtins_dict

        # get_builtins_dict reads scope["__builtins__"], so seed it.
        scope = {"__builtins__": builtins.__dict__}
        f_builtins = get_builtins_dict(scope)
        scope["__builtins_dict___0"] = f_builtins

        monkeypatch.setattr(
            OutputGraph,
            "install_builtins_dict_in_fglobals",
            OutputGraph.install_builtins_dict_in_fglobals,
        )
        _install_builtins_dict_share()
        fake = SimpleNamespace(global_scope=scope)
        name = OutputGraph.install_builtins_dict_in_fglobals(fake)
        assert name == "__builtins_dict___0"
