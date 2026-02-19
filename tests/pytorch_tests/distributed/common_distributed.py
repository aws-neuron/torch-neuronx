import importlib
import os
import pickle
import tempfile
from contextlib import suppress

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcContinuousTest, MultiProcessTestCase

import torch_neuronx
from conftest import SANITY_PT_DISTRIBUTED_TESTS
from tests.pytorch_tests.distributed.neuron_patch import NeuronPatcher, pre_run_patches
from tests.utils.neuron_test_utils import get_core_start_index, get_pytest_worker_index

# We run with a maximum of 8 workers for the distributed tests.
# Since since we have 64 cores, we need to set this to 8 to avoid over
# subscription and to avoid running out of cores for other tests
NEURON_RT_NUM_CORES = "8"


class NeuronDistributedBase:
    @property
    def world_size(self):
        return 8

    @property
    def backend(self):
        return "neuron"

    @property
    def device(self):
        return torch.device("neuron", self.rank)

    @classmethod
    def _setup_neuron_environment(cls, rank, set_rt_visible_cores=False):
        process_index = get_pytest_worker_index()
        rt_port = int(os.environ.get("NEURON_RT_PORT", "2025"))
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"localhost:{rt_port + process_index}"
        if set_rt_visible_cores:
            core_start_index = get_core_start_index()
            os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_start_index + rank)

    @classmethod
    def _apply_patches(cls, exclude: list[str] | None = None):
        exclude = exclude or []
        patches = NeuronPatcher.collect_patches()
        for p_name, p in patches.items():
            if p_name not in exclude:
                p.start()

    def _setup_process_group(self, register_default=False):
        """Common process group setup logic"""
        if not dist.is_initialized():
            store = dist.FileStore(self.file_name, self.world_size)
            world_size = self.world_size
            core_start_index = get_core_start_index()
            os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_start_index + self.rank)
            dist.init_process_group(
                backend=self.backend, world_size=world_size, rank=self.rank, store=store
            )

        if register_default:
            torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

        return dist.distributed_c10d._get_default_group()

    def init_pg(self, *args, **kwargs) -> None:
        self._setup_process_group()

    def _get_process_group(self):
        return self._setup_process_group()

    def create_pg(self, device=None):
        """Create and return a process group for the given device"""
        return self._setup_process_group()

    def _init_process_group(self, *args, **kwargs):
        """Initialize process group with neuron backend"""
        self._setup_process_group(register_default=True)


def create_test_classes(base_class, module_globals, spec_file="test_spec.json"):
    # Running Pre-run Patches.
    pre_run_patches()

    spec_basename = os.path.splitext(os.path.basename(spec_file))[0]
    cache_key = f"NEURON_TEST_CONFIG_CACHE_{spec_basename}"
    cache_prefix = f"neuron_test_configs_{spec_basename}_"

    config_cache_file = os.environ.get(cache_key)
    if not config_cache_file:
        fd, config_cache_file = tempfile.mkstemp(prefix=cache_prefix, suffix=".pkl")
        os.close(fd)
        os.environ[cache_key] = config_cache_file

    config_data = None
    if os.path.exists(config_cache_file):
        try:
            with open(config_cache_file, "rb") as f:
                config_data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            # Corrupted cache, remove it
            os.remove(config_cache_file)

    if config_data is None:
        from tests.pytorch_tests.utils import setup_pytorch_tests

        configs = setup_pytorch_tests(
            spec_file=spec_file,
            sanity_pt_distributed_tests=SANITY_PT_DISTRIBUTED_TESTS,
        )
        config_data = []
        for config in configs:
            config_data.append(
                {
                    "module_name": config.test_module.__name__,
                    "test_class_name": config.test_class_name,
                    "test_methods": config.test_methods,
                    "xfail": config.xfail,
                }
            )
        with open(config_cache_file, "wb") as f:
            pickle.dump(config_data, f)

    for config in config_data:
        # Skip if no test methods to run
        if not config["test_methods"]:
            continue
        needs_setup = False
        needs_teardown = False
        test_module = importlib.import_module(config["module_name"])

        test_class = getattr(test_module, config["test_class_name"])
        class_name = config["test_class_name"]
        neuron_class_name = f"TestNeuron{class_name}"

        selected_base_class = _select_neuron_base_class(test_class, base_class)

        neuron_test_class = type(
            neuron_class_name,
            (selected_base_class, test_class),
            {"__module__": module_globals.get("__name__", "__main__")},
        )

        if hasattr(test_class, "world_size"):
            test_class_world_size = test_class.world_size
            # Copy the world_size from test_class to override base_class's world_size
            neuron_test_class.world_size = test_class_world_size

        test_methods_set = set(config["test_methods"])
        all_test_methods = [
            name
            for name in dir(neuron_test_class)
            if name.startswith("test_") and callable(getattr(neuron_test_class, name))
        ]

        for method_name in all_test_methods:
            if method_name not in test_methods_set:

                def create_skip_method(name, original_method):
                    @pytest.mark.skip(reason=f"Method {name} not in filtered test methods")
                    def skipped_method(self):
                        pass

                    # Preserve __wrapped__ attribute if it exists on the original method
                    # This is needed for parametrized tests that access __wrapped__ in setUp
                    if hasattr(original_method, "__wrapped__"):
                        skipped_method.__wrapped__ = original_method.__wrapped__
                    else:
                        # Set __wrapped__ to the original method itself as fallback
                        skipped_method.__wrapped__ = original_method

                    return skipped_method

                original_method = getattr(neuron_test_class, method_name)
                setattr(
                    neuron_test_class, method_name, create_skip_method(method_name, original_method)
                )

        for xfail in config["xfail"]:
            method_name = next(iter(xfail.keys()))
            if hasattr(neuron_test_class, method_name):
                method = getattr(neuron_test_class, method_name)
                method = pytest.mark.xfail(reason=xfail[method_name]["reason"])(method)
                setattr(neuron_test_class, method_name, method)

        if not hasattr(test_class, "setUp") or not callable(test_class.setUp):
            needs_setup = True
        if not hasattr(test_class, "tearDown") or not callable(test_class.tearDown):
            needs_teardown = True

        if needs_setup:

            def setUp(self, bc=selected_base_class):
                os.environ["NEURON_RT_NUM_CORES"] = NEURON_RT_NUM_CORES
                super(bc, self).setUp()
                self._spawn_processes()

            neuron_test_class.setUp = setUp

        if needs_teardown:

            def tearDown(self, bc=selected_base_class):
                super(bc, self).tearDown()
                with suppress(OSError, AttributeError):
                    os.remove(self.file_name)

            neuron_test_class.tearDown = tearDown

        module_globals[neuron_class_name] = neuron_test_class


class NeuronMultiProcessTest(NeuronDistributedBase, MultiProcessTestCase):
    @property
    def device_type(self):
        return "neuron"

    @classmethod
    def _run(cls, rank, test_name, file_name, parent_pipe, **kwargs):
        super()._setup_neuron_environment(rank)
        cls._apply_patches()
        super()._run(rank, test_name, file_name, parent_pipe, **kwargs)


class NeuronMultiProcessTestDDP(NeuronDistributedBase, MultiProcessTestCase):
    @property
    def device_type(self):
        return "neuron"

    @classmethod
    def _run(cls, rank, test_name, file_name, parent_pipe, **kwargs):
        super()._setup_neuron_environment(rank, set_rt_visible_cores=True)
        cls._apply_patches()
        super()._run(rank, test_name, file_name, parent_pipe, **kwargs)


class NeuronCommonTest(NeuronDistributedBase):
    @property
    def device_type(self):
        return "neuron"

    @classmethod
    def _run(cls, rank, test_name, file_name, parent_pipe, **kwargs):
        """Called inside each spawned child process"""
        cls._setup_neuron_environment(rank, set_rt_visible_cores=True)
        cls._apply_patches()
        super()._run(rank, test_name, file_name, parent_pipe, **kwargs)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        with suppress(OSError, AttributeError):
            os.remove(self.file_name)


class NeuronCommonContTest(NeuronDistributedBase):
    @classmethod
    def _worker_loop(cls, rank, world_size, rdvz_file, task_queue, completion_queue):
        """Called inside MultiProcContinuousTest class to spawn process"""
        cls._setup_neuron_environment(rank, set_rt_visible_cores=True)
        cls._apply_patches()
        super()._worker_loop(rank, world_size, rdvz_file, task_queue, completion_queue)

    @classmethod
    def device_type(cls):
        return "neuron"

    @classmethod
    def backend_str(cls):
        return "neuron"


_BASE_CLASS_MAPPING = {
    MultiProcContinuousTest: NeuronCommonContTest,
    # TODO: Add more mappings as needed
}


def _select_neuron_base_class(test_class, default_base_class):
    """
    Select appropriate base class based on test class hierarchy
    Args:
        test_class: The original PT test class
        default_base_class: Fallback base class
    Returns:
        selected_base_class
    """

    parent_class_names = test_class.__mro__[1:]  ##Exclude self

    for parent_name in parent_class_names:
        if parent_name in _BASE_CLASS_MAPPING:
            return _BASE_CLASS_MAPPING[parent_name]

    return default_base_class
