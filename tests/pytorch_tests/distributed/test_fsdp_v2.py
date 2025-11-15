import types
from typing import Any

import torch_neuronx
from tests.pytorch_tests.distributed.common_distributed import create_test_classes
from tests.pytorch_tests.distributed.neuron_patch import NeuronPatcher, PatchName, patch_common_fsdp
from tests.pytorch_tests.distributed.test_fsdp import NeuronFSDPTest


def override_world_size(globals_dict: dict[str, Any], world_size: int = 2) -> None:
    """Override world_size property for all test classes."""
    for _, obj in list(globals_dict.items()):
        # __bases__ is to ensure that we only overide when this is a TestClass
        if hasattr(obj, "world_size") and hasattr(obj, "__bases__"):
            obj.world_size = property(lambda self, ws=world_size: ws)


def patch_test_train_parity_multi_group_device_assert() -> None:
    """
    Patch device type assertions to include specified device in
    _train_parity_multi_group_device_assert in test_fully_shard_training.py

    Assertion line: assert test_device_type in ("cuda", "hpu", "xpu", "cpu"),
    f"{test_device_type}"
    """
    # Patch the created test classes
    method_name = "_test_train_parity_multi_group"
    neuron_device = "neuron"
    for name, obj in list(globals().items()):
        if name.startswith("TestNeuron") and hasattr(obj, method_name):
            method = getattr(obj, method_name)
            if hasattr(method, "__code__"):
                consts = list(method.__code__.co_consts)
                for i, const in enumerate(consts):
                    if isinstance(const, tuple) and "cuda" in const and neuron_device not in const:
                        consts[i] = (*const, neuron_device)
                new_code = method.__code__.replace(co_consts=tuple(consts))
                new_func = types.FunctionType(
                    new_code,
                    method.__globals__,
                    method.__name__,
                    method.__defaults__,
                    method.__closure__,
                )
                setattr(obj, method_name, new_func)


class NeuronFSDPv2Test(NeuronFSDPTest):
    @classmethod
    def _apply_patches(cls, exclude: list[str] | None = None):
        if exclude is None:
            exclude = [PatchName.TENSOR_PATCHES.name]
        else:
            exclude += [PatchName.TENSOR_PATCHES.name]
        patches = NeuronPatcher.collect_patches()
        for p_name, p in patches.items():
            if p_name not in exclude:
                p.start()


patch_common_fsdp()
create_test_classes(NeuronFSDPv2Test, globals(), "fsdp_v2_spec.json")
override_world_size(globals())
patch_test_train_parity_multi_group_device_assert()


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
