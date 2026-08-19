"""Replicate + pipeline-parallel composability.
Ports same test coverage from upstream with chip-aligned device mesh.
Reference upstream test: https://github.com/pytorch/pytorch/blob/22e3367eb5eecd6c56d4e91dd5d8600d8869b728/test/distributed/_composable/test_composability/test_pp_composability.py#L349
"""

import copy
import os

import pytest
import torch
import torch.nn as nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleSingle,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
)

from ..utils import DistributedTester

WORLD_SIZE = 8
PP_SIZE = 2

SCHEDULE_CLASSES = [
    ScheduleGPipe,
    Schedule1F1B,
    ScheduleInterleaved1F1B,
    ScheduleLoopedBFS,
    ScheduleInterleavedZeroBubble,
]
SCHEDULE_IDS = [cls.__name__ for cls in SCHEDULE_CLASSES]


def chip_aligned_layout(world_size: int = WORLD_SIZE, pp_size: int = PP_SIZE) -> torch.Tensor:
    """Rank layout for a ("replicate", "pp") mesh with chip-aligned replicate groups.

    Same shape and dim names as upstream, but the rank layout is transposed.
    This puts consecutive ranks in the same replicate group, so a replicate group
    is one whole chip and a pp group is 2 ranks.
    """
    replicate_size = world_size // pp_size
    return torch.arange(world_size).view(pp_size, replicate_size).T.contiguous()


def build_mesh() -> DeviceMesh:
    return DeviceMesh("neuron", chip_aligned_layout(), mesh_dim_names=("replicate", "pp"))


# MLP Layer -- same shape as the upstream test's MLPModule.
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


def run_replicate_pp(rank, world_size, kwargs):
    """Port of upstream ComposabilityTest.test_replicate_pp."""
    ScheduleClass = kwargs["schedule_cls"]  # noqa: N806
    MixedPrecisionParam = kwargs["param_dtype"]  # noqa: N806

    # torch_neuronx sets local_device_start_index to the local rank, so the
    # PyTorch device index for this process is the rank -- not 0, even though
    # only one core is visible. See NeuronDevice.cpp get_vnc_id().
    device = torch.device("neuron", rank)

    dim = 8
    num_microbatches = 8
    device_mesh = build_mesh()
    torch.manual_seed(42)
    dp_mesh = device_mesh["replicate"]
    pp_mesh = device_mesh["pp"]
    pp_group = device_mesh["pp"].get_group()

    # create "entire model"
    total_layers = 8
    full_model = nn.ModuleList([MLPModule(dim) for _ in range(total_layers)])
    ref_full_model = copy.deepcopy(full_model)

    # dummy loss needed just to force backwards to run in schedule step
    def loss_fn(y, target):
        return y.sum()

    # Apply DP to stage module
    def apply_replicate(partial_model):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=MixedPrecisionParam,
            reduce_dtype=torch.float32,
        )
        replicate_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        for layer_id in range(len(partial_model)):
            replicate(partial_model[layer_id], **replicate_config)
        return replicate(partial_model, **replicate_config)

    # Apply same precision to reference model (without replicate)
    def apply_same_precision(partial_model):
        if MixedPrecisionParam != torch.float32:
            partial_model = partial_model.to(dtype=MixedPrecisionParam)
        return partial_model

    n_virtual = 1 if issubclass(ScheduleClass, PipelineScheduleSingle) else 2
    num_stages = pp_group.size() * n_virtual
    layers_per_stage = total_layers // num_stages
    stages = []
    ref_stages = []
    for i in range(n_virtual):
        stage_idx = pp_group.rank() + pp_group.size() * i
        start_layer = stage_idx * layers_per_stage
        end_layer = start_layer + layers_per_stage
        # divide the model layers by the number of stages
        partial_model = nn.Sequential(*full_model[start_layer:end_layer])
        partial_model.to(device)

        ref_partial_model = nn.Sequential(*ref_full_model[start_layer:end_layer])
        ref_partial_model.to(device)

        dp_model = apply_replicate(partial_model)
        ref_dp_model = apply_same_precision(ref_partial_model)

        stages.append(PipelineStage(dp_model, stage_idx, num_stages, device, group=pp_group))
        ref_stages.append(
            PipelineStage(ref_dp_model, stage_idx, num_stages, device, group=pp_group)
        )

        partial_models = [pipeline_stage.submod for pipeline_stage in stages]
        ref_partial_models = [pipeline_stage.submod for pipeline_stage in ref_stages]

    if issubclass(ScheduleClass, PipelineScheduleSingle):
        stages = stages[0]
        ref_stages = ref_stages[0]

    schedule_kwargs = {
        "n_microbatches": num_microbatches,
        "loss_fn": loss_fn,
        "scale_grads": False,
    }
    pipeline_schedule = ScheduleClass(stages, **schedule_kwargs)
    ref_pipeline_schedule = ScheduleClass(ref_stages, **schedule_kwargs)

    optimizer_kwargs = {
        "lr": 0.01,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": False,
        "foreach": True,
    }
    optimizers = [
        torch.optim.AdamW(model.parameters(), **optimizer_kwargs) for model in partial_models
    ]
    ref_optimizers = [
        torch.optim.AdamW(model.parameters(), **optimizer_kwargs) for model in ref_partial_models
    ]

    for _train_step in range(5):
        for optimizer in optimizers:
            optimizer.zero_grad()
        for ref_optimizer in ref_optimizers:
            ref_optimizer.zero_grad()

        inputs = torch.rand((num_microbatches, dim), device=device, dtype=MixedPrecisionParam)
        labels = torch.rand((num_microbatches, dim), device=device, dtype=MixedPrecisionParam)
        is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1
        if pp_mesh.get_local_rank() == 0:
            pipeline_schedule.step(inputs)
            ref_pipeline_schedule.step(inputs)
        elif is_last_stage:
            losses = []
            ref_losses = []
            pipeline_schedule.step(target=labels, losses=losses)
            ref_pipeline_schedule.step(target=labels, losses=ref_losses)

            for loss, ref_loss in zip(losses, ref_losses, strict=True):
                torch.testing.assert_close(loss, ref_loss)
        else:
            pipeline_schedule.step()
            ref_pipeline_schedule.step()

        for optimizer in optimizers:
            optimizer.step()
        for ref_optimizer in ref_optimizers:
            ref_optimizer.step()


def run_replicate_pp_grads(rank, world_size, kwargs):
    """Port of upstream ComposabilityTest.test_replicate_pp_grads."""
    ScheduleClass = kwargs["schedule_cls"]  # noqa: N806
    device = torch.device("neuron", rank)

    dim = 8
    num_microbatches = 8
    device_mesh = build_mesh()
    torch.manual_seed(42)
    dp_mesh = device_mesh["replicate"]
    pp_mesh = device_mesh["pp"]
    pp_group = device_mesh["pp"].get_group()
    dp_group = device_mesh["replicate"].get_group()

    # create "entire model"
    total_layers = 8
    full_model = nn.ModuleList([MLPModule(dim) for _ in range(total_layers)])
    ref_model = nn.Sequential(*copy.deepcopy(full_model)).to(device)

    # dummy loss needed just to force backwards to run in schedule step
    def loss_fn(y, target):
        return y.sum()

    def simulate_stage_forward_backward(model, inputs, labels):
        """Simulate forward and backward passes through stages for microbatch processing"""
        batch_size, _ = inputs.shape
        total_loss = 0
        microbatch_size = batch_size // num_microbatches

        for mb_idx in range(num_microbatches):
            start_idx = mb_idx * microbatch_size
            end_idx = start_idx + microbatch_size
            mb_input = inputs[start_idx:end_idx]
            mb_label = labels[start_idx:end_idx] if labels is not None else None

            if issubclass(ScheduleClass, PipelineScheduleSingle):
                num_stages = pp_group.size()
                layers_per_stage = total_layers // pp_group.size()
            else:
                n_virtual = 2
                num_stages = pp_group.size() * n_virtual
                layers_per_stage = total_layers // num_stages

            x = mb_input
            for stage in range(num_stages):
                start_layer = stage * layers_per_stage
                end_layer = start_layer + layers_per_stage
                for layer_idx in range(start_layer, min(end_layer, len(model))):
                    x = model[layer_idx](x)

            mb_loss = loss_fn(x, mb_label)
            total_loss += mb_loss
            mb_loss.backward()

        return total_loss / num_microbatches

    def apply_replicate(partial_model):
        for layer_id in range(len(partial_model)):
            replicate(partial_model[layer_id], mesh=dp_mesh)
        return replicate(partial_model, mesh=dp_mesh)

    def pipelined_models_parameters(start_layer, model):
        layer_idx = start_layer
        for layer in model.children():
            for name, param in layer.named_parameters():
                pipeline_model_parameter_dict[f"{layer_idx}.{name}"] = param
            layer_idx += 1

    def check_gradient_parity(pipeline_model_parameter_dict, ref_model_parameter_dict):
        for parameter in pipeline_model_parameter_dict:
            if parameter not in ref_model_parameter_dict:
                raise AssertionError(f"Parameter {parameter} not found in ref_model_parameter_dict")

            pipeline_parameter = pipeline_model_parameter_dict[parameter]
            if pipeline_parameter.grad is not None:
                pipeline_parameter_grad = pipeline_parameter.grad.to_local()
                ref_parameter = ref_model_parameter_dict[parameter]
                if ref_parameter.grad is not None:
                    torch.testing.assert_close(
                        pipeline_parameter_grad,
                        ref_parameter.grad,
                        rtol=1e-4,
                        atol=1e-5,
                    )
                else:
                    raise AssertionError(
                        f"Expected pipeline_parameter.grad to be None for {parameter}"
                    )

    pipeline_model_parameter_dict = {}

    n_virtual = 1 if issubclass(ScheduleClass, PipelineScheduleSingle) else 2
    num_stages = pp_group.size() * n_virtual
    layers_per_stage = total_layers // num_stages
    stages = []
    for i in range(n_virtual):
        stage_idx = pp_group.rank() + pp_group.size() * i
        start_layer = stage_idx * layers_per_stage
        end_layer = start_layer + layers_per_stage
        # divide the model layers by the number of stages
        partial_model = nn.Sequential(*full_model[start_layer:end_layer])
        partial_model.to(device)

        dp_model = apply_replicate(partial_model)
        pipelined_models_parameters(start_layer, dp_model)
        stages.append(PipelineStage(dp_model, stage_idx, num_stages, device, group=pp_group))
        partial_models = [pipeline_stage.submod for pipeline_stage in stages]

    if issubclass(ScheduleClass, PipelineScheduleSingle):
        stages = stages[0]

    pipeline_schedule = ScheduleClass(
        stages,
        n_microbatches=num_microbatches,
        loss_fn=loss_fn,
        scale_grads=False,
    )

    optimizer_kwargs = {
        "lr": 0.01,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": False,
        "foreach": True,
    }
    optimizers = [
        torch.optim.AdamW(model.parameters(), **optimizer_kwargs) for model in partial_models
    ]
    ref_optimizer = torch.optim.AdamW(ref_model.parameters(), **optimizer_kwargs)

    def simulate_all_reduce_grads(model, group):
        """Simulate all-reduce operation on gradients like replicate does"""
        for param in model.parameters():
            if param.grad is not None:
                # Scale by the number of replicas (like replicate does)
                param.grad.div_(group.size())
                torch.distributed.all_reduce(param.grad, group=group)

    ref_model_parameter_dict = dict(ref_model.named_parameters())

    torch.manual_seed(42 + rank)
    for _ in range(5):
        for optimizer in optimizers:
            optimizer.zero_grad()
        ref_optimizer.zero_grad()

        inputs = torch.rand((num_microbatches, dim), device=device)
        labels = torch.rand((num_microbatches, dim), device=device)

        # Ensure all ranks use the same inputs/labels for comparison
        torch.distributed.broadcast(inputs, 0)
        torch.distributed.broadcast(labels, 0)

        is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1
        losses = []

        if pp_mesh.get_local_rank() == 0:
            pipeline_schedule.step(inputs)
        elif is_last_stage:
            pipeline_schedule.step(target=labels, losses=losses)
        else:
            pipeline_schedule.step()

        if is_last_stage:
            ref_loss = simulate_stage_forward_backward(ref_model, inputs, labels)
            simulate_all_reduce_grads(ref_model, dp_group)

            if len(losses) > 0:
                avg_pipeline_loss = sum(losses) / len(losses)
                torch.testing.assert_close(avg_pipeline_loss, ref_loss, rtol=1e-4, atol=1e-5)
        else:
            simulate_stage_forward_backward(ref_model, inputs, None)
            simulate_all_reduce_grads(ref_model, dp_group)

        for optimizer in optimizers:
            optimizer.step()
        ref_optimizer.step()

        check_gradient_parity(pipeline_model_parameter_dict, ref_model_parameter_dict)


class TestReplicatePPChipAligned:
    """Upstream's replicate+PP tests on a mesh whose groups are chip-aligned."""

    @pytest.fixture(autouse=True)
    def process_index(self, worker_id):
        return int(worker_id[2:]) if worker_id != "master" else 0

    @pytest.fixture(autouse=True)
    def distributed_tester(self, process_index):
        rt_port = int(os.environ.get("NEURON_RT_PORT", "2025"))
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"localhost:{rt_port + process_index}"
        self.distributed_tester = DistributedTester(
            world_size=WORLD_SIZE, process_index=process_index
        )
        return self.distributed_tester

    @pytest.mark.parametrize("schedule_cls", SCHEDULE_CLASSES, ids=SCHEDULE_IDS)
    @pytest.mark.parametrize(
        "param_dtype", [torch.bfloat16, torch.float32], ids=["bfloat16", "float32"]
    )
    def test_replicate_pp(self, schedule_cls, param_dtype):
        self.distributed_tester.run_test(
            run_replicate_pp,
            schedule_cls=schedule_cls,
            param_dtype=param_dtype,
        )

    @pytest.mark.parametrize("schedule_cls", SCHEDULE_CLASSES, ids=SCHEDULE_IDS)
    def test_replicate_pp_grads(self, schedule_cls):
        self.distributed_tester.run_test(run_replicate_pp_grads, schedule_cls=schedule_cls)
