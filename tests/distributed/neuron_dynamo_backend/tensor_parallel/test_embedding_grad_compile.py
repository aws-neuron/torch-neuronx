"""
Assert that an eager RowwiseParallel embedding still receives its weight
gradient when a downstream feed-forward block is torch.compile.

The compiled block's input is redistributed ``Shard(1) -> Replicate``, so it
receives an ``AsyncCollectiveTensor`` graph input. The test checks that
``tok_embeddings.weight.grad`` is present, nonzero, and matches a CPU reference.
"""

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    parallelize_module,
)

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest

VOCAB, HIDDEN, FFN, SEQ, BS = 512, 64, 128, 16, 1

TP_PLAN = {
    "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
    "ff": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
    "ff.w1": ColwiseParallel(),
    "ff.w3": ColwiseParallel(),
    "ff.w2": RowwiseParallel(output_layouts=Shard(1)),
}


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(HIDDEN, FFN, bias=False)
        self.w2 = nn.Linear(FFN, HIDDEN, bias=False)
        self.w3 = nn.Linear(HIDDEN, FFN, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(VOCAB, HIDDEN)
        self.ff = FeedForward()

    def forward(self, ids):
        return self.ff(self.tok_embeddings(ids))


def _make_model():
    torch.manual_seed(1)
    return Model()


def run_embedding_grad_compile_test(rank, world_size, kwargs):
    """RowwiseParallel embedding (eager) -> torch.compile'd FF block; the embedding
    weight gradient must survive compilation and match a CPU reference."""
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (BS, SEQ), dtype=torch.long)

    # CPU ground-truth gradient for the embedding weight.
    ref = _make_model()
    ref(ids).float().pow(2).sum().backward()
    ref_grad = ref.tok_embeddings.weight.grad.detach().float()
    assert ref_grad.abs().max() > 0, "CPU reference produced a zero embedding grad"

    mesh = init_device_mesh("neuron", (world_size,), mesh_dim_names=("tp",))

    model = _make_model().to(f"neuron:{rank}")
    parallelize_module(model, mesh, TP_PLAN)
    # Compile only the FF block, leaving the embedding eager (torchtitan pattern).
    model.ff = torch.compile(model.ff, backend="neuron", fullgraph=True)

    model(ids.to(f"neuron:{rank}")).float().pow(2).sum().backward()

    emb_grad = model.tok_embeddings.weight.grad
    # Sanity: the compiled block's own weights get grads, so a missing embedding
    # grad below reflects the input-grad path, not a broken setup.
    assert model.ff.w1.weight.grad is not None, "compiled block weight grad missing"

    assert emb_grad is not None, "tok_embeddings.weight.grad is None under torch.compile"
    # Embedding weight is sharded; gather to a full tensor to compare with the CPU ref.
    emb_grad = emb_grad.full_tensor() if hasattr(emb_grad, "full_tensor") else emb_grad
    emb_grad = emb_grad.detach().float().cpu()
    assert emb_grad.abs().max() > 0, "tok_embeddings.weight.grad is all zeros under torch.compile"

    torch.testing.assert_close(emb_grad, ref_grad, rtol=2e-2, atol=2e-2)


class TestEmbeddingGradCompile(BaseCollectiveOpTest):
    """Embedding weight gradient is preserved when a downstream block is compiled."""

    @pytest.mark.multi_device
    def test_embedding_grad_preserved_under_compile(self):
        self.distributed_tester.run_test(run_embedding_grad_compile_test)
