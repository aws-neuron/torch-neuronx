"""
8-Layer MLP Model with TP x FSDP Sharding and Multi-Stream Execution

Configuration:
- WORLD_SIZE workers total
- TP (Tensor Parallelism): TP_SIZE-way, batch-sharded activations
- FSDP (Fully Sharded Data Parallelism): FSDP_SIZE-way weight sharding
- hidden_dim: HIDDEN_DIM, intermediate_dim: INTERMEDIATE_DIM
- Two streams: default (compute + TP collectives), FSDP stream (weight allgather)
- Event-based synchronization between streams

TP Groups (FSDP_SIZE groups of TP_SIZE): ranks grouped consecutively
FSDP Groups (TP_SIZE groups of FSDP_SIZE): ranks strided by TP_SIZE
"""

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional

import torch_neuronx

# =============================================================================
# Configuration
# =============================================================================
WORLD_SIZE = 16
TP_SIZE = 4
FSDP_SIZE = WORLD_SIZE // TP_SIZE  # Dynamically compute FSDP size
NUM_LAYERS = 8
HIDDEN_DIM = 8 * 1024
INTERMEDIATE_DIM = 4 * HIDDEN_DIM
SEQ_LEN = 4 * 1024  # Total sequence length (will be divided by TP_SIZE)


def get_tp_rank(global_rank):
    """Get TP rank within the TP group (0-3)."""
    return global_rank % TP_SIZE


def get_fsdp_rank(global_rank):
    """Get FSDP rank within the FSDP group (0-15)."""
    return global_rank // TP_SIZE


def get_tp_group_id(global_rank):
    """Get which TP group this rank belongs to (0-15)."""
    return global_rank // TP_SIZE


def get_fsdp_group_id(global_rank):
    """Get which FSDP group this rank belongs to (0-3)."""
    return global_rank % TP_SIZE


def create_process_groups(rank):
    """
    Create TP and FSDP process groups.

    TP groups: [[0,1,2,3], [4,5,6,7], ..., [60,61,62,63]]
    FSDP groups: [[0,4,8,...,60], [1,5,9,...,61], [2,6,10,...,62], [3,7,11,...,63]]
    """
    # Create TP groups (FSDP_SIZE groups, each with TP_SIZE consecutive ranks)
    tp_groups = []
    for i in range(FSDP_SIZE):  # FSDP_SIZE TP groups
        tp_ranks = list(range(i * TP_SIZE, (i + 1) * TP_SIZE))
        tp_groups.append(tp_ranks)

    # Create FSDP groups (TP_SIZE groups, each with FSDP_SIZE ranks strided by TP_SIZE)
    fsdp_groups = []
    for i in range(TP_SIZE):  # TP_SIZE FSDP groups
        fsdp_ranks = list(range(i, WORLD_SIZE, TP_SIZE))
        fsdp_groups.append(fsdp_ranks)

    # Create actual process groups
    my_tp_group = None
    my_fsdp_group = None

    for tp_ranks in tp_groups:
        group = dist.new_group(ranks=tp_ranks)
        if rank in tp_ranks:
            my_tp_group = group

    for fsdp_ranks in fsdp_groups:
        group = dist.new_group(ranks=fsdp_ranks)
        if rank in fsdp_ranks:
            my_fsdp_group = group

    return my_tp_group, my_fsdp_group


class MLPLayer:
    """
    Single MLP layer with TP x FSDP sharding.

    Forward flow:
    1. FSDP Stream: AllGather W1 and W2 shards
    2. Event sync: Wait for FSDP allgather to complete
    3. Default Stream: TP AllGather activation -> MatMul1 -> GeLU -> MatMul2 -> TP ReduceScatter

    Weight shapes (per TP rank):
    - W1: [HIDDEN_DIM, INTERMEDIATE_DIM/TP_SIZE]
    - W1 FSDP-sharded: [HIDDEN_DIM/FSDP_SIZE, INTERMEDIATE_DIM/TP_SIZE]
    - W2: [INTERMEDIATE_DIM/TP_SIZE, HIDDEN_DIM]
    - W2 FSDP-sharded: [INTERMEDIATE_DIM/TP_SIZE/FSDP_SIZE, HIDDEN_DIM]
    """

    def __init__(self, layer_idx, rank, tp_group, fsdp_group, device):
        self.layer_idx = layer_idx
        self.rank = rank
        self.tp_group = tp_group
        self.fsdp_group = fsdp_group
        self.device = device
        self.tp_rank = get_tp_rank(rank)
        self.fsdp_rank = get_fsdp_rank(rank)

        # Dimensions
        self.hidden_dim = HIDDEN_DIM
        self.intermediate_dim_per_tp = INTERMEDIATE_DIM // TP_SIZE  # INTERMEDIATE_DIM/TP_SIZE

        # FSDP-sharded weight dimensions
        self.w1_shard_rows = HIDDEN_DIM // FSDP_SIZE  # HIDDEN_DIM/FSDP_SIZE
        self.w2_shard_rows = (
            self.intermediate_dim_per_tp // FSDP_SIZE
        )  # intermediate_dim_per_tp/FSDP_SIZE

        # Initialize FSDP-sharded weights
        # W1: shard along hidden_dim (rows)
        # Shape: [HIDDEN_DIM/FSDP_SIZE, INTERMEDIATE_DIM/TP_SIZE]
        self.w1_shard = (
            torch.randn(
                self.w1_shard_rows, self.intermediate_dim_per_tp, device=device, dtype=torch.float32
            )
            * 0.01
        )

        # W2: shard along intermediate_dim (rows)
        # Shape: [INTERMEDIATE_DIM/TP_SIZE/FSDP_SIZE, HIDDEN_DIM]
        self.w2_shard = (
            torch.randn(self.w2_shard_rows, self.hidden_dim, device=device, dtype=torch.float32)
            * 0.01
        )

        # Pre-allocate buffers for full weights after FSDP allgather
        # This enables using all_gather_into_tensor which avoids stack/copy overhead
        self.w1_full = torch.zeros(
            HIDDEN_DIM, self.intermediate_dim_per_tp, device=device, dtype=torch.float32
        )
        self.w2_full = torch.zeros(
            self.intermediate_dim_per_tp, self.hidden_dim, device=device, dtype=torch.float32
        )

    def fsdp_allgather_w1(self, fsdp_stream, event_w1):
        """
        AllGather W1 across FSDP group on the FSDP stream.

        W1: [HIDDEN_DIM/FSDP_SIZE, INTERMEDIATE_DIM/TP_SIZE] -> AllGather dim=0
            -> [HIDDEN_DIM, INTERMEDIATE_DIM/TP_SIZE]
        Records event_w1 after completion.

        Uses all_gather_into_tensor for efficiency (single output tensor, no stack/copy overhead).
        """
        with torch.neuron.stream(fsdp_stream):
            # AllGather W1 directly into pre-allocated buffer
            dist.all_gather_into_tensor(self.w1_full, self.w1_shard, group=self.fsdp_group)

            # Record event after W1 allgather completes
            event_w1.record(fsdp_stream)

    def fsdp_allgather_w2(self, fsdp_stream, event_w2):
        """
        AllGather W2 across FSDP group on the FSDP stream.

        W2: [INTERMEDIATE_DIM/TP_SIZE/FSDP_SIZE, HIDDEN_DIM] -> AllGather dim=0
            -> [INTERMEDIATE_DIM/TP_SIZE, HIDDEN_DIM]
        Records event_w2 after completion.

        Uses all_gather_into_tensor for efficiency (single output tensor, no stack/copy overhead).
        """
        with torch.neuron.stream(fsdp_stream):
            # AllGather W2 directly into pre-allocated buffer
            dist.all_gather_into_tensor(self.w2_full, self.w2_shard, group=self.fsdp_group)

            # Record event after W2 allgather completes
            event_w2.record(fsdp_stream)

    def forward(self, x, default_stream, fsdp_stream, event_w1, event_w2):
        """
        Forward pass for MLP layer with pipelined weight gathering.

        Args:
            x: Input activation [seq/TP, hidden_dim]
            default_stream: Stream for compute and TP collectives
            fsdp_stream: Stream for FSDP allgather
            event_w1: Event to synchronize W1 allgather -> MatMul1
            event_w2: Event to synchronize W2 allgather -> MatMul2

        Returns:
            output: [seq/TP, hidden_dim]

        Timeline (optimized - TP AllGather overlaps with FSDP AllGather):
            FSDP Stream:    [AllGather W1]──record(evt_w1)──[AllGather W2]──record(evt_w2)──►
                                    │                                  │
                                    │                                  ▼ wait(evt_w2)
            Default Stream: [TP AllGather]──────wait(evt_w1)─►[MatMul1]─►[GeLU]─►[MatMul2]─►[TP RS]

            Overlap: TP AllGather runs in parallel with FSDP AllGather W1 + W2
        """
        # Step 1: Start FSDP AllGather for W1 (on FSDP stream)
        self.fsdp_allgather_w1(fsdp_stream, event_w1)

        # Step 2: Start FSDP AllGather for W2 (on FSDP stream, after W1)
        # This will overlap with TP AllGather + MatMul1 + GeLU on default stream
        self.fsdp_allgather_w2(fsdp_stream, event_w2)

        # Step 3: TP AllGather activation (on default stream, in parallel with FSDP AllGather)
        # x: [SEQ_LEN/TP_SIZE, HIDDEN_DIM] -> [SEQ_LEN, HIDDEN_DIM]
        # This does NOT depend on W1, so it can run in parallel with FSDP AllGather W1
        x_gathered_list = [torch.zeros_like(x) for _ in range(TP_SIZE)]
        dist.all_gather(x_gathered_list, x, group=self.tp_group)
        x_full = torch.cat(x_gathered_list, dim=0)  # [SEQ_LEN, HIDDEN_DIM]

        # Step 4: Wait for W1 allgather before starting MatMul1
        # At this point, both TP AllGather and FSDP AllGather W1 should be done
        default_stream.wait_event(event_w1)

        # Step 5: MatMul1 (column-parallel)
        # x_full: [SEQ_LEN, HIDDEN_DIM] @ W1: [HIDDEN_DIM, INTERMEDIATE_DIM/TP_SIZE]
        #   -> [SEQ_LEN, INTERMEDIATE_DIM/TP_SIZE]
        hidden = torch.matmul(x_full, self.w1_full)

        # Step 6: GeLU activation
        hidden = functional.gelu(hidden)

        # Step 7: Wait for W2 allgather before MatMul2
        default_stream.wait_event(event_w2)

        # Step 8: MatMul2 (row-parallel)
        # hidden: [SEQ_LEN, INTERMEDIATE_DIM/TP_SIZE] @ W2: [INTERMEDIATE_DIM/TP_SIZE, HIDDEN_DIM]
        #   -> [SEQ_LEN, HIDDEN_DIM]
        output_full = torch.matmul(hidden, self.w2_full)

        # Step 9: TP ReduceScatter (batch dim)
        # output_full: [SEQ_LEN, HIDDEN_DIM] -> [SEQ_LEN/TP_SIZE, HIDDEN_DIM] per rank
        # First reshape for reduce_scatter: split into TP_SIZE chunks
        output_chunks = output_full.chunk(TP_SIZE, dim=0)
        output_local = torch.zeros(
            SEQ_LEN // TP_SIZE, HIDDEN_DIM, device=self.device, dtype=torch.float32
        )
        dist.reduce_scatter(output_local, list(output_chunks), group=self.tp_group)

        return output_local


class MLP8Layer:
    """NUM_LAYERS-layer MLP model with TP x FSDP sharding."""

    def __init__(self, rank, tp_group, fsdp_group, device):
        self.rank = rank
        self.tp_group = tp_group
        self.fsdp_group = fsdp_group
        self.device = device

        # Create NUM_LAYERS MLP layers
        self.layers = [MLPLayer(i, rank, tp_group, fsdp_group, device) for i in range(NUM_LAYERS)]

    def forward(self, x, default_stream, fsdp_stream):
        """
        Forward pass through all 8 layers.

        Args:
            x: Input [seq/TP, hidden_dim]
            default_stream: Stream for compute
            fsdp_stream: Stream for FSDP collectives

        Returns:
            output: [seq/TP, hidden_dim]
        """
        # Create sync events for each layer (2 events per layer for W1 and W2)
        events_w1 = [torch.neuron.Event() for _ in range(NUM_LAYERS)]
        events_w2 = [torch.neuron.Event() for _ in range(NUM_LAYERS)]

        for i, layer in enumerate(self.layers):
            x = layer.forward(x, default_stream, fsdp_stream, events_w1[i], events_w2[i])

        return x


def run_mlp_forward(rank, world_size):
    """Main function to run on each worker."""
    # Initialize distributed environment
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)

    # Initialize process group
    dist.init_process_group("neuron", rank=rank, world_size=world_size)

    # Create TP and FSDP process groups
    tp_group, fsdp_group = create_process_groups(rank)

    # Get rank info
    tp_rank = get_tp_rank(rank)
    fsdp_rank = get_fsdp_rank(rank)

    print(f"[Rank {rank}] TP rank: {tp_rank}, FSDP rank: {fsdp_rank}")

    # Create streams
    default_stream = torch.neuron.default_stream()
    fsdp_stream = torch.neuron.Stream()

    print(f"[Rank {rank}] Default stream: {default_stream}")
    print(f"[Rank {rank}] FSDP stream: {fsdp_stream}")

    # Create model
    device = "neuron"
    model = MLP8Layer(rank, tp_group, fsdp_group, device)

    # Create input tensor (batch-sharded)
    # Each TP rank has [seq/TP, hidden]
    seq_per_rank = SEQ_LEN // TP_SIZE
    x = torch.randn(seq_per_rank, HIDDEN_DIM, device=device, dtype=torch.float32)

    print(f"[Rank {rank}] Input shape: {x.shape}")

    # Timing events
    start_event = torch.neuron.Event(enable_timing=True)
    end_event = torch.neuron.Event(enable_timing=True)

    # Run forward pass
    print(f"[Rank {rank}] Starting forward pass...")
    start_event.record(default_stream)

    output = model.forward(x, default_stream, fsdp_stream)

    end_event.record(default_stream)

    # Synchronize streams
    default_stream.synchronize()
    fsdp_stream.synchronize()

    # Calculate elapsed time
    elapsed_time = start_event.elapsed_time(end_event)

    # Print results
    print(f"\n[Rank {rank}] ===== Results =====")
    print(f"[Rank {rank}] Output shape: {output.shape}")
    print(f"[Rank {rank}] Output mean: {output.mean().item():.6f}")
    print(f"[Rank {rank}] Forward pass time: {elapsed_time:.2f} ms")

    # Verify output shape
    expected_shape = (seq_per_rank, HIDDEN_DIM)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    print(f"[Rank {rank}] Verification passed!")

    dist.destroy_process_group()


def main():
    print(f"8-Layer MLP with TP{TP_SIZE} x FSDP{FSDP_SIZE} on {WORLD_SIZE} workers")
    print("=" * 70)
    print("Configuration:")
    print(f"  - hidden_dim: {HIDDEN_DIM}")
    print(f"  - intermediate_dim: {INTERMEDIATE_DIM}")
    print(f"  - num_layers: {NUM_LAYERS}")
    print(f"  - seq_len: {SEQ_LEN}")
    print(f"  - TP size: {TP_SIZE}")
    print(f"  - FSDP size: {FSDP_SIZE}")
    print("=" * 70)
    print("Streams:")
    print("  - Default stream: Compute + TP collectives")
    print("  - FSDP stream: Weight AllGather")
    print("=" * 70)
    print("Event synchronization: FSDP stream -> Default stream per layer")
    print("=" * 70)

    mp.spawn(run_mlp_forward, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
