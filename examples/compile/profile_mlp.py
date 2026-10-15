"""
Profile a torch.compile workload on Neuron using torch.profiler.

Run:
    python profile_mlp.py
"""

import os

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import torch_neuronx
from torch_neuronx.profiling import NeuronConfig, ProfileMode


class MLP(torch.nn.Module):
    def __init__(self, hidden: int = 4096):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden, hidden * 4, bias=False)
        self.fc2 = torch.nn.Linear(hidden * 4, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    torch.manual_seed(0)
    device = torch.device("neuron")

    model = MLP(hidden=4096).to(device, dtype=torch.bfloat16)
    compiled_model = torch.compile(model, backend="neuron", dynamic=False)
    x = torch.randn(8, 4096, device=device, dtype=torch.bfloat16)

    output_dir = "./profile_compile"
    os.makedirs(output_dir, exist_ok=True)

    exp_config = NeuronConfig(
        modes=[ProfileMode.RUNTIME],
        profile_output_dir=output_dir,
        max_events_per_nc=100000,
    )

    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            experimental_config=exp_config,
        ) as prof,
        record_function("compiled_model_inference"),
        torch.inference_mode(),
    ):
        for _ in range(3):  # warmup
            _ = compiled_model(x)
        torch_neuronx.synchronize()

    trace_file = os.path.join(output_dir, "neuron_framework_trace.json")
    prof.export_chrome_trace(trace_file)
    print(f"Chrome trace exported to {trace_file}")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=20, top_level_events_only=False
        )
    )


if __name__ == "__main__":
    main()
