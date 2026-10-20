import os

import torch
import torch.profiler
from torch.profiler import ProfilerActivity

from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

OUTPUT_DIR = "./profile_compiled_rmsnorm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = OUTPUT_DIR
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

device = torch.device("neuron")
x = torch.randn(4, 512, dtype=torch.bfloat16, device=device)
weight = torch.ones(512, dtype=torch.bfloat16, device=device)


@torch.compile(backend="neuron", fullgraph=True)
def naive_compiled(t):
    x_sq = t * t
    mean_sq = x_sq.mean(dim=-1, keepdim=True)
    rms = torch.rsqrt(mean_sq + 1e-6)
    return t * rms * weight


nc = NeuronConfig(
    modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
    profile_output_dir=OUTPUT_DIR,
    capture_enabled_for_nc="0",
    correlate_device_time=True,
)
exporter = NeuronProfiler(nc)

with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    experimental_config=nc,
    record_shapes=False,
    on_trace_ready=exporter.export_trace,
    schedule=torch.profiler.schedule(wait=4, warmup=1, active=1),
) as prof:
    for _ in range(6):
        with torch.no_grad(), torch.profiler.record_function("rmsnorm"):
            _ = naive_compiled(x)
        torch.neuron.synchronize()
        prof.step()

print(prof.key_averages().table(sort_by="self_privateuse1_time_total", row_limit=1))
