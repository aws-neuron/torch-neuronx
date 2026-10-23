"""Compare stock and opt-in loss gradients on CPU, Neuron, or TPU.

This runner checks small, replicated vocabularies. It does not test a fused
Pallas loss, a sharded loss, throughput, or a complete training workload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as functional
from stable_softmax import (
    stable_distillation_cross_entropy,
    stable_hard_cross_entropy,
)


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hard_inputs(classes, dtype):
    rows, targets = [], []
    for index, margin in enumerate((4.0, 5.5, 7.5, 15.0, 16.6, 18.0, 22.0)):
        target = (0, 1, classes - 1)[index % 3]
        row = torch.zeros(classes, dtype=torch.float64)
        row[target] = margin + torch.log(torch.tensor(classes - 1.0)).item()
        rows.append(row)
        targets.append(target)
    rows.extend(
        [
            torch.zeros(classes, dtype=torch.float64),
            torch.cat((torch.tensor([9.0]), torch.full((classes - 1,), -9.0))),
            rows[-1] + 32.0,
        ]
    )
    targets.extend((classes - 1, classes - 1, targets[-1]))
    return torch.stack(rows).to(dtype), torch.tensor(targets, dtype=torch.int64)


def _hard_reference(values, targets):
    z = values.double()
    selected = z.gather(-1, targets[:, None])
    mask = torch.arange(z.shape[-1]) == targets[:, None]
    shifted = (z - selected).masked_fill(mask, -torch.inf)
    # These sweep margins fit FP64. This is an independent analytic reference.
    mass = shifted.exp()
    denominator = 1.0 + mass.sum(-1, keepdim=True)
    gradient = mass / denominator
    gradient.scatter_(-1, targets[:, None], -gradient.sum(-1, keepdim=True))
    loss = torch.log1p(mass.sum(-1)).sum()
    return loss, gradient


def _metrics(loss, gradient, reference_loss, reference_gradient, rtol):
    actual = gradient.double()
    absolute = (actual - reference_gradient).abs()
    reference_mass = reference_gradient.abs().sum(-1)
    tiny = torch.finfo(torch.float64).tiny
    row_relative = absolute.sum(-1) / reference_mass.clamp_min(tiny)
    zero_reference = reference_mass == 0
    stationary = bool((actual[zero_reference] == 0).all())
    component_relative = torch.where(
        reference_gradient != 0,
        absolute / reference_gradient.abs().clamp_min(tiny),
        torch.zeros_like(absolute),
    )
    finite = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(loss))
    passed = finite and stationary and bool((row_relative <= rtol).all())
    return {
        "loss": float(loss),
        "reference_loss_fp64": float(reference_loss),
        "gradient": actual.tolist(),
        "reference_gradient_fp64": reference_gradient.tolist(),
        "absolute_gradient_error": absolute.tolist(),
        "component_relative_error_nonzero_reference": component_relative.tolist(),
        "row_l1_relative_error": row_relative.tolist(),
        "gradient_row_sum": actual.sum(-1).tolist(),
        "row_sum_over_gradient_mass": (
            actual.sum(-1).abs() / actual.abs().sum(-1).clamp_min(tiny)
        ).tolist(),
        "exact_stationarity_on_zero_reference_rows": stationary,
        "relative_error_threshold": rtol,
        "finite": finite,
        "gradient_check_passed": passed,
    }


def _synchronize(device, extension):
    if device.type == "neuron":
        extension.synchronize(device)
    elif device.type == "tpu":
        torch.accelerator.synchronize()


def _evaluate(fn, values, mode, args, device, extension):
    work = values.to(device).detach().requires_grad_(True)
    if device.type == "neuron":
        extension.clear_op_tracking()
    backend = {"cpu": "aot_eager", "neuron": "neuron", "tpu": "tpu"}[device.type]
    if mode == "compiled":
        fn = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
    loss = fn(work)
    gradient = torch.autograd.grad(loss, work)[0]
    _synchronize(device, extension)
    placement = {
        "input_device": str(work.device),
        "loss_device": str(loss.device),
        "gradient_device": str(gradient.device),
        "input_dtype": str(work.dtype),
        "loss_dtype": str(loss.dtype),
        "gradient_dtype": str(gradient.dtype),
    }
    if device.type == "neuron":
        tracking = {
            "fallback_ops": extension.get_fallback_ops(),
            "executed_ops": extension.get_executed_ops(),
            "qualification": "An empty executed list does not prove native execution.",
        }
    else:
        tracking = {"qualification": "Placement alone does not prove each operator ran natively."}
    placement["represented_inputs_match_cpu_fixture"] = torch.equal(work.detach().cpu(), values)
    return loss.detach().cpu().double(), gradient.detach().cpu(), placement, tracking


def run(args, result):
    extension = None
    if args.device == "neuron":
        for flag in (
            "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION",
            "TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY",
        ):
            if os.environ.get(flag) != "1":
                raise RuntimeError(f"Set {flag}=1 before starting Python.")
        import torch_neuronx

        extension = torch_neuronx
        if not torch.neuron.is_available():
            raise RuntimeError("No Neuron device is available.")
    elif args.device == "tpu":
        import torch_tpu

        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None or accelerator.type != "tpu" or not torch.accelerator.is_available():
            raise RuntimeError("No TPU device is available.")
        result["tpu_allow_excess_precision"] = bool(torch.tpu.allow_excess_precision)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    compute_dtype = torch.float32 if args.loss_fp32 else None
    torch.set_float32_matmul_precision(args.matmul_precision)
    modes = ("eager", "compiled") if args.mode == "both" else (args.mode,)
    rtol = args.rtol if args.rtol is not None else (0.05 if dtype == torch.bfloat16 else 0.0005)

    def context():
        if args.autocast:
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return nullcontext()

    result["checks"] = []
    for classes in (3, 33):
        values, targets = _hard_inputs(classes, dtype)
        stable_hard_cross_entropy(values, targets, compute_dtype=compute_dtype)
        reference_loss, reference_gradient = _hard_reference(values, targets)
        device_targets = targets.to(device)

        def stock_hard(x, targets=device_targets):
            with context():
                work = x.float() if args.loss_fp32 else x
                return functional.cross_entropy(work, targets, reduction="sum")

        def stable_hard(x, targets=device_targets):
            with context():
                return stable_hard_cross_entropy(
                    x,
                    targets,
                    reduction="sum",
                    compute_dtype=compute_dtype,
                    validate=False,
                )

        for mode in modes:
            for name, fn in (("stock", stock_hard), ("stable", stable_hard)):
                loss, gradient, placement, tracking = _evaluate(
                    fn, values, mode, args, device, extension
                )
                result["checks"].append(
                    {
                        "case": "hard_target_margin_sweep",
                        "classes": classes,
                        "implementation": name,
                        "mode": mode,
                        "represented_logits": values.double().tolist(),
                        "targets": targets.tolist(),
                        "placement": placement,
                        "operator_tracking": tracking,
                        **_metrics(loss, gradient, reference_loss, reference_gradient, rtol),
                    }
                )

    teacher = torch.tensor([[2.0, 1.0, 0.0]] * 3, dtype=dtype)
    student = teacher.clone()
    perturbation = 0.03125 if dtype == torch.bfloat16 else 0.00001
    student[1] += torch.tensor([-perturbation, perturbation, 0.0], dtype=dtype)
    student[2] += 1.0
    stable_distillation_cross_entropy(student, teacher, compute_dtype=compute_dtype)
    p64 = teacher.double().softmax(-1)
    q64 = student.double().softmax(-1)
    reference_gradient = q64 - p64
    reference_loss = -(p64 * student.double().log_softmax(-1)).sum()
    device_teacher = teacher.to(device)
    teacher_placement = {
        "device": str(device_teacher.device),
        "dtype": str(device_teacher.dtype),
        "represented_inputs_match_cpu_fixture": torch.equal(device_teacher.detach().cpu(), teacher),
    }

    def stock_distillation(x):
        with context():
            work = x.float() if args.loss_fp32 else x
            fixed_teacher = device_teacher.float() if args.loss_fp32 else device_teacher
            return -(fixed_teacher.softmax(-1) * work.log_softmax(-1)).sum()

    def stable_distillation(x):
        with context():
            return stable_distillation_cross_entropy(
                x,
                device_teacher,
                reduction="sum",
                compute_dtype=compute_dtype,
                validate=False,
            )

    for mode in modes:
        for name, fn in (("stock", stock_distillation), ("stable", stable_distillation)):
            loss, gradient, placement, tracking = _evaluate(
                fn, student, mode, args, device, extension
            )
            result["checks"].append(
                {
                    "case": "fixed_teacher_stationarity_and_near_equality",
                    "implementation": name,
                    "mode": mode,
                    "represented_teacher": teacher.double().tolist(),
                    "represented_student": student.double().tolist(),
                    "teacher_placement": teacher_placement,
                    "reference_objective": "Teacher logits define the probabilities.",
                    "stock_qualification": (
                        "Rounded target probabilities define a separate objective."
                    ),
                    "placement": placement,
                    "operator_tracking": tracking,
                    **_metrics(loss, gradient, reference_loss, reference_gradient, rtol),
                }
            )
    stable_checks = [row for row in result["checks"] if row["implementation"] == "stable"]
    result["stable_gradient_checks_passed"] = all(
        row["gradient_check_passed"] for row in stable_checks
    )
    result["represented_input_checks_passed"] = (
        all(row["placement"]["represented_inputs_match_cpu_fixture"] for row in result["checks"])
        and teacher_placement["represented_inputs_match_cpu_fixture"]
    )
    if device.type == "neuron":
        result["fallback_check_passed"] = all(
            not row["operator_tracking"]["fallback_ops"] for row in result["checks"]
        )
    else:
        result["fallback_check_passed"] = None
    result["status"] = "executed"
    passed = (
        result["stable_gradient_checks_passed"]
        and result["represented_input_checks_passed"]
        and result["fallback_check_passed"] is not False
    )
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "neuron", "tpu"), default="cpu")
    parser.add_argument("--mode", choices=("eager", "compiled", "both"), default="both")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--loss-fp32", action="store_true")
    parser.add_argument(
        "--matmul-precision", choices=("medium", "high", "highest"), default="medium"
    )
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rtol is not None and (args.rtol <= 0 or not torch.isfinite(torch.tensor(args.rtol))):
        parser.error("--rtol must be positive and finite.")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "not_executed",
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: _version(name)
            for name in ("torch", "torch-neuronx", "neuronx-cc", "torch-tpu", "libtpu")
        },
        "configuration": {key: str(value) for key, value in vars(args).items()},
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "TORCH_TPU_DEFER_AND_FUSE",
                "TORCH_TPU_LAUNCH_BLOCKING",
                "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION",
                "TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY",
            )
        },
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("stable_softmax.py", "check_loss_gradients.py")
        },
        "limits": [
            "Small replicated class-last vocabularies; no Pallas or sharded loss.",
            "No production convergence, performance, or training result.",
            "CPU aot_eager does not verify an accelerator compiler.",
            "FP64 references use the same represented inputs within their numeric range.",
        ],
    }
    try:
        exit_code = run(args, result)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(args.output.resolve())
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
