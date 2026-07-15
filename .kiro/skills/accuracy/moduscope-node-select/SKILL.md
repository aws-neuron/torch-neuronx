---
name: moduscope-node-select
description: Find truly-free compute nodes before launching ModuScope component-test runs on the TRN (Neuron) cluster (TRN nodes]) and the GPU cluster (GPU nodes]) clusters. Use when asked to pick nodes, check which nodes are free, find idle GPU nodes, or check TRN occupancy before scaling test runs. Critical knowledge — GPU free-ness comes straight from Slurm (sinfo -t idle + squeue), but TRN nodes always show 'resv' (a reservation, NOT busy) so real occupancy MUST be probed by counting $USER python processes over ssh; 0 means free.
---

# ModuScope Node Selection

## Overview

Pick verified-free compute nodes before launching ModuScope component tests. GPU nodes are scheduled by Slurm so `sinfo -t idle` is authoritative; TRN nodes live in a reservation and always print `resv`, so freeness MUST be confirmed by counting `$USER` python processes over ssh (0 = free). Wrong node selection causes process contention and corrupted numerical results.

## Usage

Use this skill when:
- Asked to "find free nodes", "pick nodes", or "check node occupancy" before a run
- Selecting one node per (test × config) cell for a parallel ModuScope sweep
- Confirming a TRN node is actually idle even though Slurm shows it `resv`
- Choosing idle GPU nodes for `gpu_save` reference capture

## Core Concepts

### GPU side (the GPU cluster, partition `compute1`) — Slurm is the truth
Nodes `<gpu-node>]` are allocated via `srun`. There is no
personal reservation, so Slurm state is accurate:
- A node is free iff it is in state `idle` **and** has no `squeue` job.
- Probe from the GPU login host with `/opt/slurm/bin/sinfo -p compute1 -t idle`
  and cross-check `/opt/slurm/bin/squeue -p compute1`.

### TRN side (the TRN cluster, `<trn-node>]`) — `resv` is NOT busy
These nodes sit in a Slurm **reservation**, so `sinfo`/`squeue` show `resv` for all
of them regardless of real load. That state is EXPECTED and tells you nothing about
occupancy. You MUST probe each node directly:
```bash
ssh -i $KEY <node> "ps -fu $USER | grep python | grep -v grep | wc -l"
# 0  => free ;  >0 => a run is already using that node
```
`KEY=$SSH_KEY`. Reach nodes from the
`$LOGIN_HOST` bastion. The probe is read-only and safe to run repeatedly.

### One node per cell
ModuScope runs are CPU/accelerator-heavy single-process jobs. To avoid contention
and result corruption, each test cell MUST get its own node — never co-schedule two
runs on one node. The free-node list from this skill feeds directly into
`moduscope-parallel-orchestration`.

## Quick Reference

| Cluster | Where to run | Command | Free criterion |
|---|---|---|---|
| GPU | GPU login host | `bash scripts/find_free_gpu_nodes.sh` | state `idle` AND no squeue job |
| TRN | `$LOGIN_HOST` bastion | `bash scripts/find_free_trn_nodes.sh` | $USER python process count == 0 |

Both scripts print verified-free node names to **stdout** (one per line, capturable
into an array) and human-readable diagnostics to **stderr**.

```bash
# Capture free TRN nodes into an array for the orchestration skill:
mapfile -t FREE < <(bash scripts/find_free_trn_nodes.sh 1 16 2>/dev/null)
echo "free: ${FREE[*]}"
```

## Deterministic Scripts

### scripts/find_free_gpu_nodes.sh
Lists idle GPU nodes in `compute1` using `sinfo -t idle`, expands the Slurm
hostlist, and filters out any node that appears in `squeue`. Run on the GPU login
host. Optional arg: partition name (default `compute1`).

### scripts/find_free_trn_nodes.sh
Probes TRN nodes `<trn-node>]` (default 1..16) over
ssh, ignoring the misleading `resv` state, and prints nodes with zero `$USER`
python processes. Run from the `$LOGIN_HOST` bastion. Honors `KEY` env override.

## How to test this skill

Structural / syntax check (no cluster access required):
```bash
bash -n scripts/find_free_gpu_nodes.sh && \
bash -n scripts/find_free_trn_nodes.sh && \
echo "SYNTAX_OK"
```
Success criterion: prints `SYNTAX_OK` with exit code 0.

Live smoke test (only when on the respective cluster): running
`bash scripts/find_free_trn_nodes.sh 1 2` from the `$LOGIN_HOST` bastion prints a
`FREE`/`busy` line per probed node to stderr and a `== N free TRN node(s) ==`
footer — proving the `resv`-aware ps-probe works end to end.

## Common Mistakes

- **Trusting `resv` as busy on TRN.** It is a reservation marker, not occupancy.
  Always run the ps-probe.
- **Launching two cells on one node.** Causes contention and unreliable ratios —
  one node per cell.
- **Running the GPU script on TRN (or vice-versa).** `sinfo` lives only on the GPU
  login host for `compute1`; the ps-probe needs the TRN `$KEY`.
