# Run Recipes

Concrete recipes for each `(model × topology × mode × platform)` combination we've used. Copy-paste ready commands.

## Glossary
- **TS**: timestamp `$(date -u +%Y%m%d_%H%M%S)`
- **TAG**: unique run identifier embedded in log/dump paths
- **the TRN cluster ssh prefix**: `ssh -F $SSH_CONFIG $LOGIN_HOST`
- **GPU ssh prefix**: `ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com`

---

## llama3_8B 1N TRN eager apples
```bash
ssh -F $SSH_CONFIG $LOGIN_HOST "
TS=\$(date -u +%Y%m%d_%H%M%S)
TAG=llama3_8b_trn_eager_10k_\${TS}
DUMP=$SHARED/<neuron-env>/runs/llama3_8b_1n/eager_10k_\${TS}
SEED=$SHARED/llama3_8b_seed_checkpoint/step-0
ssh -i $SSH_KEY -o StrictHostKeyChecking=no <trn-node> \"
  setsid nohup bash $SHARED/scripts_10k/run_on_node_accuracy_moe_tim.sh \
    \${TAG} llama3_8b eager \${SEED} 16 \${DUMP} 10000 \\
    > $SHARED/logs/run_\${TAG}.bgout 2>&1 < /dev/null &
\"
echo TAG=\${TAG}
"
```

## llama3_8B 1N TRN compile apples
Same as eager, replace `eager` with `compile`. Compile adds ~3 min to startup but ~30% faster steady-state.

## llama3_8B 1N GPU eager apples
```bash
ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com "
TS=\$(date -u +%Y%m%d_%H%M%S)
TAG=llama3_8b_apples_gpu_eager_10k_\${TS}
DUMP=$SHARED/clean-may18/runs/llama3_8b_1n_apples/eager_10k_\${TS}
ssh -o StrictHostKeyChecking=no <gpu-node> \"
  setsid nohup bash $SHARED/scripts_10k/run_on_node_gpu_apples.sh \
    \${TAG} llama3_8b eager 16 \${DUMP} 10000 \\
    > $SHARED/logs/run_\${TAG}.bgout 2>&1 < /dev/null &
\"
echo TAG=\${TAG}
"
```

## llama3_8B 4N TRN eager apples (GBS=64)
```bash
ssh -F $SSH_CONFIG $LOGIN_HOST "
TS=\$(date -u +%Y%m%d_%H%M%S)
TAG=llama3_8b_4n_apples_trn_eager_10k_\${TS}
DUMP=$SHARED/<neuron-env>/runs/llama3_8b_4n_apples/eager_10k_\${TS}
SEED=$SHARED/llama3_8b_seed_checkpoint/step-0
PORT=29754
MASTER=<trn-node>
for i in 0 1 2 3; do
  NODE=\$(printf '<trn-node>' \$((1+i)))
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \$NODE \"
    setsid nohup bash $SHARED/scripts_10k/run_4n_apples_trn.sh \
      \${TAG} llama3 llama3_8b trn_eager \$i \${MASTER} \${PORT} \
      \${SEED} 64 \${DUMP} 10000 \\
      > $SHARED/logs/run_\${TAG}.bgout.rank\$i 2>&1 < /dev/null &
  \"
done
echo TAG=\${TAG}
"
```

## llama3_8B 4N TRN compile apples (GBS=64)
Same as 4N eager — change `trn_eager` → `trn_compile`. The launcher adds `--compile.enable --compile.backend=neuron`.

## llama3_8B 4N GPU eager apples (GBS=64)
```bash
ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com "
TS=\$(date -u +%Y%m%d_%H%M%S)
TAG=llama3_8b_4n_apples_gpu_eager_10k_\${TS}
DUMP=$SHARED/clean-may18/runs/llama3_8b_4n_apples/eager_10k_\${TS}
SEED=$SHARED/llama3_8b_seed_checkpoint/step-0
PORT=29755
MASTER=<gpu-node>
NODES=(1-13 1-14 1-15 1-16)
for i in 0 1 2 3; do
  NODE=\${NODES[\$i]}
  ssh -o StrictHostKeyChecking=no <gpu-node> \"
    setsid nohup bash $SHARED/scripts_10k/run_4n_apples_gpu.sh \
      \${TAG} llama3 llama3_8b gpu_eager \$i \${MASTER} \${PORT} \
      64 \${DUMP} 10000 \${SEED} \\
      > $SHARED/logs/run_\${TAG}.bgout.rank\$i 2>&1 < /dev/null &
  \"
done
echo TAG=\${TAG}
"
```

## llama3_70B 4N TRN eager (GBS=64, no seed checkpoint)
```bash
ssh -F $SSH_CONFIG $LOGIN_HOST "
TS=\$(date -u +%Y%m%d_%H%M%S)
TAG=llama3_70b_4n_gbs64_c4local_trn_eager_10k_\${TS}
DUMP=$SHARED/<neuron-env>/runs/llama3_70b_4n_gbs64_c4local/eager_10k_\${TS}
PORT=29762
MASTER=<trn-node>
for i in 0 1 2 3; do
  NODE=\$(printf '<trn-node>' \$((13+i)))
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \$NODE \"
    setsid nohup bash $SHARED/scripts_10k/run_4n_tim_70b_trn_v2.sh \
      \${TAG} \$i \${MASTER} \${PORT} 10000 \${DUMP} 64 1 '' \\
      > $SHARED/logs/run_\${TAG}.bgout.rank\$i 2>&1 < /dev/null &
  \"
done
echo TAG=\${TAG}
"
```

## llama3_70B 4N TRN compile (GBS=64)
Same — replace `''` (last arg) with `'--compile.enable --compile.backend=neuron'`.

## llama3_70B 8N GPU eager (GBS=64 LBS=4 grad_accum=1)
```bash
ssh -i ~/cgk-p5en.pem -o StrictHostKeyChecking=no ubuntu@ec2-108-137-91-71.ap-southeast-3.compute.amazonaws.com "
TS=\$(date -u +%Y%m%d_%H%M%S)
TAG=llama3_70b_8n_gbs64_lbs4_c4local_gpu_eager_10k_\${TS}
DUMP=$SHARED/clean-may18/runs/llama3_70b_8n_gbs64_c4local/eager_10k_lbs4_\${TS}
PORT=29764
MASTER=<gpu-node>
NODES=(1-4 1-6 1-7 1-8 1-9 1-10 1-11 1-12)
for i in 0 1 2 3 4 5 6 7; do
  NODE=\${NODES[\$i]}
  ssh -o StrictHostKeyChecking=no <gpu-node> \"
    setsid nohup bash $SHARED/scripts_10k/run_8n_tim_70b_gpu_v2.sh \
      \${TAG} \$i \${MASTER} \${PORT} 10000 \${DUMP} 64 4 \\
      > $SHARED/logs/run_\${TAG}.bgout.rank\$i 2>&1 < /dev/null &
  \"
done
echo TAG=\${TAG}
"
```

If LBS=4 OOMs, try LBS=2 (grad_accum=2) → LBS=1 (grad_accum=4). At H200 (140GB), LBS=4 uses only 36% memory so OOM very unlikely.

## qwen3_32B 1N
Use `run_on_node_accuracy_moe_tim.sh` (TRN) or `run_on_node_gpu_apples.sh` (GPU) with model name `qwen3_32b`. GBS=16 default. No seed checkpoint typically needed.

## Verifying after launch
After ~3 min:
```bash
LOG=$SHARED/logs/run_<TAG>.log.rank0
ssh -F $SSH_CONFIG $LOGIN_HOST "
grep 'pre-tokenizing\|pre-tokenized [0-9]' $LOG          # MUST show 893530397 c4_local
grep 'Trainer is initialized' $LOG | head -1              # MUST show warmup 10, GBS=64, seq 8192
grep -m2 'Compiling.*torch.compile' $LOG                  # only present if compile mode
"
```

After ~5 min, look for `step:  1  loss:` line; first-5-tokens at rank 0 should be `[2002, 66, 101, 103, 105]` for c4_local with seed=42.
