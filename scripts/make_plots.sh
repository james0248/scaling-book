#!/bin/bash
# Run every approach on both model families and regenerate all figures into plots/.
set -euo pipefail
cd "$(dirname "$0")/.."

DENSE_A1=outputs/approach_1
DENSE_A2=outputs/approach_2
MOE_A1=outputs/approach_1_moe
MOE_A2=outputs/approach_2_moe
A3_CUT=6.6e14

echo "=== dense ==="
uv run python -m tiny.analyze_1 --run_dir $DENSE_A1 --prefix dense/a1
# 3e14: vertex at the sampled-range edge; 1.8e15: concave -> dropped
uv run python -m tiny.analyze_2 --run_dir $DENSE_A2 --filter-range --exclude 3e14,1.8e15 --prefix dense/a2
uv run python -m tiny.analyze_3 $DENSE_A1 $DENSE_A2 --prefix dense/a3 --max_flops $A3_CUT

for VIEW in active_params total_params; do
    KEY=$([ "$VIEW" = active_params ] && echo params || echo total_params)
    echo
    echo "=== MoE ($VIEW) ==="
    uv run python -m tiny.analyze_1 --run_dir $MOE_A1 $MOE_A2 \
        --prefix moe/$VIEW/a1 --params_key "$KEY"
    uv run python -m tiny.analyze_3 $MOE_A1 $MOE_A2 \
        --prefix moe/$VIEW/a3 --params_key "$KEY" --max_flops $A3_CUT
done

echo
echo "=== MoE approach 2 (both param views) ==="
# 9e14 and 1.8e15: minima at the sampled-range edge -> dropped
uv run python -m tiny.analyze_2_moe --run_dir $MOE_A2 --out_root plots/moe --exclude 9e14,1.8e15

echo
echo "=== phase transition ==="
uv run python -m tiny.transition

echo
echo "=== ablations ==="
uv run python -m tiny.analyze_ablations

echo
echo "figures written:"
find plots -name '*.png' | sort
