#!/bin/bash
# One variant per TPU chip, all four in parallel.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-42}
NAMES=(rope absolute nope noqk)
VARIANTS=(
  "model.config.pos_embed=rope"
  "model.config.pos_embed=absolute"
  "model.config.pos_embed=none"
  "model.config.use_qk_norm=false"
)

for i in "${!VARIANTS[@]}"; do
  TPU_VISIBLE_DEVICES=$i uv run python -m tiny.train -cn config \
    model=3_17m total_steps=128000 seed=$SEED ${VARIANTS[$i]} \
    hydra.run.dir=outputs/ablations/${NAMES[$i]}_s$SEED &
done
wait
