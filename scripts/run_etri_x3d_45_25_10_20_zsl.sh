#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/home/youhan/ws/X3D_full
readonly CONDA=/opt/miniforge3/bin/conda
readonly ENV_NAME=clipgcn
readonly SOURCE_DIR="$ROOT/data/etri_rgb_c001_cs_70_10_20"
readonly DATA_DIR="$ROOT/data/etri_rgb_c001_cs_45_25_10_20"
readonly OUTPUT_DIR="$ROOT/outputs/x3d-s_clipgcn_tensor_cs_45_25_10_20_zsl50_5_182"
readonly MANIFEST=/home/youhan/ws/ETRI_subject_split_45_25_10_20.json
readonly LOG_DIR="$ROOT/logs/etri_x3d_45_25_10_20_zsl_20260906"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date --iso-8601=seconds)] stage=repartition"
"$CONDA" run --no-capture-output -n "$ENV_NAME" python -u \
  tools/repartition_etri_cache_45_25_10_20.py \
  --source-dir "$SOURCE_DIR" --output-dir "$DATA_DIR" \
  --manifest "$MANIFEST" --seed 20260906 \
  2>&1 | tee -a "$LOG_DIR/repartition.log"

echo "[$(date --iso-8601=seconds)] stage=train_strict_zsl"
"$CONDA" run --no-capture-output -n "$ENV_NAME" python -u tools/train_etri_x3d.py \
  --config configs/x3d-s_etri_rgb_cs_45_25_10_20_zsl50_5_182.yaml \
  --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
  --pretrained "$ROOT/pretrained/x3d_s.pyth" \
  --exclude-classes 9 10 11 17 49 \
  --workers 8 --prefetch-factor 2 \
  --max-iter 7000 --warmup-iterations 235 --lr-steps 3500 5800 \
  --save-step 250 --eval-step 250 --resume \
  2>&1 | tee -a "$LOG_DIR/train.log"

echo "[$(date --iso-8601=seconds)] stage=complete"
