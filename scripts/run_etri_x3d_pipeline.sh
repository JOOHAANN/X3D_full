#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/home/youhan/ws/X3D_full"
readonly CONDA="/opt/miniforge3/bin/conda"
readonly ENV_NAME="clipgcn"
readonly RGB_ROOT="/home/youhan/ws/ETRI-Activity3D-RGB"
readonly DATA_DIR="${ROOT}/data/etri_rgb_c001_cs_70_10_20"
readonly OUTPUT_DIR="${ROOT}/outputs/x3d-s_clipgcn_tensor_cs_70_10_20_182"
readonly PRETRAINED="${ROOT}/pretrained/x3d_s.pyth"
readonly LOG_DIR="${ROOT}/logs/etri_x3d_20260906"

mkdir -p -- "$DATA_DIR" "$OUTPUT_DIR" "${ROOT}/pretrained" "$LOG_DIR"
cd -- "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date --iso-8601=seconds)] stage=preflight"
"$CONDA" run -n "$ENV_NAME" python -c \
  'import cv2, numpy, torch; print("python/torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'

if [[ ! -s "$PRETRAINED" ]]; then
  echo "[$(date --iso-8601=seconds)] stage=download_pretrained"
  curl --fail --location --retry 10 --continue-at - \
    --output "$PRETRAINED" \
    https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_s.pyth
fi

echo "[$(date --iso-8601=seconds)] stage=build_cache"
"$CONDA" run --no-capture-output -n "$ENV_NAME" python -u tools/build_etri_rgb_tensor.py \
  --rgb-root "$RGB_ROOT" \
  --output-dir "$DATA_DIR" \
  --camera C001 \
  --workers 24 \
  2>&1 | tee -a "$LOG_DIR/build_cache.log"

echo "[$(date --iso-8601=seconds)] stage=train"
"$CONDA" run --no-capture-output -n "$ENV_NAME" python -u tools/train_etri_x3d.py \
  --config configs/x3d-s_etri_rgb_cs_70_10_20_182.yaml \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --pretrained "$PRETRAINED" \
  --workers 8 \
  --prefetch-factor 2 \
  --max-iter 12000 \
  --save-step 500 \
  --eval-step 500 \
  --resume \
  2>&1 | tee -a "$LOG_DIR/train.log"

echo "[$(date --iso-8601=seconds)] stage=complete"
