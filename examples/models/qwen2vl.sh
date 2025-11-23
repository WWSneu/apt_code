# Run and exactly reproduce qwen2vl results!
# mme as an example
#!/bin/bash
pip3 install qwen_vl_utils
# Optional: choose patch selection method and alpha via env vars
# PATCH_SELECTION_METHOD: 'v1' | 'v2' | 'budget' (default: 'budget')
# ALPHA: float (for 'v2' it's a threshold multiplier; for 'budget' it's fraction 0-1)
PATCH_SELECTION_METHOD=${PATCH_SELECTION_METHOD:-budget}
ALPHA=${ALPHA:-0.25}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRANSFORMERS_SRC="${PROJECT_ROOT}/Qwen2-VL/transformers/src"

export PYTHONPATH="${TRANSFORMERS_SRC}:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=4,5,6,7 \
accelerate launch --num_processes=4 --main_process_port=12345 -m lmms_eval \
    --model qwen2_vl \
    --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=2359296,patch_selection_method=${PATCH_SELECTION_METHOD},alpha=${ALPHA} \
    --tasks mmstar  \
    --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/