#!/bin/bash
# 简单测试脚本：验证 padding 逻辑是否被调用
# 只运行一个样本来快速检查日志输出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRANSFORMERS_SRC="${PROJECT_ROOT}/APT_eval/Qwen2-VL/transformers/src"

export PYTHONPATH="${TRANSFORMERS_SRC}:${PYTHONPATH}"

# 设置 APT 参数
PATCH_SELECTION_METHOD=${PATCH_SELECTION_METHOD:-budget}
ALPHA=${ALPHA:-0.25}

echo "=========================================="
echo "Testing Padding Logic in Qwen2VL"
echo "=========================================="
echo "Method: ${PATCH_SELECTION_METHOD}"
echo "Alpha: ${ALPHA}"
echo "=========================================="
echo ""
echo "Looking for [PADDING DEBUG] messages..."
echo ""

# 只运行1个样本来快速测试
CUDA_VISIBLE_DEVICES=0 \
python -m lmms_eval \
    --model qwen2_vl \
    --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=2359296,patch_selection_method=${PATCH_SELECTION_METHOD},alpha=${ALPHA} \
    --tasks mmstar \
    --batch_size 1 \
    --limit 1 \
    --output_path ./logs/ \
    2>&1 | tee padding_test.log

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo ""
echo "Checking for padding debug messages..."
grep -n "PADDING DEBUG" padding_test.log | head -20

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Padding logic IS being called!"
else
    echo ""
    echo "✗ No padding debug messages found"
    echo "  The padding logic might not be called, or logging is disabled"
fi

echo ""
echo "Full log saved to: padding_test.log"
echo "=========================================="
