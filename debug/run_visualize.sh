#!/bin/bash
# 使用 apt conda 环境运行可视化脚本的便捷脚本

# 激活 apt 环境
conda activate apt

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 运行 Python 脚本
python "$SCRIPT_DIR/visualize_patches.py" "$@"
