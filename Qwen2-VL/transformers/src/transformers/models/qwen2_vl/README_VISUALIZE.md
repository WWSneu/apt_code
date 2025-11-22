# 补丁重要性热力图可视化工具

## 功能介绍

这个工具用于可视化 Qwen2-VL 中的 APT（Adaptive Patch Tokenization）补丁处理过程，包括：

1. **重要性热力图** - 显示不同尺度下补丁的重要性评分
2. **补丁边界预览** - 在原始图像上绘制选中的补丁区域
3. **处理过程可视化** - 从原始图像到处理后图像的完整流程

## 文件位置

```
visualize_patches.py
```

## 环境要求

### 推荐方式：使用 apt conda 环境

项目已配置 `apt` conda 环境，包含所有必需依赖（torch、numpy、matplotlib、opencv-python 等）。

**激活环境：**
```bash
conda activate apt
```

**使用便捷脚本运行：**
```bash
./run_visualize.sh /path/to/image.jpg
```

### 手动安装依赖

如果不使用 conda 环境，需要安装以下包：
```bash
pip install torch numpy matplotlib pillow opencv-python loguru transformers
```

## 使用方法

### 基础用法

**推荐方式（使用 apt 环境）：**
```bash
conda activate apt
python visualize_patches.py /path/to/image.jpg
```

**或使用便捷脚本：**
```bash
./run_visualize.sh /path/to/image.jpg
```

### 完整参数

```bash
python visualize_patches.py /path/to/image.jpg \
    -o ./output_dir \
    --patch-size 14 \
    --temporal-patch-size 2 \
    --merge-size 2 \
    --thresholds 3 2 \
    --device cuda
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `image_path` | 必需 | 输入图像的路径 |
| `-o, --output` | `./visualizations` | 输出目录 |
| `--patch-size` | 14 | 基础补丁大小 |
| `--temporal-patch-size` | 2 | 时间维度补丁大小 |
| `--merge-size` | 2 | 补丁合并大小 |
| `--thresholds` | 3 2 | 重要性阈值（多个值） |
| `--device` | auto | 计算设备（cuda/cpu） |

## 输出文件

脚本会在输出目录生成以下文件：

1. **importance_map_scale_14.png** - 14像素尺度的重要性热力图
2. **importance_map_scale_28.png** - 28像素尺度的重要性热力图
3. **importance_map_scale_56.png** - 56像素尺度的重要性热力图
4. **patches_visualization.jpg** - 补丁边界预览（绿色边界表示选中的补丁）
5. **original_image.jpg** - 原始输入图像
6. **resized_image.jpg** - 动态调整后的图像

## 示例

### 示例 1: 使用默认参数

```bash
python visualize_patches.py /path/to/cat.jpg
```

输出: `./visualizations/` 目录下的所有可视化图片

### 示例 2: 自定义输出目录和阈值

```bash
python visualize_patches.py /path/to/cat.jpg -o ./debug_output --thresholds 2 1
```

### 示例 3: 使用 CPU（CUDA 不可用时）

```bash
python visualize_patches.py /path/to/cat.jpg --device cpu
```

## 输出解释

### 重要性热力图（importance_map_scale_*.png）

- **颜色**: 使用 `inferno` 色彩映射
  - 深紫色 = 低重要性区域
  - 黄色 = 高重要性区域
- **用途**: 显示模型认为哪些补丁最重要
- **尺度**: 不同的 PNG 文件代表不同分辨率的补丁评分

### 补丁边界预览（patches_visualization.jpg）

- **绿色边框**: 表示被选中用于处理的补丁区域
- **边界厚度**: 2 像素
- **用途**: 直观展示补丁采样的结果

## 技术细节

### 处理流程

1. **加载**: 读取 RGB 图像
2. **调整**: 使用 `smart_resize` 动态调整大小（保持宽高比）
3. **归一化**: 使用 CLIP 均值和标准差
4. **补丁化**: 创建 `PatchTokenizer` 对象
5. **重要性计算**: 计算多尺度重要性热力图
6. **掩码生成**: 根据阈值生成选择掩码
7. **可视化**: 绘制热力图和补丁边界

### 关键参数

- **patch_size (14)**: 基础补丁大小，决定最小补丁尺度
- **merge_size (2)**: 补丁合并系数，影响补丁网格的粒度
- **thresholds ([3, 2])**: 重要性阈值，用于确定哪些补丁被保留
  - 较高的阈值 → 更少的补丁被选中 → 处理速度更快
  - 较低的阈值 → 更多的补丁被选中 → 更高的精度

## 故障排除

### 错误：图像文件不存在
```
错误: 图像文件不存在: /path/to/image.jpg
```
**解决**: 检查图像路径是否正确，文件是否存在

### 错误：CUDA 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 
- 使用 `--device cpu` 改用 CPU
- 或选择更小的图像

### 错误：找不到依赖模块
```
ModuleNotFoundError: No module named 'apt'
```
**解决**: 确保项目路径正确，apt 模块已安装

## 依赖项

- torch
- numpy
- matplotlib
- PIL
- cv2 (opencv-python)
- loguru
- transformers
- apt（项目内部模块）

## 性能提示

- **大图像**: 使用 CUDA 加速，通常需要 2-5 秒
- **小图像**: CPU 处理也相对快速，不到 1 秒
- **内存**: 单张图像通常需要 < 2GB 内存

## 许可证

遵循原始 Qwen2-VL 代码的 Apache License 2.0
