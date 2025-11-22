# 可视化调试工具

这个工具用于可视化 Qwen2-VL 补丁重要性热力图和补丁边界，支持单个图像和批量处理。

## 功能特性

- ✅ 单个图像处理
- ✅ 批量处理文件夹中的所有图片
- ✅ 组织化输出目录结构
- ✅ 支持多种图像格式 (JPG, PNG, BMP, TIFF, WEBP)
- ✅ GPU/CPU 自动检测

## 使用方法

### 1. 处理单个图像

```bash
# 基本用法
python visualize_patches.py test.jpeg

# 指定输出目录
python visualize_patches.py test.jpeg -o ./my_output

# 使用启动脚本
./run_visualize.sh test.jpeg
```

### 2. 批量处理文件夹

```bash
# 处理整个文件夹中的所有图片
python visualize_patches.py /path/to/image/folder

# 指定输出目录
python visualize_patches.py /path/to/image/folder -o ./batch_output
```

### 3. 高级参数

```bash
python visualize_patches.py input \
    --patch-size 14 \
    --temporal-patch-size 2 \
    --merge-size 2 \
    --thresholds 5 5 \
    --device cuda
```

## 输出目录结构

### 单个图像输出

```
visualizations/
└── test
    ├── 01_heatmaps/           # 热力图
    │   ├── importance_map_scale_0.png
    │   ├── importance_map_scale_1.png
    │   └── importance_map_scale_2.png
    ├── 02_patches_grid/       # 补丁网格预览
    │   └── patches_visualization.jpg
    └── 03_original_resized/   # 原始和调整后的图像
        ├── original_image.jpg
        └── resized_image.jpg
```

### 批量输出

```
visualizations_batch/
├── image1
│   ├── 01_heatmaps/
│   ├── 02_patches_grid/
│   └── 03_original_resized/
├── image2
│   ├── 01_heatmaps/
│   ├── 02_patches_grid/
│   └── 03_original_resized/
└── ...
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | - | 输入图像或文件夹路径 (必需) |
| `-o, --output` | `./visualizations` | 输出目录 |
| `--patch-size` | 14 | 补丁大小 |
| `--temporal-patch-size` | 2 | 时间补丁大小 |
| `--merge-size` | 2 | 合并大小 |
| `--thresholds` | 5 5 | 重要性阈值列表 |
| `--device` | cuda/cpu | 计算设备 |

## 示例

### 快速测试

```bash
./run_visualize.sh test.jpeg
```

### 处理大量图像

```bash
# 创建一个包含多个图像的文件夹
mkdir image_batch
cp /path/to/images/*.jpg image_batch/

# 批量处理
python visualize_patches.py image_batch -o ./results
```

### 调整阈值

```bash
# 更严格的阈值（选择更少的补丁）
python visualize_patches.py test.jpeg --thresholds 10 10

# 宽松的阈值（选择更多的补丁）
python visualize_patches.py test.jpeg --thresholds 1 1
```

## 输出文件说明

### 热力图 (01_heatmaps/)
- `importance_map_scale_0.png` - 第 0 层的重要性热力图
- `importance_map_scale_1.png` - 第 1 层的重要性热力图
- `importance_map_scale_2.png` - 第 2 层的重要性热力图

### 补丁网格 (02_patches_grid/)
- `patches_visualization.jpg` - 绿色边框标记的重要补丁网格

### 原始图像 (03_original_resized/)
- `original_image.jpg` - 原始输入图像
- `resized_image.jpg` - 调整后用于处理的图像

## 注意事项

1. **环境要求**：需要 conda 环境中安装 `apt` 依赖
2. **GPU 内存**：处理大型图像时可能需要足够的 GPU 内存
3. **图像格式**：支持 JPG, PNG, BMP, TIFF, WEBP
4. **文件夹模式**：会自动递归搜索支持的图像格式

## 故障排除

### 问题：导入错误
```
ModuleNotFoundError: No module named 'apt'
```
**解决**：确保在 apt conda 环境中运行
```bash
conda activate apt
```

### 问题：GPU 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**：使用 CPU 或减小图像大小
```bash
python visualize_patches.py test.jpeg --device cpu
```

### 问题：找不到图像
```
错误: 图像文件不存在
```
**解决**：检查路径是否正确，使用绝对路径
```bash
python visualize_patches.py /absolute/path/to/image.jpg
```
