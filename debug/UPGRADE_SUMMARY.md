# 可视化工具升级总结

## 📋 升级内容

### 1. **批量处理功能** ✅
- 支持处理单个图像或整个文件夹
- 自动扫描文件夹中的所有支持格式的图片
- 支持的格式：JPG, PNG, BMP, TIFF, WEBP

### 2. **组织化输出结构** ✅
每个图像的输出都按类型分类组织：
```
image_name/
├── 01_heatmaps/           # 所有热力图文件
│   ├── importance_map_scale_0.png
│   ├── importance_map_scale_1.png
│   └── importance_map_scale_2.png
├── 02_patches_grid/       # 补丁边界可视化
│   └── patches_visualization.jpg
└── 03_original_resized/   # 原始和调整后的图像
    ├── original_image.jpg
    └── resized_image.jpg
```

### 3. **改进的用户界面** ✅
- 彩色的进度输出
- 详细的处理信息
- 统计汇总（成功/失败）
- 清晰的输出目录结构说明

### 4. **错误处理和日志** ✅
- 单张图片处理失败不会中断整个批处理
- 详细的错误信息和统计
- 文件大小统计

## 🎯 使用示例

### 单个图像处理
```bash
cd /data/zhujiayi-20251002/workspace/llava-apt/apt_code/debug
./run_visualize.sh test.jpeg
# 或
python visualize_patches.py test.jpeg
```

### 批量处理文件夹
```bash
# 处理整个文件夹
python visualize_patches.py /path/to/images/

# 指定输出目录
python visualize_patches.py /path/to/images/ -o ./batch_results
```

### 自定义参数
```bash
python visualize_patches.py test.jpeg \
    --patch-size 14 \
    --thresholds 5 5 \
    --device cuda
```

## 📂 文件结构

```
/data/zhujiayi-20251002/workspace/llava-apt/apt_code/debug/
├── visualize_patches.py      # 主程序（439行）
├── run_visualize.sh          # 启动脚本
├── test.jpeg                 # 测试图像
└── README.md                 # 详细文档

```

## 🔧 技术改进

| 改进项 | 原版本 | 升级后 |
|--------|--------|--------|
| 处理模式 | 仅单个文件 | 单个文件 + 批量文件夹 |
| 输出组织 | 平铺输出 | 分类输出（3个子目录） |
| 错误处理 | 简单 | 详细统计和恢复 |
| 文件格式支持 | 推断 | 显式支持6种格式 |
| 代码行数 | ~150 行 | 439 行 |
| 文档 | 无 | 详细 README |

## ✨ 关键特性

1. **智能模式检测** - 自动识别输入是文件还是文件夹
2. **多格式支持** - JPG, PNG, BMP, TIFF, WEBP
3. **灵活输出** - 支持自定义输出目录
4. **性能跟踪** - 文件大小和处理统计
5. **可靠性** - 单个失败不影响批处理进度

## 📊 处理流程

```
输入 (文件或文件夹)
  ↓
检测类型
  ├→ 文件：单个处理
  └→ 文件夹：批量处理
    ├→ 扫描图像
    └→ 逐个处理
      ├→ 图像预处理
      ├→ 计算热力图
      ├→ 生成可视化
      └→ 保存到组织化目录
  ↓
汇总统计和报告
```

## 🚀 快速开始

1. 进入 debug 目录
   ```bash
   cd /data/zhujiayi-20251002/workspace/llava-apt/apt_code/debug
   ```

2. 激活环境（如果还未激活）
   ```bash
   conda activate apt
   ```

3. 运行程序
   ```bash
   # 处理单个图像
   python visualize_patches.py test.jpeg
   
   # 处理文件夹
   python visualize_patches.py /path/to/images/
   ```

4. 查看结果
   ```bash
   ls -R visualizations/
   ```

## 📝 输出示例

批量处理文件夹的输出结果：

```
============================================================
批量图像处理工具
============================================================
输入文件夹: /path/to/images
找到 3 个图像文件
============================================================

[1/3] 处理: image1.jpg
  [1/5] 加载图像...
  [2/5] 动态调整图像大小...
  [3/5] 计算重要性热力图...
  [4/5] 保存重要性热力图...
  [5/5] 生成补丁边界预览图...
✓ 保存重要性热力图: ...
✓ 保存补丁边界预览: ...
输出结构:
  ├── 01_heatmaps/          - 热力图文件
  ├── 02_patches_grid/      - 补丁网格预览
  └── 03_original_resized/  - 原始和调整后的图像

[2/3] 处理: image2.jpg
...

[3/3] 处理: image3.jpg
...

============================================================
批量处理完成
============================================================
总数: 3
成功: 3 ✓
失败: 0 ✗
输出位置: /path/to/visualizations_batch
============================================================
```

## 🔍 路径配置

程序已更新以从 apt_code 根目录查找依赖：
- **APT 模块路径**：`apt_code/Qwen2-VL/transformers/src/apt/`
- **相对导入**：从项目根目录正确解析

无需手动修改 Python 路径！
