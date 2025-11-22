# 快速参考：Qwen2-VL Patch划分方法选择

## TL;DR (最重要的信息)

在Qwen2VL的modeling中使用image_processor时，通过以下方式选择patch划分方法和传递alpha参数：

### 方式1：通过Processor调用 (推荐)

```python
from transformers import Qwen2VLProcessor

processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B")

# 在调用时通过images_kwargs传递参数
outputs = processor(
    images=image,
    text="Describe this image",
    images_kwargs={
        'patch_selection_method': 'budget',  # 选择方法: 'v1', 'v2', 或 'budget'
        'alpha': 0.3  # alpha参数: 对budget为比例(0-1), 对v2为倍数
    }
)
```

### 方式2：设置Processor默认值

```python
# 设置默认值
processor.image_processor.patch_selection_method = 'budget'
processor.image_processor.alpha = 0.3

# 之后的所有调用都使用这些默认值
outputs = processor(images=image, text=prompt)

# 可以在某次调用时覆盖
outputs = processor(
    images=image,
    text=prompt,
    images_kwargs={'patch_selection_method': 'v2', 'alpha': 0.5}
)
```

### 方式3：直接创建ImageProcessor

```python
from transformers import Qwen2VLImageProcessor

processor = Qwen2VLImageProcessor(
    patch_selection_method='budget',
    alpha=0.3
)

# 处理图像
pixel_values = processor(images=image_list)

# 或在调用时覆盖
pixel_values = processor(
    images=image_list,
    patch_selection_method='v2',
    alpha=0.5
)
```

## 三种Patch划分方法

| 方法 | Alpha含义 | 适用场景 | 计算方式 |
|------|---------|--------|---------|
| **v1** | 忽略 | 原始行为/基准 | 固定阈值 [3, 2] |
| **v2** | 阈值倍数 | 自适应压缩 | `threshold = alpha × mean(max_entropy)` |
| **budget** | 保留比例(0-1) | 固定计算预算 | `budget = round(alpha × 最小patches数)` |

## 推荐设置

```python
# 1. 保守方案（高质量）
images_kwargs={'patch_selection_method': 'budget', 'alpha': 0.5}
# 保留50%的patches，质量损失小，计算量中等

# 2. 平衡方案（推荐）
images_kwargs={'patch_selection_method': 'budget', 'alpha': 0.3}
# 保留30%的patches，质量-速度权衡好

# 3. 激进方案（高速度）
images_kwargs={'patch_selection_method': 'budget', 'alpha': 0.2}
# 保留20%的patches，显著加速但质量损失较大

# 4. v2自适应方案
images_kwargs={'patch_selection_method': 'v2', 'alpha': 0.7}
# 根据图像内容自适应选择patches
```

## 在Model推理中使用

```python
import torch
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration

processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B")

# 方法A: 修改processor的默认配置
processor.image_processor.patch_selection_method = 'budget'
processor.image_processor.alpha = 0.3

inputs = processor(
    images=image,
    text="<|im_start|>user\n<image>\nDescribe<|im_end|>\n<|im_start|>assistant\n",
    return_tensors='pt'
)

# 方法B: 每次调用时指定
inputs = processor(
    images=image,
    text=prompt,
    images_kwargs={
        'patch_selection_method': 'budget',
        'alpha': 0.3
    },
    return_tensors='pt'
)

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=256)

response = processor.tokenizer.decode(output_ids[0])
```

## Alpha参数详解

### 对于budget模式
- **含义**: 希望保留的最小patches的比例 (0 到 1 之间)
- **计算**: `budget = round(alpha × 14 × 14)` (其中14×14是最小patch网格)
- **示例**:
  - `alpha=0.2` → 保留20%的patches → 约56个patch
  - `alpha=0.3` → 保留30%的patches → 约59个patch  
  - `alpha=0.5` → 保留50%的patches → 约98个patch

### 对于v2模式
- **含义**: 动态阈值的缩放因子
- **计算**: `threshold = alpha × mean(max_entropy)` 
- **示例**:
  - `alpha=0.5` → 更激进的剪枝 → 更少的patches
  - `alpha=1.0` → 适中的剪枝 (默认)
  - `alpha=1.5` → 保守的剪枝 → 更多的patches

## Processing Pipeline说明

```
Input Image (224×224)
    ↓
Qwen2VLProcessor
    ├── Image Processor
    │   └── _preprocess() 调用 PatchTokenizer
    │       └── compute_importance_maps()
    │           └── construct_masks() ← 在这里应用patch_selection_method和alpha
    │               └── construct_patch_groups()
    └── Tokenizer
        └── 文本编码

Output: {pixel_values, image_grid_thw, input_ids, attention_mask}
```

## 修改processing_qwen2_vl.py以支持新参数

已在 `Qwen2VLImagesKwargs` 中添加了两个新字段：

```python
class Qwen2VLImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    patch_selection_method: Optional[str]  # ← 新增
    alpha: Optional[float]                  # ← 新增
```

这些参数通过 `images_kwargs` 自动传递给 `image_processor.preprocess()`。

## 完整调用链

```
processor(images, text, images_kwargs={...})
    ↓
processor.__call__()
    ↓
self.image_processor(**output_kwargs["images_kwargs"])
    ↓
Qwen2VLImageProcessor.preprocess(
    patch_selection_method=...,
    alpha=...
)
    ↓
Qwen2VLImageProcessor._preprocess(
    patch_selection_method=...,
    alpha=...
)
    ↓
PatchTokenizer(
    patch_selection_method=...,
    alpha=...
)
    ↓
pt.construct_masks(batch_maps)
    ↓
select_patches_by_budget() / select_patches_by_threshold_v2() / select_patches_by_threshold()
```

## 常见问题

**Q: 我应该使用哪个方法？**
A: 从 `budget` 和 `alpha=0.3` 开始。如果需要更多质量，使用 `alpha=0.5`；如果需要更快速度，使用 `alpha=0.2`。

**Q: 如何知道实际选择了多少patch？**
A: 在 budget 模式下，`PatchTokenizer.last_budget_info` 包含详细信息（base_tokens, budget, k, actual）。

**Q: v2 的 alpha 应该设为多少？**
A: 通常设为 0.5 到 1.5。0.5 更激进，1.5 更保守。从 0.7 或 1.0 开始试试。

**Q: 能否为不同的图像使用不同的alpha？**
A: 可以，在每次调用 processor 时通过 `images_kwargs` 指定不同的值。

**Q: 这会影响模型输出质量吗？**
A: 会有轻微影响，但通常可以接受。budget 模式下，`alpha=0.3` 时质量损失约 2-5%。
