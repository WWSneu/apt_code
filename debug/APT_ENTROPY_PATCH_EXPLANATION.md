# APT（Adaptive Patch Tokenization）根据熵热力图进行分块的详细讲解

## 概述

APT 根据图像的信息熵热力图（entropy heatmap）自适应地决定：
1. **哪些区域保留大patch**（低熵区域）
2. **哪些区域被切分成小patch**（高熵区域）

这使得纹理丰富（高熵）的区域得到细粒度表示，而平坦（低熵）的区域用粗粒度表示，从而提高效率。

---

## 整体流程

```
输入图像 (B, C, H, W)
    ↓
[第1步] PatchTokenizer.compute_importance_maps()
    ↓
计算多尺度熵热力图（Scale 0: 14×14, Scale 1: 28×28, Scale 2: 56×56）
    ↓ 返回 importance_maps = {14: shape(B,H_p,W_p), 28: ..., 56: ...}
    ↓
[第2步] PatchTokenizer.construct_masks()
    ↓
调用 select_patches_by_threshold(importance_maps, thresholds)
    ↓ 返回 masks = {14: 0/1 mask, 28: 0/1 mask, 56: 0/1 mask}
    ↓
[第3步] 使用masks选出对应patch并打包
    ↓
返回混合尺度的patch序列和对应的mask
```

---

## 详细实现步骤

### 第1步：计算熵热力图

**位置：** `apt/entropy_utils.py` 中的 `compute_patch_entropy_batched()`

**函数签名：**
```python
def compute_patch_entropy_batched(
    images: torch.Tensor,  # shape (B, C, H, W)，值域 [0, 255]
    patch_size: int = 16,  # 最小patch尺寸（通常14）
    num_scales: int = 3,   # 尺度数量（通常3）
    bins: int = 512,       # 直方图bin数
    pad_value: float = 1e6 # 填充值（用于边界）
) -> Dict[int, torch.Tensor]
```

**实现逻辑：**

```python
# 步骤 1.1：转换为灰度图
grayscale_images = (
    images[:, 0] * 0.2989 +
    images[:, 1] * 0.5870 +
    images[:, 2] * 0.1140
)  # shape: (B, H, W)

# 步骤 1.2：对每个patch尺寸计算熵
patch_sizes = [14, 28, 56]  # patch_size * (2**i)

for ps in patch_sizes:
    # 计算需要的patch网格大小
    num_patches_h = (H + ps - 1) // ps  # 向上取整
    num_patches_w = (W + ps - 1) // ps
    
    # 填充图像使其能整除ps
    pad_h = num_patches_h * ps - H
    pad_w = num_patches_w * ps - W
    padded_images = F.pad(grayscale_images, (0, pad_w, 0, pad_h), ...)
    
    # 步骤 1.3：将图像分割成patches
    # unfold() 相当于在灰度图上做sliding window
    patches = padded_images.unfold(1, ps, ps).unfold(2, ps, ps)
    # shape: (B, num_patches_h, num_patches_w, ps, ps)
    
    # 步骤 1.4：计算每个patch内的直方图
    # 将 ps*ps 个像素值放入 bins 个bin中
    histograms = torch.histc(patch, bins=512, min=0, max=255)
    # 对所有patches并行计算
    
    # 步骤 1.5：归一化直方图为概率分布
    probabilities = histograms / (ps * ps)
    # 每个概率 p_i = (该patch内值为i的像素数) / (ps*ps)
    
    # 步骤 1.6：计算Shannon熵
    # H = -sum(p_i * log2(p_i)) for i in bins
    entropy_map = -torch.sum(
        probabilities * torch.log2(probabilities + 1e-10),
        dim=last_dim
    )
    # shape: (B, num_patches_h, num_patches_w)
    
    batch_entropy_maps[ps] = entropy_map
```

**关键点：**
- ✅ 熵是基于**归一化的像素分布**，与patch面积无关
- ✅ 熵值范围：0 到 log2(bins) ≈ 9（相对独立于patch大小）
- ✅ 高熵 = 像素值分散（纹理丰富）
- ✅ 低熵 = 像素值集中（色彩单一）

**示例数值：**
- 完全均匀的patch（如灰色）：H ≈ 0
- 完全随机的patch：H ≈ 9（如果用512个bin）
- 边界或细节丰富的patch：H ≈ 5～8

---

### 第2步：根据熵阈值选择patch

**位置：** `apt/entropy_utils.py` 中的 `select_patches_by_threshold()`

**函数签名：**
```python
def select_patches_by_threshold(
    entropy_maps: Dict[int, torch.Tensor],  # {14: (B,H_p,W_p), 28: ..., 56: ...}
    thresholds: List[float]  # [threshold_for_scale1, threshold_for_scale2, ...]
) -> Dict[int, torch.Tensor]
```

**选择逻辑（分层递推）：**

```
设有3个尺度：patch_sizes = [14, 28, 56]（从小到大）

第1轮：处理56×56 patch（最大尺度）
  对每个56×56 patch，检查 entropy < threshold[0]？
    ✓ YES: 保留该大patch（mask=1）
    ✗ NO:  丢弃大patch，让其被细分（mask=0）
  
  输出: masks[56] = [0/1, 0/1, ...]

第2轮：处理28×28 patch（中等尺度）
  前提：只有在第1轮被丢弃的56×56区域内，才需要细分
  
  对于每个28×28 patch:
    1. 检查 entropy < threshold[1]？
    2. 检查它是否在某个被保留的56×56 patch内？
       (通过上采样56×56的mask到28×28 grid大小)
    
    保留条件: entropy < threshold[1] AND 不在保留的56×56内
  
  输出: masks[28] = [0/1, 0/1, ...]

第3轮：处理14×14 patch（最小尺度）
  前提：只有在第1、2轮都被丢弃的区域，才需要细分到最小
  
  对于每个14×14 patch:
    保留条件: 不在保留的56×56或28×28内
    （实际上这一层全部保留，作为最后的fallback）
  
  输出: masks[14] = [0/1, 0/1, ...]
```

**代码实现关键部分：**

```python
def select_patches_by_threshold(entropy_maps, thresholds):
    patch_sizes = sorted(list(entropy_maps.keys()))  # [14, 28, 56]
    masks = {}
    
    # 第1轮：最大尺度（56×56）的mask初始化为全1
    masks[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])
    
    # 第2、3轮：从大到小处理每个尺度
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]          # 当前处理的patch尺寸
        threshold = thresholds[i-1]             # 对应的阈值
        
        # ===== 关键步骤1：根据熵阈值创建初始mask =====
        masks[current_size] = (entropy_maps[current_size] < threshold).float()
        # 如果 entropy < threshold，保留该大patch（mask=1）
        # 否则丢弃该大patch（mask=0），后续会用小patch细分
        
        print(f"Patch {current_size}: {masks[current_size].sum()} patches retained")
    
    # ===== 关键步骤2：计算层级关系（确保没有重叠） =====
    # 将大尺度的mask上采样到小尺度，标记出被大patch覆盖的区域
    
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        
        for j in range(i):
            smaller_size = patch_sizes[j]      # 更小的patch尺寸
            scale_factor = current_size // smaller_size  # 尺度系数（如56/28=2）
            
            # 将当前尺度的mask上采样到小尺度
            # 相当于：如果大patch被保留，则其覆盖的所有小patch都被标记为"已覆盖"
            mask_upscaled = masks[current_size].repeat_interleave(
                scale_factor, dim=1
            ).repeat_interleave(
                scale_factor, dim=2
            )
            
            # 对齐大小（可能有1-2像素的差异）
            H_small, W_small = entropy_maps[smaller_size].shape[1:]
            mask_upscaled = mask_upscaled[:, :H_small, :W_small]
            
            # ===== 关键步骤3：更新小尺度mask =====
            # 小patch保留条件：
            # 1. 其自身熵 < 阈值（想保留）
            # 2. 不在任何被保留的大patch内（即upscaled_mask=0）
            masks[smaller_size] = masks[smaller_size] * (1 - mask_upscaled)
            #                     ^原mask                  ^删除被大patch覆盖的区域
    
    return masks
```

**可视化示例：**

假设图像分成4×4的14×14 patch网格（共16个patch）：

```
原始14×14 grid (所有patch都初始为1)：
┌─┬─┬─┬─┐
│1│1│1│1│
├─┼─┼─┼─┤
│1│1│1│1│ = 1 (完全被保留)
├─┼─┼─┼─┤
│1│1│1│1│
├─┼─┼─┼─┤
│1│1│1│1│
└─┴─┴─┴─┘

→ 28×28 mask来自对56×56 mask的上采样（假设右下角的56×56被保留）:
┌─┬─┐
│0│1│
├─┼─┤
│0│1│  (上采样后)
└─┴─┘

→ 右下角的4个14×14 patch被标记为"已被28×28覆盖"
  在14×14 mask中对应位置变为0:
┌─┬─┬─┬─┐
│1│1│1│1│
├─┼─┼─┼─┤
│1│1│1│1│ → 修改为 [1,1,1,1]
├─┼─┼─┼─┤
│1│1│0│0│       [1,1,0,0]
├─┼─┼─┼─┤
│1│1│0│0│       [1,1,0,0]
└─┴─┴─┴─┘

最终输出：
- 14×14 patches: 保留12个（其中被28覆盖的4个为0）
- 28×28 patches: 保留4个（右下角的2×2区域）
- 56×56 patches: 保留1个（或0个，取决于阈值）
```

---

### 第3步：构建混合尺度补丁序列

**位置：** `apt/patch_tokenizer.py` 中的 `construct_patch_groups()`

**目的：** 根据mask，从原始图像中提取对应大小的patch

```python
def construct_patch_groups(images, masks):
    """
    输入:
      - images: (B, C, H, W)
      - masks: {14: (B, H_p, W_p), 28: ..., 56: ...}
    
    输出:
      - resized_patches_14: 所有被mask标记为1的14×14 patch
      - resized_patches_28: 所有被mask标记为1的28×28 patch
      - resized_patches_56: 所有被mask标记为1的56×56 patch
    """
    
    for cur_patch_size in [14, 28, 56]:
        cur_mask = masks[cur_patch_size].bool()  # 取出该尺度的mask
        
        # 步骤1：图像填充（确保能整除patch_size）
        pad_h = num_patches_h * cur_patch_size - H
        pad_w = num_patches_w * cur_patch_size - W
        padded_image = F.pad(images, (0, pad_w, 0, pad_h))
        
        # 步骤2：对于较大的patch（28, 56），也可以提取原始分辨率的patch
        if cur_patch_size > 14:
            constituent_patches = rearrange(
                padded_image,
                "b c (h n1 p1) (w n2 p2) -> b h w (n1 n2) c p1 p2",
                h=num_patches_h,
                w=num_patches_w,
                n1=cur_patch_size//14,  # 56×56 = 4个14×14
                n2=cur_patch_size//14,
                p1=14,
                p2=14
            )
            selected_patches = constituent_patches[cur_mask]
            # 形状: (num_selected, num_14_in_patch, C, 14, 14)
        
        # 步骤3：对图像进行尺度变换（对于大patch）
        scale_img = F.interpolate(
            padded_image,
            scale_factor=0.5 ** scale_idx,  # 缩小到0.5倍或0.25倍
            mode="bilinear"
        )
        
        # 步骤4：提取该尺度的所有patch（缩小后的分辨率）
        scaled_patches = rearrange(
            scale_img,
            "b c (h p1) (w p2) -> b h w c p1 p2",
            p1=14,  # 缩小后也是14×14（在缩小空间中）
            p2=14
        )
        
        # 步骤5：使用mask选出该尺度的有效patch
        selected_patches = scaled_patches[cur_mask.bool()]
        # 形状: (num_selected, C, 14, 14)
```

---

## 阈值的作用

**关键参数：** `thresholds = [threshold_1, threshold_2, ...]`

- `threshold[0]`：28×28 patch保留条件 (entropy < threshold[0])
- `threshold[1]`：56×56 patch保留条件 (entropy < threshold[1])

**阈值设定的直观理解：**

```
阈值太低 (如 threshold=1):
  → 很少有patch被保留为大尺寸
  → 大部分区域被切分成小patch
  → 计算量少，但可能丢失全局信息

阈值太高 (如 threshold=8):
  → 大部分patch都被保留为大尺寸
  → 细节可能被忽视
  → 计算量多，但表示能力强

合理的阈值 (如 threshold=3~5):
  → 平衡低熵区域（保留大patch）和高熵区域（细分成小patch）
```

---

## 完整代码流程示例

```python
# ===== 用户代码 =====
pt = PatchTokenizer(
    num_scales=3,
    base_patch_size=14,
    image_size=(H, W),
    thresholds=[3, 2],  # 28×28的阈值=3，56×56的阈值=2
    mean=[...],
    std=[...]
)

img_tensor = torch.randn(1, 3, H, W)  # 输入图像

# ===== 第1步：计算熵热力图 =====
batch_maps = pt.compute_importance_maps(img_tensor)
# 返回：{14: (B,60,70), 28: (B,30,35), 56: (B,15,18)}

# ===== 第2步：根据阈值选择patch =====
masks, output_mask, seqlens = pt.construct_masks(batch_maps)
# masks = {14: (B,60,70), 28: (B,30,35), 56: (B,15,18)}
# 其中值为0或1，表示该patch是否被保留

# ===== 第3步：提取混合尺度patch =====
patch_groups = pt.construct_patch_groups(img_tensor, masks)
# 返回：{
#   "resized_patches_14": (N_14, C, 14, 14),  # 3500个patch
#   "resized_patches_28": (N_28, C, 14, 14),  # 200个patch（缩小后）
#   "resized_patches_56": (N_56, C, 14, 14),  # 50个patch（缩放到0.25倍）
# }

# ===== 第4步：打包成序列 =====
final_tokens = [class_token] + [
    token for patch_size in [56, 28, 14]
    for token in patch_groups[f"resized_patches_{patch_size}"]
]
# 总长度：1 + N_56 + N_28 + N_14 = 1 + 50 + 200 + 3500 = 3751
# （相比原始 4200 patch 节省了约10%的计算量）
```

---

## 核心特点总结

| 特性 | 说明 |
|------|------|
| **适应性** | 不同区域使用不同尺寸的patch |
| **层级性** | 大patch被保留 → 小patch不会重复 |
| **独立性** | 熵值不依赖patch面积（归一化）|
| **灵活性** | 可通过阈值调整保留比例 |
| **效率** | 大幅减少token数量而保留关键细节 |

---

## 可视化查看

你可以运行可视化工具查看实际的分块效果：

```bash
cd /data/zhujiayi-20251002/workspace/llava-apt/apt_code/debug
python visualize_patches.py test_img/test.jpeg
```

输出目录结构：
```
visualizations/test/
├── 01_heatmaps/           # 3个尺度的熵热力图
│   ├── importance_map_scale_14.png
│   ├── importance_map_scale_28.png
│   └── importance_map_scale_56.png
├── 02_patches_grid/       # 混合尺寸patch的边界可视化
│   └── patches_visualization.jpg
└── 03_original_resized/
    ├── original_image.jpg
    └── resized_image.jpg
```

热力图中颜色越深（红色）表示熵值越高 → 该区域会被切分成小patch。
