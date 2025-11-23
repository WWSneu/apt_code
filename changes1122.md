# 2025-11-22 代码修改总结

以下是本次调试过程中对代码所做的所有修改总结：

### 1. `transformers/src/apt/entropy_utils.py`
*   **修改内容**：在 `select_patches_by_budget` 函数中，**移除了对 `ent_l2` 和 `ent_l3` 的裁剪 (cropping) 操作**。
*   **原因**：Budget 模式需要将图像尺寸 padding 到能被 4 整除（为了 L3 层级处理）。原代码中的裁剪会导致生成的 mask 尺寸变回未 padding 的大小，从而与后续处理中期望的 padding 后尺寸不一致，导致 `IndexError` 或形状不匹配。保留 padding 后的尺寸可以确保 mask 与 padding 后的图像特征图对齐。

### 2. `transformers/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`
*   **修改内容 1 (Padding 修复)**：在 `preprocess` 方法中增加了**自动 Padding 逻辑**。
    *   **原因**：当 APT 算法（Budget 模式）生成的 mask 尺寸（例如 16x16）大于实际图像 patch 的尺寸（例如 14x14）时，会发生 `RuntimeError`。新增的代码会检测这种情况，并用 0 填充 `patches` 张量，使其形状与 mask 匹配。
*   **修改内容 2 (日志功能)**：添加了 `eval_logger.info` 打印 **Image Stats**。
    *   **原因**：应你的要求，输出了每张图的原始尺寸 (`Original Size`)、Resize 后的尺寸 (`Resized Size`) 以及最终保留的 Patch 数量 (`Final APT Patch Count`)，便于观察模型行为。

### 3. `transformers/src/apt/patch_tokenizer.py`
*   **修改内容**：在 `construct_patch_groups` 函数中，修改了 `einops.rearrange` 的调用方式。
    *   **原代码**：使用静态的 `self.image_size` 计算 patch 数量 (例如 `(h w) = 256`).
    *   **新代码**：改为使用**动态变量** `h=num_patches_h, w=num_patches_w`。
*   **原因**：修复了 `einops.EinopsError: Shape mismatch`。原代码假设图片总是固定的正方形（如 224x224），但在处理动态分辨率或经过 padding 的图片时，实际 patch 数量（如 196）与静态计算的数量（如 256）不一致。改为动态获取高度和宽度后，问题解决。

---

**总结效果**：
这些修改共同解决了在 **Budget 模式** 下，由于图像 Padding 导致的 Mask 尺寸与 Tensor 尺寸不匹配的问题，以及动态分辨率下的维度重排错误，并增加了调试所需的日志信息。
