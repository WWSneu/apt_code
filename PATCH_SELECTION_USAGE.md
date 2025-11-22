# Patch Selection Method Usage Guide

## Overview

The Qwen2-VL image processor now supports three patch selection methods:
- **v1**: Fixed thresholds per scale (original behavior)
- **v2**: Dynamic thresholds based on alpha parameter (threshold = alpha × mean(max_entropy))
- **budget**: Budget-based selection with alpha as fraction of smallest patches to retain

## Using Patch Selection in Modeling

### 1. Direct ImageProcessor Usage

```python
from transformers import Qwen2VLImageProcessor

# Create processor with specific selection method and alpha parameter
processor = Qwen2VLImageProcessor(
    patch_selection_method='v2',  # or 'v1', 'budget'
    alpha=0.5  # for v2: threshold scaling; for budget: retention fraction
)

# Preprocess images
pixel_values = processor(images=image_list, patch_selection_method='v2', alpha=0.5)
```

### 2. Using Qwen2VLProcessor (Recommended)

The `Qwen2VLProcessor` wraps both image processor and tokenizer. You can pass patch selection parameters through `images_kwargs`:

```python
from transformers import Qwen2VLProcessor
from PIL import Image

# Initialize processor
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B")

# Load an image
image = Image.open("path/to/image.jpg")

# Process with specific patch selection method
outputs = processor(
    images=image,
    text="Describe this image",
    images_kwargs={
        'patch_selection_method': 'budget',  # Use budget mode
        'alpha': 0.3  # Retain 30% of smallest patches
    }
)

# Now outputs contains:
# - input_ids: tokenized text
# - pixel_values: processed image patches
# - image_grid_thw: image grid dimensions
```

### 3. Different Patch Selection Modes

#### Mode v1: Fixed Thresholds (Default)
```python
outputs = processor(
    images=image,
    text="Describe this image",
    images_kwargs={'patch_selection_method': 'v1'}
)
```
Uses fixed thresholds [3, 2] for entropy-based pruning at different scales.

#### Mode v2: Dynamic Thresholds
```python
outputs = processor(
    images=image,
    text="Describe this image",
    images_kwargs={
        'patch_selection_method': 'v2',
        'alpha': 0.5  # threshold = 0.5 × mean(max_entropy)
    }
)
```
Scales the threshold dynamically. Alpha values:
- Lower alpha (0.1-0.3): More aggressive pruning, fewer patches
- Higher alpha (1.0-2.0): Less aggressive pruning, more patches

#### Mode budget: Adaptive Budget-Based
```python
outputs = processor(
    images=image,
    text="Describe this image",
    images_kwargs={
        'patch_selection_method': 'budget',
        'alpha': 0.3  # Keep 30% of smallest patches
    }
)
```
Allocates a budget (number of patches to keep) based on alpha:
- `budget = round(alpha × H1 × W1)` where H1×W1 is smallest patch grid
- This ensures consistent patch reduction across different image sizes

### 4. Complete Example with Model Inference

```python
import torch
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from PIL import Image

# Load processor and model
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B")

# Load image
image = Image.open("frieren.jpg").convert("RGB")

# Process with budget mode (30% retention)
inputs = processor(
    images=image,
    text="<|im_start|>user\n<image>\nDescribe this image<|im_end|>\n<|im_start|>assistant\n",
    images_kwargs={
        'patch_selection_method': 'budget',
        'alpha': 0.3
    },
    return_tensors='pt'
)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate response
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128)

# Decode output
response = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
```

### 5. Configuring Default Values

You can set default patch selection parameters when creating the processor:

```python
from transformers import Qwen2VLImageProcessor

# Create processor with budget mode as default
image_processor = Qwen2VLImageProcessor(
    patch_selection_method='budget',
    alpha=0.3
)

# This processor will use budget mode by default
pixel_values = image_processor(images=image_list)

# But you can still override at call time
pixel_values = image_processor(
    images=image_list,
    patch_selection_method='v2',
    alpha=0.5
)
```

## Performance Implications

### v1 (Fixed Thresholds)
- **Pros**: Consistent, simple, predictable
- **Cons**: May not adapt well to different image characteristics
- **Use case**: Baseline, reproducibility

### v2 (Dynamic Thresholds)
- **Pros**: Adapts to image entropy
- **Cons**: Requires tuning alpha parameter
- **Use case**: Balancing compression vs quality
- **Typical alpha values**: 0.5 - 1.5

### budget (Adaptive Budget)
- **Pros**: Controls exact reduction ratio, adapts to image size
- **Cons**: More complex computation
- **Use case**: Fixed computational budget
- **Typical alpha values**: 0.2 - 0.5 (20%-50% retention)

## Debugging and Inspection

The image processor records statistics in budget mode:

```python
from apt_code.patch_tokenizer import PatchTokenizer

# These stats are available when using budget mode
# You can check them in the image processor's internal state
# base_small_tokens: Total smallest patches (14×14 grid)
# budget: Allocated patches (round(alpha × base_small_tokens))
# k: Additional patches from larger scales
# actual: Actually selected patches
```

## Notes

1. **Alpha Semantics**:
   - **v2**: `alpha` is a multiplier for threshold (scaling factor)
   - **budget**: `alpha` is a fraction of smallest patches (0-1)

2. **Default Behavior**:
   - Default `patch_selection_method='v1'` maintains backward compatibility
   - Default `alpha=1.0` preserves original v2 behavior

3. **Batch Processing**:
   - All three methods work with batched inputs
   - Parameters apply to entire batch consistently
