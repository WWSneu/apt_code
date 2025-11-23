#!/usr/bin/env python3
"""
简单测试脚本：验证 Qwen2VL 模型的 padding 逻辑是否被调用
"""

import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from lmms_eval.logging_utils import eval_logger

def test_padding_in_inference():
    """
    测试推理过程中 padding 逻辑是否被调用
    """
    print("=" * 70)
    print("Qwen2VL Padding Logic Test")
    print("=" * 70)
    
    # 1. 加载模型和处理器
    print("\n[1/4] Loading model and processor...")
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 注意：这里需要使用你修改过的 transformers
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print(f"✓ Model loaded on {device}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # 2. 准备测试图像
    print("\n[2/4] Preparing test image...")
    
    # 创建一个简单的测试图像（或使用现有图像）
    test_image_path = Path(__file__).parent / "apt_code" / "debug" / "test_img"
    
    if test_image_path.exists():
        # 尝试找到一个测试图像
        image_files = list(test_image_path.glob("*.jpg")) + list(test_image_path.glob("*.png"))
        if image_files:
            image = Image.open(image_files[0]).convert('RGB')
            print(f"✓ Using existing test image: {image_files[0].name}")
        else:
            # 创建一个合成图像
            image = Image.new('RGB', (512, 512), color=(73, 109, 137))
            print("✓ Created synthetic test image (512x512)")
    else:
        # 创建一个合成图像
        image = Image.new('RGB', (512, 512), color=(73, 109, 137))
        print("✓ Created synthetic test image (512x512)")
    
    # 3. 准备输入
    print("\n[3/4] Processing input...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image briefly."}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 使用 APT 处理
    try:
        # 这里会调用 APT 的 patch tokenizer
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        print("✓ Input processed with APT")
    except Exception as e:
        print(f"✗ Failed to process input: {e}")
        return
    
    # 4. 运行推理并观察日志
    print("\n[4/4] Running inference...")
    print("-" * 70)
    print("Watch for [PADDING DEBUG] messages in the output below:")
    print("-" * 70)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        # 解码输出
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print("-" * 70)
        print("\n✓ Inference completed successfully!")
        print(f"\nGenerated text: {generated_text[:200]}...")
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)
    print("\nIf you see '[PADDING DEBUG]' messages above, the padding logic")
    print("is being called during evaluation.")
    print("=" * 70)


if __name__ == "__main__":
    test_padding_in_inference()
