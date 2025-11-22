#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化补丁重要性热力图和补丁边界的独立脚本
用于调试和展示APT（Adaptive Patch Tokenization）的效果
"""

import sys
import argparse
from pathlib import Path
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 添加项目路径以便导入模块
# 当前文件: transformers/models/qwen2_vl/visualize_patches.py
# 目标模块: apt/patch_tokenizer.py (在同级 src 目录下)
# 需要回溯: qwen2_vl -> models -> transformers -> src(当前)
current_file = Path(__file__).resolve()
src_path = current_file.parent.parent.parent.parent  # 回到 src 目录
sys.path.insert(0, str(src_path))

from apt.patch_tokenizer import PatchTokenizer
from apt.entropy_utils import visualize_selected_patches_cv2_non_overlapping


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """动态调整图像大小，保持宽高比"""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def save_importance_heatmap(importance_map, output_path, scale_name=""):
    """保存重要性热力图为matplotlib热力图"""
    # 转换为numpy数组
    if isinstance(importance_map, torch.Tensor):
        entropy_np = importance_map.squeeze(0).cpu().numpy()
    else:
        entropy_np = importance_map.squeeze(0)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(entropy_np, cmap='inferno')
    ax.set_title(f'Importance Map - Scale {scale_name}', fontsize=14)
    ax.axis('off')
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Importance Score', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ 保存重要性热力图: {output_path}")


def visualize_patches(
    image_path: str,
    output_dir: str = None,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    thresholds: list = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    处理图像并生成可视化
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录，默认为当前目录下的visualizations文件夹
        patch_size: 补丁大小（默认14）
        temporal_patch_size: 时间补丁大小（默认2）
        merge_size: 合并大小（默认2）
        thresholds: 重要性阈值列表（默认[3, 2]）
        device: 计算设备
    """
    
    if thresholds is None:
        thresholds = [3, 2]
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(__file__).parent / "visualizations"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"图像处理和可视化工具")
    print(f"{'='*60}")
    print(f"输入图像: {image_path}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    
    # 加载和预处理图像
    print(f"\n[1/5] 加载图像...")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)
    print(f"  - 原始大小: {original_size}")
    
    # 转换为numpy数组并归一化
    img_array = np.array(img, dtype=np.float32) / 255.0
    print(f"  - 数组形状: {img_array.shape}")
    
    # 调整大小
    print(f"\n[2/5] 动态调整图像大小...")
    height, width = img_array.shape[:2]
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=56 * 56,
        max_pixels=14 * 14 * 4 * 1280,
    )
    print(f"  - 调整后大小: ({resized_height}, {resized_width})")
    
    # 使用PIL调整大小
    img_resized = img.resize((resized_width, resized_height), Image.BICUBIC)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # 归一化
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img_array = (img_array - mean) / std
    
    # 转换为(C, H, W)格式
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    print(f"  - 张量形状: {img_tensor.shape}")
    
    # 创建补丁令牌化器
    print(f"\n[3/5] 计算重要性热力图...")
    pt = PatchTokenizer(
        num_scales=3,
        base_patch_size=patch_size,
        image_size=(resized_height, resized_width),
        thresholds=thresholds,
        mean=mean,
        std=std,
    )
    pt.to(device)
    
    # 计算重要性热力图
    batch_maps = pt.compute_importance_maps(img_tensor)
    print(f"  - 计算出的尺度数: {len(batch_maps)}")
    
    # 保存重要性热力图
    print(f"\n[4/5] 保存重要性热力图...")
    for scale_key, importance_map in batch_maps.items():
        output_path = output_dir / f"importance_map_scale_{scale_key}.png"
        save_importance_heatmap(importance_map, str(output_path), scale_name=str(scale_key))
    
    # 构建掩码
    masks, _, seqlens = pt.construct_masks(batch_maps)
    
    # 生成补丁边界可视化
    print(f"\n[5/5] 生成补丁边界预览图...")
    
    # 准备可视化掩码
    visualization_masks = {}
    for scale, mask in masks.items():
        visualization_masks[scale] = mask.squeeze(0)
    
    # 准备原始图像用于绘制
    img_for_vis = img_tensor[0].permute(1, 2, 0)  # (H, W, C)
    # 反归一化
    img_for_vis = img_for_vis * torch.tensor(std, device=device).view(1, 1, 3)
    img_for_vis = img_for_vis + torch.tensor(mean, device=device).view(1, 1, 3)
    img_for_vis = (img_for_vis * 255).clamp(0, 255).byte()  # 转换为uint8
    
    # 生成补丁大小列表
    patch_sizes = [patch_size * (2**i) for i in range(3)]
    
    # 绘制补丁
    try:
        vis_img = visualize_selected_patches_cv2_non_overlapping(
            image_tensor=img_for_vis,
            masks=visualization_masks,
            patch_sizes=patch_sizes,
            color=(0, 255, 0),  # 绿色
            thickness=2
        )
        output_path = output_dir / "patches_visualization.jpg"
        if isinstance(vis_img, torch.Tensor):
            vis_img = vis_img.cpu().numpy()
        if hasattr(vis_img, 'save'):
            vis_img.save(str(output_path))
        else:
            # 如果是numpy数组或cv2图像
            cv2.imwrite(str(output_path), vis_img)
        print(f"✓ 保存补丁边界预览: {output_path}")
    except Exception as e:
        print(f"⚠ 补丁边界可视化失败: {e}")
    
    # 保存原始图像
    img.save(output_dir / "original_image.jpg")
    print(f"✓ 保存原始图像: {output_dir / 'original_image.jpg'}")
    
    # 保存调整后的图像
    img_resized.save(output_dir / "resized_image.jpg")
    print(f"✓ 保存调整后的图像: {output_dir / 'resized_image.jpg'}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"{'='*60}")
    print(f"输出位置: {output_dir}")
    print(f"\n生成的文件:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="可视化Qwen2-VL补丁重要性热力图和补丁边界"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="输入图像路径"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出目录（默认: ./visualizations）"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="补丁大小（默认: 14）"
    )
    parser.add_argument(
        "--temporal-patch-size",
        type=int,
        default=2,
        help="时间补丁大小（默认: 2）"
    )
    parser.add_argument(
        "--merge-size",
        type=int,
        default=2,
        help="合并大小（默认: 2）"
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[5, 5],
        help="重要性阈值列表（默认: 3 2）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备（默认: cuda或cpu）"
    )
    
    args = parser.parse_args()
    
    # 检查图像文件是否存在
    if not Path(args.image_path).exists():
        print(f"错误: 图像文件不存在: {args.image_path}")
        sys.exit(1)
    
    visualize_patches(
        image_path=args.image_path,
        output_dir=args.output,
        patch_size=args.patch_size,
        temporal_patch_size=args.temporal_patch_size,
        merge_size=args.merge_size,
        thresholds=args.thresholds,
        device=args.device
    )


if __name__ == "__main__":
    main()
