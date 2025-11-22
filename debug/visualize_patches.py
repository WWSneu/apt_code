#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化补丁重要性热力图和补丁边界的独立脚本
用于调试和展示APT（Adaptive Patch Tokenization）的效果
支持批量处理文件夹中的所有图片
"""

import sys
import argparse
from pathlib import Path
import math
from typing import List, Dict, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
import csv

# 添加项目路径以便导入模块
# 当前文件: apt_code/debug/visualize_patches.py
# 目标模块: apt_code/Qwen2-VL/transformers/src/apt/
# 需要指向: Qwen2-VL/transformers/src
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # apt_code根目录
apt_src_path = project_root / "Qwen2-VL" / "transformers" / "src"
sys.path.insert(0, str(apt_src_path))

from apt.patch_tokenizer import PatchTokenizer
from apt.entropy_utils import visualize_selected_patches_cv2_non_overlapping

# 支持的图像格式
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


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


def create_output_structure(output_dir: Path, image_name: str) -> Dict[str, Path]:
    """创建有组织的输出目录结构"""
    # 去掉图像扩展名作为子目录名
    image_stem = Path(image_name).stem
    
    subdirs = {
        'root': output_dir / image_stem,
        'heatmaps': output_dir / image_stem / '01_heatmaps',
        'patches': output_dir / image_stem / '02_patches_grid',
        'images': output_dir / image_stem / '03_original_resized',
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return subdirs


def visualize_patches(
    image_path: str,
    output_dir: str = None,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    thresholds: list = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patch_selection_method: str = "v1",
    alpha: float = 1.0
):
    """
    处理单个图像并生成可视化
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录，默认为当前目录下的visualizations文件夹
        patch_size: 补丁大小（默认14）
        temporal_patch_size: 时间补丁大小（默认2）
        merge_size: 合并大小（默认2）
        thresholds: 重要性阈值列表（默认[3, 2]）
        device: 计算设备
        patch_selection_method: patch选择方法（'v1'或'v2'，默认'v1'）
        alpha: v2方法的超参数，控制threshold = alpha * mean(max_patch_entropy)，默认1.0
    """
    
    if thresholds is None:
        thresholds = [3, 2]
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(__file__).parent / "visualizations"
    else:
        output_dir = Path(output_dir)
    
    # 创建有组织的输出结构
    subdirs = create_output_structure(output_dir, Path(image_path).name)
    
    print(f"\n{'='*60}")
    print(f"图像处理和可视化工具")
    print(f"{'='*60}")
    print(f"输入图像: {image_path}")
    print(f"输出目录: {subdirs['root']}")
    print(f"设备: {device}")
    print(f"Patch选择方法: {patch_selection_method}")
    
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
        patch_selection_method=patch_selection_method,
        alpha=alpha,
    )
    pt.to(device)
    
    # 计算重要性热力图
    batch_maps = pt.compute_importance_maps(img_tensor)
    print(f"  - 计算出的尺度数: {len(batch_maps)}")
    
    # 保存重要性热力图
    print(f"\n[4/5] 保存重要性热力图...")
    for scale_key, importance_map in batch_maps.items():
        output_path = subdirs['heatmaps'] / f"importance_map_scale_{scale_key}.png"
        save_importance_heatmap(importance_map, str(output_path), scale_name=str(scale_key))
    
    # 构建掩码
    masks, _, seqlens = pt.construct_masks(batch_maps)

    # 如果使用 budget 模式，尝试读取 tokenizer 中记录的 budget 信息
    budget_info = None
    try:
        if hasattr(pt, 'last_budget_info') and pt.last_budget_info is not None:
            budget_info = pt.last_budget_info
    except Exception:
        budget_info = None

    # 统计原始最小块(最小patch)数量和APT后保留的patch数量
    # resized_height/width 已是调整后的尺寸 (H, W)
    num_patches_h = math.ceil(resized_height / patch_size)
    num_patches_w = math.ceil(resized_width / patch_size)
    total_min_patches = int(num_patches_h * num_patches_w)

    # seqlens 是一个列表（每个batch项），包含 class token (1) + 选中patch数
    batch_size = 1
    if isinstance(seqlens, (list, tuple)):
        seqlens_list = seqlens
    else:
        # 如果是 torch tensor 列表或其他类型，尝试转换
        try:
            seqlens_list = list(seqlens)
        except Exception:
            seqlens_list = [int(s) for s in seqlens]

    retained_including_cls = sum(seqlens_list)
    retained_excluding_cls = retained_including_cls - len(seqlens_list)
    retained_per_image = [s - 1 for s in seqlens_list]

    # 每个尺度的选中patch数量（对batch第一项）
    per_scale_selected = {}
    for scale_key, mask in masks.items():
        try:
            sel = int(mask[0].sum().item())
        except Exception:
            sel = int(mask.sum().item())
        per_scale_selected[scale_key] = sel

    # 打印统计信息（在生成补丁边界预览前）
    print(f"\n[4.5] Patch 统计信息:")
    print(f"  - 原始图像大小: {original_size} (W,H)")
    print(f"  - 调整后大小: ({resized_width}, {resized_height}) (W,H)")
    print(f"  - 最小块 ( {patch_size}x{patch_size} ) 网格: {num_patches_h} x {num_patches_w} = {total_min_patches} patches")
    print(f"  - APT 后保留的 patch (每张图, 不含 class token): {retained_per_image}  (总计 {retained_excluding_cls})")
    print(f"  - APT 后保留比例 (总计): {retained_excluding_cls}/{total_min_patches} = {retained_excluding_cls/total_min_patches*100:.2f}%")
    print(f"  - 每个尺度选中 patch 数量: {per_scale_selected}")

    # 将统计信息写入输出目录 (JSON + TXT)
    stats = {
        "image_name": Path(image_path).name,
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "resized_size": {"width": resized_width, "height": resized_height},
        "patch_size": patch_size,
        "grid_patches": {"num_patches_h": num_patches_h, "num_patches_w": num_patches_w, "total": total_min_patches},
        "per_scale_selected": per_scale_selected,
        "retained_per_image": retained_per_image,
        "retained_total_excluding_cls": retained_excluding_cls,
        "retained_ratio_percent": round(retained_excluding_cls / total_min_patches * 100, 4)
    }

    # 如果有 budget_info，把相关字段加入 stats
    if budget_info is not None:
        stats['budget_info'] = budget_info

    # 在终端打印 budget 对比信息（如果存在）
    if budget_info is not None:
        try:
            base_small = budget_info.get('base_small_tokens', None)
            if base_small is not None:
                print(f"\n[Budget Info] base_small_tokens={base_small}, base_tokens={budget_info['base_tokens']}, budget={budget_info['budget']}, k={budget_info['k']}")
                print(f"  - alpha(target fraction) = {alpha}")
            else:
                print(f"\n[Budget Info] base_tokens={budget_info['base_tokens']}, budget={budget_info['budget']}, k={budget_info['k']}")
                print(f"  - alpha(target fraction) = {alpha}")

            # 计算实际保留 tokens（不含 class token）
            actual_retained = retained_excluding_cls
            print(f"  - 实际保留 patch (不含 class token): {actual_retained}")
            print(f"  - 与 budget(target) 对比: {actual_retained}/{budget_info['budget']} ({actual_retained/budget_info['budget']*100 if budget_info['budget']>0 else 0:.2f}%)")
        except Exception:
            pass

    try:
        stats_json_path = subdirs['root'] / "stats.json"
        with open(stats_json_path, 'w', encoding='utf-8') as jf:
            json.dump(stats, jf, indent=2, ensure_ascii=False)

        stats_txt_path = subdirs['root'] / "stats.txt"
        with open(stats_txt_path, 'w', encoding='utf-8') as tf:
            tf.write("Image Statistics\n")
            tf.write("================\n")
            tf.write(f"Image: {stats['image_name']}\n")
            tf.write(f"Original size (W,H): {stats['original_size']['width']},{stats['original_size']['height']}\n")
            tf.write(f"Resized size (W,H): {stats['resized_size']['width']},{stats['resized_size']['height']}\n")
            tf.write(f"Patch size: {stats['patch_size']}\n")
            tf.write(f"Grid patches: {stats['grid_patches']['num_patches_h']} x {stats['grid_patches']['num_patches_w']} = {stats['grid_patches']['total']}\n")
            tf.write(f"Retained patches per image (excluding class token): {stats['retained_per_image']}\n")
            tf.write(f"Retained total (excluding cls): {stats['retained_total_excluding_cls']}\n")
            tf.write(f"Retained ratio: {stats['retained_ratio_percent']}%\n")
            tf.write(f"Per-scale selected: {stats['per_scale_selected']}\n")
    except Exception as e:
        print(f"⚠ 写入统计文件失败: {e}")
    
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
        output_path = subdirs['patches'] / "patches_visualization.jpg"
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
    img.save(subdirs['images'] / "original_image.jpg")
    print(f"✓ 保存原始图像: {subdirs['images'] / 'original_image.jpg'}")
    
    # 保存调整后的图像
    img_resized.save(subdirs['images'] / "resized_image.jpg")
    print(f"✓ 保存调整后的图像: {subdirs['images'] / 'resized_image.jpg'}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"{'='*60}")
    print(f"输出位置: {subdirs['root']}")
    print(f"\n输出结构:")
    print(f"  ├── 01_heatmaps/          - 热力图文件")
    print(f"  ├── 02_patches_grid/      - 补丁网格预览")
    print(f"  └── 03_original_resized/  - 原始和调整后的图像")
    print(f"\n生成的文件:")
    for subdir_name, subdir_path in subdirs.items():
        if subdir_name != 'root':
            for file in sorted(subdir_path.glob("*")):
                print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
    print()
    
    # 返回统计信息以便批处理汇总
    return stats


def process_folder(
    folder_path: str,
    output_dir: str = None,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    thresholds: list = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patch_selection_method: str = "v1",
    alpha: float = 1.0
):
    """
    批量处理文件夹中的所有图片
    
    Args:
        folder_path: 输入文件夹路径
        output_dir: 输出目录
        patch_size: 补丁大小
        temporal_patch_size: 时间补丁大小
        merge_size: 合并大小
        thresholds: 重要性阈值列表
        device: 计算设备
        patch_selection_method: patch选择方法（'v1'或'v2'）
        alpha: v2方法的超参数，默认1.0
    """
    folder_path = Path(folder_path)
    
    if not folder_path.is_dir():
        print(f"错误: {folder_path} 不是一个有效的目录")
        sys.exit(1)
    
    # 收集所有支持的图像文件
    image_files = []
    for ext in SUPPORTED_IMAGE_FORMATS:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"⚠ 在 {folder_path} 中未找到任何图像文件")
        return
    
    # 去除重复项并排序
    image_files = sorted(set(image_files))
    
    print(f"\n{'='*60}")
    print(f"批量图像处理工具")
    print(f"{'='*60}")
    print(f"输入文件夹: {folder_path}")
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"{'='*60}\n")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(__file__).parent / "visualizations_batch"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    # 处理每个图像
    # 准备批量汇总 CSV 文件
    summary_csv_path = output_dir / "batch_stats.csv"
    csv_header = [
        "image_name",
        "original_w",
        "original_h",
        "resized_w",
        "resized_h",
        "patch_size",
        "grid_total",
        "retained_total_excl_cls",
        "retained_ratio_percent",
        "per_scale_selected_json"
    ]
    if not summary_csv_path.exists():
        try:
            with open(summary_csv_path, 'w', newline='', encoding='utf-8') as cf:
                writer = csv.writer(cf)
                writer.writerow(csv_header)
        except Exception as e:
            print(f"⚠ 无法创建批量统计CSV: {e}")

    # 处理每个图像
    successful = 0
    failed = 0
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            print(f"\n[{idx}/{len(image_files)}] 处理: {image_file.name}")
            stats = visualize_patches(
                image_path=str(image_file),
                output_dir=str(output_dir),
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
                thresholds=thresholds,
                device=device,
                patch_selection_method=patch_selection_method,
                alpha=alpha
            )
            # 将 stats 写入汇总 CSV
            try:
                with open(summary_csv_path, 'a', newline='', encoding='utf-8') as cf:
                    writer = csv.writer(cf)
                    writer.writerow([
                        stats.get('image_name', ''),
                        stats['original_size']['width'],
                        stats['original_size']['height'],
                        stats['resized_size']['width'],
                        stats['resized_size']['height'],
                        stats.get('patch_size', patch_size),
                        stats['grid_patches']['total'],
                        stats.get('retained_total_excluding_cls', ''),
                        stats.get('retained_ratio_percent', ''),
                        json.dumps(stats.get('per_scale_selected', {}), ensure_ascii=False)
                    ])
            except Exception as e:
                print(f"⚠ 无法写入汇总CSV: {e}")

            successful += 1
        except Exception as e:
            print(f"✗ 处理失败: {image_file.name}")
            print(f"  错误: {e}")
            failed += 1
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"{'='*60}")
    print(f"总数: {len(image_files)}")
    print(f"成功: {successful} ✓")
    print(f"失败: {failed} ✗")
    print(f"输出位置: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="可视化Qwen2-VL补丁重要性热力图和补丁边界（支持单个图像或批量处理文件夹）"
    )
    
    # 添加模式选择参数
    parser.add_argument(
        "input",
        type=str,
        help="输入图像文件路径或包含图像的文件夹路径"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出目录（默认: ./visualizations 或 ./visualizations_batch）"
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
        default=[2, 3],
        help="重要性阈值列表（默认: 5 5）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备（默认: cuda或cpu）"
    )
    parser.add_argument(
        "--patch-selection-method",
        type=str,
        default="v1",
        choices=["v1", "v2", "budget"],
        help="Patch选择方法（'v1'原始方法，'v2'动态阈值，'budget'按预算分裂，默认: v1）"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help=("当选择 'v2' 时: 控制 threshold = alpha * mean(max_patch_entropy)。\n"
              "当选择 'budget' 时: alpha 表示相对于全部使用最小尺寸补丁时的保留比例（0..1），\n"
              "即希望保留的小补丁数量 = round(alpha * total_small_patches)。 更多的 alpha -> 更多保留的 patch（默认: 1.0）")
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 判断是文件还是文件夹
    if input_path.is_file():
        # 单个文件模式
        if not input_path.exists():
            print(f"错误: 图像文件不存在: {args.input}")
            sys.exit(1)
        
        visualize_patches(
            image_path=args.input,
            output_dir=args.output,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            merge_size=args.merge_size,
            thresholds=args.thresholds,
            device=args.device,
            patch_selection_method=args.patch_selection_method,
            alpha=args.alpha
        )
    elif input_path.is_dir():
        # 文件夹模式
        process_folder(
            folder_path=args.input,
            output_dir=args.output,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            merge_size=args.merge_size,
            thresholds=args.thresholds,
            device=args.device,
            patch_selection_method=args.patch_selection_method,
            alpha=args.alpha
        )
    else:
        print(f"错误: {args.input} 既不是文件也不是文件夹")
        sys.exit(1)


if __name__ == "__main__":
    main()
