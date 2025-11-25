"""Utilities for computing and visualizing patch-wise entropy in images.

This module provides functions for:
1. Computing entropy maps for different patch sizes in images using vectorized operations
2. Selecting patches based on entropy thresholds for mixed-resolution processing
3. Visualizing the selected patches with different sizes

The main components are:
- compute_patch_entropy_vectorized: Efficiently computes entropy for multiple patch sizes
- select_patches_by_threshold: Selects patches of different sizes based on entropy threshold
- visualize_selected_patches_cv2: Visualizes the selected patches using OpenCV

These utilities are particularly useful for mixed-resolution image processing where
different regions of an image can be processed at different scales based on their
information content (entropy).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from torchvision.transforms import functional as TF
import math
import cv2
from PIL import Image
import ipdb
from loguru import logger as eval_logger
def compute_patch_entropy_vectorized(image, patch_size=16, num_scales=2, bins=512, pad_value=1e6):
    """
    Compute entropy maps for multiple patch sizes in the input image using vectorized operations.
    
    Args:
        image: torch.Tensor of shape (C, H, W) or (H, W) with values in range [0, 255]
        patch_sizes: list of patch sizes (default: [16, 32])
        bins: number of bins for histogram (default: 512)
        pad_value: high entropy value to pad incomplete patches with (default: 1e6)
    
    Returns:
        entropy_maps: dict containing torch.Tensor entropy maps for each patch size
    """
    if len(image.shape) == 3:
        # Convert to grayscale if image is RGB
        if image.shape[0] == 3:
            image = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        else:
            image = image[0]
    
    entropy_maps = {}
    H, W = image.shape

    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]

    for patch_size in patch_sizes:
        num_patches_h = (H + patch_size - 1) // patch_size  # Round up
        num_patches_w = (W + patch_size - 1) // patch_size  # Round up

        # Pad image to ensure it fits into patches cleanly
        pad_h = num_patches_h * patch_size - H
        pad_w = num_patches_w * patch_size - W
        padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # Unfold the image into patches
        patches = padded_image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        patches = patches.reshape(num_patches_h * num_patches_w, -1)

        # Compute histograms for all patches
        bin_edges = torch.linspace(0, 255, bins + 1, device=image.device)
        histograms = torch.stack([torch.histc(patch, bins=bins, min=0, max=255) for patch in patches])

        # Normalize histograms to get probabilities
        probabilities = histograms / (patch_size * patch_size)

        # Compute entropy: -sum(p * log2(p)), avoiding log(0)
        epsilon = 1e-10
        entropy = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=1)

        # Reshape back to spatial dimensions
        entropy_map = entropy.reshape(num_patches_h, num_patches_w)

        # Assign a high value to padded regions
        if pad_h > 0:
            entropy_map[-1, :] = pad_value  # High entropy at bottom row
        if pad_w > 0:
            entropy_map[:, -1] = pad_value  # High entropy at right column

        entropy_maps[patch_size] = entropy_map

    return entropy_maps

def compute_patch_entropy_batched(images, patch_size=16, num_scales=2, bins=512, pad_value=1e6):
    """
    Compute entropy maps for multiple patch sizes in a batch of images using fully vectorized operations.
    
    Args:
        images: torch.Tensor of shape (B, C, H, W) with values in range [0, 255]
        patch_size: base patch size (default: 16)
        num_scales: number of scales to compute (default: 2)
        bins: number of bins for histogram (default: 512)
        pad_value: high entropy value to pad incomplete patches with (default: 1e6)
    
    Returns:
        batch_entropy_maps: dict containing torch.Tensor entropy maps for each patch size
                           with shape (B, H_p, W_p) where H_p and W_p depend on the patch size
    """
    batch_size, channels, H, W = images.shape
    device = images.device
    
    # Convert batch of images to grayscale using vectorized operations
    if channels == 3:
        # Apply RGB to grayscale conversion using broadcasting
        grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
        grayscale_images = (images * grayscale_weights).sum(dim=1)  # Sum across channel dimension
    else:
        grayscale_images = images[:, 0]  # Take first channel
    
    # Initialize output dictionary
    batch_entropy_maps = {}
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
    
    # Process each patch size
    for ps in patch_sizes:
        num_patches_h = (H + ps - 1) // ps  # Round up
        num_patches_w = (W + ps - 1) // ps  # Round up
        
        # Pad images to ensure they fit into patches cleanly
        pad_h = num_patches_h * ps - H
        pad_w = num_patches_w * ps - W
        padded_images = F.pad(grayscale_images, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Unfold the batch of images into patches
        # Shape: (B, num_patches_h, num_patches_w, ps, ps)
        patches = padded_images.unfold(1, ps, ps).unfold(2, ps, ps)
        
        # Reshape to (B, num_patches_h, num_patches_w, ps*ps)
        flat_patches = patches.reshape(batch_size, num_patches_h, num_patches_w, ps*ps)
        
        # Quantize the pixel values to integers in [0, bins-1]
        flat_patches_int = (flat_patches * (bins / 256.0)).long().clamp(0, bins-1)
        
        # Fully vectorized histogram computation using one-hot encoding
        # Reshape to [B * num_patches_h * num_patches_w, ps*ps]
        reshaped_patches = flat_patches_int.reshape(-1, ps*ps)
        
        # Create one-hot encoding for all pixel values
        # This creates a tensor of shape [B * num_patches_h * num_patches_w * ps*ps, bins]
        one_hot = torch.zeros(reshaped_patches.size(0), ps*ps, bins, device=device)
        one_hot = one_hot.scatter_(2, reshaped_patches.unsqueeze(2), 1)
        
        # Sum over the pixels dimension to get histograms
        # Result shape: [B * num_patches_h * num_patches_w, bins]
        histograms = one_hot.sum(1)
        
        # Reshape back to separate batch and spatial dimensions
        # Shape: [B, num_patches_h, num_patches_w, bins]
        histograms = histograms.reshape(batch_size, num_patches_h, num_patches_w, bins)
        
        # Normalize histograms to get probabilities
        probabilities = histograms.float() / (ps * ps)
        
        # Compute entropy: -sum(p * log2(p)), avoiding log(0)
        epsilon = 1e-10
        entropy_map = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=3)
        
        # Assign a high value to padded regions
        if pad_h > 0:
            entropy_map[:, -1, :] = pad_value  # High entropy at bottom row
        if pad_w > 0:
            entropy_map[:, :, -1] = pad_value  # High entropy at right column
        
        batch_entropy_maps[ps] = entropy_map
    
    return batch_entropy_maps
    
def select_patches_by_threshold(entropy_maps, thresholds, alpha=1.):
    """
    Vectorized version of patch selection based on entropy thresholds.
    
    Args:
        entropy_maps (dict): Contains patch sizes as keys mapping to
            torch.Tensor entropy maps of shape (B, H_p, W_p) where
            H_p and W_p depend on the patch size
        thresholds (list): List of thresholds for selecting patches at each scale,
            should have length = len(entropy_maps) - 1
        alpha (float): Ratio of the entropy based. 
    Returns:
        masks (dict): Dictionary mapping patch sizes to their 0/1 masks
    """
    patch_sizes = sorted(list(entropy_maps.keys()))
    
    # Special case: if num_scales == 1, just return ones-mask for the first scale
    if len(patch_sizes) == 1:
        masks = {}
        masks[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])
        return masks
    
    if len(thresholds) != len(patch_sizes) - 1:
        raise ValueError(f'Number of thresholds ({len(thresholds)}) must be one less than number of patch sizes ({len(patch_sizes)})')


    # for i in range(len(thresholds)):
    
        # thresholds[i] = thresholds[i] * alpha
    masks = {}
    # Initialize mask for smallest patch size
    masks[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])
    # eval_logger.info(masks[patch_sizes[0]])
    # Process each scale from largest to smallest
    
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]

        # maximum = entropy_maps[current_size][entropy_maps[current_size] < 1e6]
        # if maximum.numel() == 0:
        #     maximum = torch.tensor(1e6, device=entropy_maps[current_size].device)
        # minimum = entropy_maps[current_size][entropy_maps[current_size] > 0.]
        # if minimum.numel() == 0:
        #     minimum = torch.tensor(0., device=entropy_maps[current_size].device)
        
        # threshold = (maximum.max() + minimum.min()) / 2
        # threshold = entropy_maps[current_size].sum().mean()
        threshold = thresholds[i-1]
        
        # Create mask for current patch size
        masks[current_size] = (entropy_maps[current_size] < threshold).float()
        # eval_logger.info(f"patch{current_size}: {masks[current_size]}")

    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]

        for j in range(i):
            # Upscale mask to match smaller patch size
            smaller_size = patch_sizes[j]
            scale_factor = current_size // smaller_size 
            mask_upscaled = masks[current_size].repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)
            
            # Ensure upscaled mask matches the dimensions of smaller patches
            H_small, W_small = entropy_maps[smaller_size].shape[1:]  # Assuming batch dimension
            mask_upscaled = mask_upscaled[:, :H_small, :W_small]
            
            # Update mask for smaller patches
            masks[smaller_size] = masks[smaller_size] * (1 - mask_upscaled)
    masks[0] = torch.ones_like(entropy_maps[patch_sizes[0]])
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        smaller_size = patch_sizes[0]
        scale_factor = current_size // smaller_size 
        mask_upscaled = masks[current_size].repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)
        
        # Ensure upscaled mask matches the dimensions of smaller patches
        H_small, W_small = entropy_maps[smaller_size].shape[1:]  # Assuming batch dimension
        mask_upscaled = mask_upscaled[:, :H_small, :W_small]
        masks[0] = masks[0] + i * mask_upscaled
    # eval_logger.info("Final combined mask: {}", masks[0])
    return masks

def select_patches_by_threshold_v2(entropy_maps, thresholds=None, alpha=1.):
    """
    V2 版本的patch选择方法：使用最大尺寸patch熵的平均值作为所有层级的threshold基准
    
    Vectorized version of patch selection based on entropy thresholds (V2 variant).
    Uses dynamic threshold computation based on maximum patch size entropy.
    
    Args:
        entropy_maps (dict): Contains patch sizes as keys mapping to
            torch.Tensor entropy maps of shape (B, H_p, W_p) where
            H_p and W_p depend on the patch size
        thresholds (list or None): Not used in v2. If provided for compatibility, it will be ignored.
        alpha (float): Hyperparameter controlling threshold = alpha * mean(max_patch_entropy)
                      Higher alpha → more patches selected (higher recall)
                      Lower alpha → fewer patches selected (lower recall)
    Returns:
        masks (dict): Dictionary mapping patch sizes to their 0/1 masks
    """
    patch_sizes = sorted(list(entropy_maps.keys()))
    
    # Special case: if num_scales == 1, just return ones-mask for the first scale
    if len(patch_sizes) == 1:
        masks = {}
        masks[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])
        return masks
    
    # Compute dynamic threshold based on maximum patch size entropy
    max_patch_size = patch_sizes[-1]
    max_entropy_map = entropy_maps[max_patch_size]
    
    # Filter out padding values (values close to 1e6) to get true entropy statistics
    valid_entropies = max_entropy_map[max_entropy_map < 1e6]
    
    if valid_entropies.numel() == 0:
        # Fallback if all values are padding
        base_threshold = max_entropy_map.mean()
    else:
        # Use mean of valid entropies as the base threshold
        base_threshold = valid_entropies.mean()
    
    # Apply alpha hyperparameter to scale the threshold uniformly across all layers
    threshold = alpha * base_threshold
    
    masks = {}
    # Initialize mask for smallest patch size
    masks[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])
    
    # Process each scale from largest to smallest using the same threshold
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        
        # Create mask for current patch size using dynamic threshold
        masks[current_size] = (entropy_maps[current_size] < threshold).float()

    # Calculate layer relationships (ensure no overlap)
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]

        for j in range(i):
            # Upscale mask to match smaller patch size
            smaller_size = patch_sizes[j]
            scale_factor = current_size // smaller_size 
            mask_upscaled = masks[current_size].repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)
            
            # Ensure upscaled mask matches the dimensions of smaller patches
            H_small, W_small = entropy_maps[smaller_size].shape[1:]
            mask_upscaled = mask_upscaled[:, :H_small, :W_small]
            
            # Update mask for smaller patches
            masks[smaller_size] = masks[smaller_size] * (1 - mask_upscaled)
    
    masks[0] = torch.ones_like(entropy_maps[patch_sizes[0]])
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        smaller_size = patch_sizes[0]
        scale_factor = current_size // smaller_size 
        mask_upscaled = masks[current_size].repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)
        
        # Ensure upscaled mask matches the dimensions of smaller patches
        H_small, W_small = entropy_maps[smaller_size].shape[1:]
        mask_upscaled = mask_upscaled[:, :H_small, :W_small]
        masks[0] = masks[0] + i * mask_upscaled
    
    return masks


def select_patches_by_budget(entropy_maps, budget: int):
    """
    Select patches based on a global budget (number of tokens) using a 3-level scheme.

    Assumptions & mapping:
    - The smallest patch-size map in `entropy_maps` is treated as L1 (highest resolution).
    - L2 is obtained by 2x2 average-pooling of L1; L3 by 4x4 average-pooling of L1.
    - Base token count (all L3 kept) = H3 * W3. Each split (1 -> 4) increases tokens by 3.

    Args:
        entropy_maps (dict): mapping from patch_size -> torch.Tensor with shape
                             either (H, W) or (B, H, W). Must contain at least
                             the smallest (L1) resolution; this function will
                             derive L2/L3 from L1.
        budget (int): desired total token budget (per image). If batch provided,
                      treated as same budget for all images.

    Returns:
        masks (dict): mapping patch_size -> float mask (0/1) with same spatial
                      dims as corresponding entropy_maps entries. The masks
                      indicate which patches are KEPT at each scale.
    """
    patch_sizes = sorted(list(entropy_maps.keys()))
    if len(patch_sizes) < 1:
        raise ValueError('entropy_maps must contain at least one patch size')

    ps_l1 = patch_sizes[0]
    ent_l1 = entropy_maps[ps_l1]

    # Ensure batch dimension: convert (H,W) -> (1,H,W)
    is_batched = True
    if ent_l1.dim() == 2:
        ent_l1 = ent_l1.unsqueeze(0)
        is_batched = False
    elif ent_l1.dim() == 3:
        is_batched = True
    else:
        raise ValueError('entropy map must have shape (H,W) or (B,H,W)')

    device = ent_l1.device
    B, H1, W1 = ent_l1.shape

    # Compute L2 (2x2 avg pool) and L3 (4x4 avg pool)
    # FIX: Pad ent_l1 so that it is divisible by 4 (for L3), to match the padding logic
    # in construct_patch_groups (which pads image to be divisible by patch size).
    # We use the shapes of the provided entropy_maps as the ground truth for target shapes.
    
    # Determine target shapes from input entropy_maps if available
    target_h2, target_w2 = None, None
    target_h3, target_w3 = None, None
    
    if len(patch_sizes) >= 2:
        ps_l2_key = patch_sizes[1]
        if ps_l2_key in entropy_maps:
            target_h2, target_w2 = entropy_maps[ps_l2_key].shape[-2:]
            
    if len(patch_sizes) >= 3:
        ps_l3_key = patch_sizes[2]
        if ps_l3_key in entropy_maps:
            target_h3, target_w3 = entropy_maps[ps_l3_key].shape[-2:]
            
    # If targets not found in map, infer minimum required padding (round up)
    if target_h2 is None:
        target_h2 = (H1 + 1) // 2
        target_w2 = (W1 + 1) // 2
    if target_h3 is None:
        target_h3 = (H1 + 3) // 4
        target_w3 = (W1 + 3) // 4

    # Calculate required padding for L1 to cover L3 target
    # We need H1_pad / 4 >= target_h3  => H1_pad >= target_h3 * 4
    req_h = max(H1, target_h2 * 2, target_h3 * 4)
    req_w = max(W1, target_w2 * 2, target_w3 * 4)
    
    pad_h = req_h - H1
    pad_w = req_w - W1
    
    ent_l1_t = ent_l1.unsqueeze(1)  # (B,1,H1,W1)
    
    if pad_h > 0 or pad_w > 0:
        # Pad with a value that won't affect max pooling selection too much, 
        # but for avg pooling it will dilute the score. 
        # Using 0 is safe for entropy (usually positive).
        ent_l1_padded = F.pad(ent_l1_t, (0, pad_w, 0, pad_h), mode='constant', value=0)
    else:
        ent_l1_padded = ent_l1_t
    
    H_pad, W_pad = ent_l1_padded.shape[-2:]

    ent_l2 = F.avg_pool2d(ent_l1_padded, kernel_size=2, stride=2, ceil_mode=False).squeeze(1)
    ent_l3 = F.avg_pool2d(ent_l1_padded, kernel_size=4, stride=4, ceil_mode=False).squeeze(1)

    # Ensure we match target dimensions exactly (crop if padded too much due to block alignment)
    # if ent_l2.shape[-2] > target_h2 or ent_l2.shape[-1] > target_w2:
    #     ent_l2 = ent_l2[..., :target_h2, :target_w2]
    # if ent_l3.shape[-2] > target_h3 or ent_l3.shape[-1] > target_w3:
    #     ent_l3 = ent_l3[..., :target_h3, :target_w3]

    H2, W2 = ent_l2.shape[1], ent_l2.shape[2]
    H3, W3 = ent_l3.shape[1], ent_l3.shape[2]

    base_tokens = H3 * W3
    k = (budget - base_tokens) // 3
    if k <= 0:
        # No splits allowed: all kept as L3
        masks = {}
        largest_ps = patch_sizes[-1]
        
        for ps in patch_sizes:
            # Calculate target shape based on padded dimensions to ensure consistency
            # with mask_0 which forces padding in the image processor
            scale_ratio = ps // ps_l1
            h_m = H_pad // scale_ratio
            w_m = W_pad // scale_ratio
            
            if ps == largest_ps:
                m = torch.ones((B, h_m, w_m), device=device)
            else:
                m = torch.zeros((B, h_m, w_m), device=device)
                
            if not is_batched:
                m = m.squeeze(0)
            masks[ps] = m.float()

        # Construct masks[0] (Label Map) for ImageProcessor compatibility
        # When k<=0, everything is L3, so mask_0 should be all 3.0
        # We use the m from largest_ps (which is ones)
        m3 = masks[largest_ps]
        if not is_batched:
             m3 = m3.unsqueeze(0)
             
        m3_for_up = m3
        if m3_for_up.dim() == 3:
             m3_for_up = m3_for_up.unsqueeze(1) # (B, 1, H3, W3)
        
        # Note: m3 corresponds to largest_ps. 
        # If largest_ps is L3 (scale 4), we upsample by 4.
        # If largest_ps is L2 (scale 2), we upsample by 2.
        scale_factor = largest_ps // ps_l1
        
        mask_0 = F.interpolate(m3_for_up, scale_factor=scale_factor, mode='nearest').squeeze(1) * 3.0
        
        if not is_batched:
            masks[0] = mask_0.squeeze(0)
        else:
            masks[0] = mask_0

        return masks

    # Scores
    score_l2 = ent_l2
    pooled_s_l2 = F.max_pool2d(score_l2.unsqueeze(1), kernel_size=2, stride=2, ceil_mode=True).squeeze(1)
    score_l3 = torch.maximum(ent_l3, pooled_s_l2)

    keep_l1_list, keep_l2_list, keep_l3_list = [], [], []

    for b in range(B):
        sc3 = score_l3[b]
        sc2 = score_l2[b]

        cand3 = sc3.reshape(-1)
        cand2 = sc2.reshape(-1)
        candidates = torch.cat([cand3, cand2], dim=0)

        num_candidates = candidates.numel()
        kk = int(min(k, num_candidates))

        if kk <= 0:
            split3 = torch.zeros_like(sc3, dtype=torch.bool)
            split2 = torch.zeros_like(sc2, dtype=torch.bool)
        else:
            if kk >= num_candidates:
                T = candidates.min() - 1e-6
            else:
                topk = torch.topk(candidates, kk)
                T = topk.values.min()

            split3 = sc3 >= T
            split2 = sc2 >= T

        keep3 = (~split3).float()

        split3_up_l2 = F.interpolate(split3.unsqueeze(0).unsqueeze(0).float(), size=(H2, W2), mode='nearest').squeeze()
        keep2 = (split3_up_l2.bool() & (~split2)).float()

        split2_up_l1 = F.interpolate(split2.unsqueeze(0).unsqueeze(0).float(), size=(H_pad, W_pad), mode='nearest').squeeze()
        keep1 = split2_up_l1.float()

        keep_l1_list.append(keep1.unsqueeze(0))
        keep_l2_list.append(keep2.unsqueeze(0))
        keep_l3_list.append(keep3.unsqueeze(0))

    keep_l1 = torch.cat(keep_l1_list, dim=0)
    keep_l2 = torch.cat(keep_l2_list, dim=0)
    keep_l3 = torch.cat(keep_l3_list, dim=0)

    # Construct masks[0] (Label Map) for ImageProcessor compatibility
    # Upsample L2 and L3 to padded L1 size, then crop to original L1 size
    keep_l2_up = F.interpolate(keep_l2.unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1)
    # keep_l2_up = keep_l2_up[:, :H1, :W1]
    
    keep_l3_up = F.interpolate(keep_l3.unsqueeze(1), scale_factor=4, mode='nearest').squeeze(1)
    # keep_l3_up = keep_l3_up[:, :H1, :W1]
    
    mask_0 = keep_l1 * 1.0 + keep_l2_up * 2.0 + keep_l3_up * 3.0

    # Map to provided patch sizes (expecting at least three levels)
    masks = {}
    if not is_batched:
        masks[0] = mask_0.squeeze(0)
    else:
        masks[0] = mask_0

    if len(patch_sizes) >= 3:
        ps_l2 = patch_sizes[1]
        ps_l3 = patch_sizes[2]
    elif len(patch_sizes) == 2:
        ps_l2 = patch_sizes[1]
        ps_l3 = patch_sizes[1] * 2
    else:
        # For single scale, mask_0 is just keep_l1 * 1.0
        masks[ps_l1] = (keep_l1.squeeze(0) if not is_batched else keep_l1)
        return masks

    if not is_batched:
        masks[ps_l1] = keep_l1.squeeze(0).float()
        masks[ps_l2] = keep_l2.squeeze(0).float()
        masks[ps_l3] = keep_l3.squeeze(0).float()
    else:
        masks[ps_l1] = keep_l1.float()
        masks[ps_l2] = keep_l2.float()
        masks[ps_l3] = keep_l3.float()

    return masks

def visualize_selected_patches_cv2(
    image_tensor, 
    masks, 
    patch_sizes,
    color=(255, 255, 255),  # BGR in OpenCV, but white is the same in BGR or RGB
    thickness=1
):
    """
    Draw rectangles (using cv2) where masks are 1 for patches of different sizes,
    then return a PIL Image.

    Args:
        image_tensor   (torch.Tensor): Grayscale or RGB image of shape (H, W) or (C, H, W).
        masks          (List[torch.Tensor]): List of 0/1 masks for different patch sizes.
        patch_sizes    (List[int]): List of patch sizes corresponding to each mask.
        color          (tuple): BGR color for rectangle outlines (default white).
        thickness      (int): Thickness of the rectangle outline.

    Returns:
        annotated_image_pil (PIL.Image): The original image with drawn rectangles (in white).
    
    Example:
        >>> # For traditional 32/16 patches
        >>> visualize_selected_patches_cv2(image, [mask32, mask16], [32, 16])
        >>> # For arbitrary patch sizes
        >>> visualize_selected_patches_cv2(image, [mask64, mask32, mask16], [64, 32, 16])
    """

    # 1. Convert the image tensor to a NumPy array for OpenCV
    if image_tensor.ndim == 3:
        # If image_tensor is (C, H, W) with channels first
        if image_tensor.shape[0] in [1, 3]:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            # Already (H, W, C)
            image_np = image_tensor.cpu().numpy()
    else:
        # (H, W) -> expand dimension for single-channel (H, W, 1)
        image_np = image_tensor.cpu().numpy()
        # Expand to 3 channels to draw colored rectangles
        image_np = np.stack([image_np]*3, axis=-1)

    # Convert to uint8 if needed
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    # 2. Draw rectangles using cv2
    annotated_np = image_np.copy()

    # Draw patches for each mask and corresponding patch size
    for patch_size in patch_sizes:
        mask = masks[patch_size]
        H, W = mask.shape
        for i in range(H):
            for j in range(W):
                if mask[i, j] == 1:
                    top_left = (j * patch_size, i * patch_size)
                    bottom_right = (
                        top_left[0] + patch_size - 1,
                        top_left[1] + patch_size - 1,
                    )
                    cv2.rectangle(annotated_np, top_left, bottom_right, color, thickness)

    # 3. Convert back to PIL image
    #    - If your image was originally RGB, OpenCV expects BGR, but since color is white
    #      the result is the same. If you had a different color, you might want to flip channels.
    #    - For pure white rectangles, you can skip channel flipping, as (255,255,255) is the same in RGB/BGR.
    annotated_image_pil = Image.fromarray(annotated_np)

    return annotated_image_pil

def compute_patch_laplacian_vectorized(image, patch_size=16, num_scales=2, aggregate='mean', pad_mode='reflect'):
    """
    Compute Laplacian response maps for multiple patch sizes in the input image using vectorized operations.
    
    Args:
        image: torch.Tensor of shape (C, H, W) or (H, W) with values in range [0, 255]
        patch_size: base patch size (default: 16)
        num_scales: number of scales to compute, where each scale doubles the patch size (default: 2)
        aggregate: method to aggregate Laplacian values within a patch ('mean', 'max', 'std', default: 'mean')
        pad_mode: padding mode for image ('reflect', 'constant', etc., default: 'reflect')
    
    Returns:
        laplacian_maps: dict containing torch.Tensor Laplacian maps for each patch size
    """
    if len(image.shape) == 3:
        # Convert to grayscale if image is RGB
        if image.shape[0] == 3:
            image = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        else:
            image = image[0]
    
    # Normalize image to [0, 1] for consistent Laplacian response
    image = image / 255.0 if image.max() > 1.0 else image
    
    # Define Laplacian kernel (8-connected version)
    laplacian_kernel = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, -8.0, 1.0],
        [1.0, 1.0, 1.0]
    ], device=image.device).view(1, 1, 3, 3)
    
    # Apply Laplacian filter to the entire image
    # Pad the image first to maintain size
    padded_image = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode=pad_mode)
    laplacian_response = F.conv2d(padded_image, laplacian_kernel)
    
    # Take absolute values since we're interested in magnitude of change, not direction
    laplacian_response = torch.abs(laplacian_response.squeeze())
    
    laplacian_maps = {}
    H, W = image.shape
    
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
    
    for patch_size in patch_sizes:
        num_patches_h = (H + patch_size - 1) // patch_size  # Round up
        num_patches_w = (W + patch_size - 1) // patch_size  # Round up
        
        # Pad laplacian response to ensure it fits into patches cleanly
        pad_h = num_patches_h * patch_size - H
        pad_w = num_patches_w * patch_size - W
        padded_response = F.pad(laplacian_response, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Unfold the response into patches
        patches = padded_response.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        
        # Aggregate Laplacian values within each patch
        if aggregate == 'mean':
            patch_laplacian = patches.mean(dim=(2, 3))
        elif aggregate == 'max':
            patch_laplacian = patches.amax(dim=(2, 3))
        elif aggregate == 'std':
            patch_laplacian = patches.std(dim=(2, 3))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        laplacian_maps[patch_size] = patch_laplacian
    
    return laplacian_maps

def compute_patch_laplacian_batched(images, patch_size=16, num_scales=2, aggregate='mean', pad_mode='reflect'):
    """
    Compute Laplacian response maps for multiple patch sizes in a batch of images using fully vectorized operations.
    
    Args:
        images: torch.Tensor of shape (B, C, H, W) with values in range [0, 255]
        patch_size: base patch size (default: 16)
        num_scales: number of scales to compute, where each scale doubles the patch size (default: 2)
        aggregate: method to aggregate Laplacian values within a patch ('mean', 'max', 'std', default: 'mean')
        pad_mode: padding mode for image ('reflect', 'constant', etc., default: 'reflect')
    
    Returns:
        batch_laplacian_maps: dict containing torch.Tensor Laplacian maps for each patch size
                             with shape (B, H_p, W_p) where H_p and W_p depend on the patch size
    """
    batch_size, channels, H, W = images.shape
    device = images.device
    
    # Convert batch of images to grayscale
    if channels == 3:
        # Apply RGB to grayscale conversion: 0.2989 * R + 0.5870 * G + 0.1140 * B
        grayscale_images = 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]
    else:
        grayscale_images = images[:, 0]  # Take first channel
    
    # Normalize images to [0, 1] for consistent Laplacian response
    grayscale_images = grayscale_images / 255.0 if grayscale_images.max() > 1.0 else grayscale_images
    
    # Define Laplacian kernel (8-connected version)
    laplacian_kernel = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, -8.0, 1.0],
        [1.0, 1.0, 1.0]
    ], device=device).view(1, 1, 3, 3)
    
    # Apply Laplacian filter to the entire batch at once
    # Add channel dimension for conv2d
    grayscale_images = grayscale_images.unsqueeze(1)
    
    # Pad the images first to maintain size
    padded_images = F.pad(grayscale_images, (1, 1, 1, 1), mode=pad_mode)
    
    # Apply convolution to get Laplacian response
    # Shape: (B, 1, H, W)
    laplacian_responses = F.conv2d(padded_images, laplacian_kernel)
    
    # Take absolute values since we're interested in magnitude of change, not direction
    laplacian_responses = torch.abs(laplacian_responses.squeeze(1))  # Shape: (B, H, W)
    
    # Initialize output dictionary
    batch_laplacian_maps = {}
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
    
    # Process each patch size
    for ps in patch_sizes:
        num_patches_h = (H + ps - 1) // ps  # Round up
        num_patches_w = (W + ps - 1) // ps  # Round up
        
        # Pad laplacian responses to ensure they fit into patches cleanly
        pad_h = num_patches_h * ps - H
        pad_w = num_patches_w * ps - W
        padded_responses = F.pad(laplacian_responses, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Unfold the responses into patches
        # Shape: (B, num_patches_h, num_patches_w, ps, ps)
        patches = padded_responses.unfold(1, ps, ps).unfold(2, ps, ps)
        
        # Aggregate Laplacian values within each patch
        if aggregate == 'mean':
            # Mean across patch dimensions (3, 4)
            patch_laplacian = patches.mean(dim=(3, 4))
        elif aggregate == 'max':
            # Max across patch dimensions (3, 4)
            patch_laplacian = patches.amax(dim=(3, 4))
        elif aggregate == 'std':
            # Standard deviation across patch dimensions (3, 4)
            patch_laplacian = patches.std(dim=(3, 4))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        batch_laplacian_maps[ps] = patch_laplacian
    
    return batch_laplacian_maps

def compute_patch_mse_batched(images, patch_size=16, num_scales=3, scale_factors=None, aggregate='mean'):
    """
    Compute MSE response maps for multiple patch sizes in a batch of images using downsample-upsample reconstruction.
    
    Args:
        images: torch.Tensor of shape (B, C, H, W) with values in range [0, 255] or [0, 1]
        patch_size: base patch size (default: 16)
        num_scales: number of scales to compute (default: 3)
        scale_factors: list of downsample factors for each scale. If None, uses [0.5, 0.5, 0.25] pattern
        aggregate: method to aggregate MSE values within a patch ('mean', 'max', 'std', default: 'mean')
    
    Returns:
        batch_mse_maps: dict containing torch.Tensor MSE response maps for each patch size
                       with shape (B, H_p, W_p) where H_p and W_p depend on the patch size
    """
    import torch
    import torch.nn.functional as F
    
    batch_size, channels, H, W = images.shape
    device = images.device
    
    # Normalize images to [0, 1] for consistent MSE computation
    if images.max() > 1.0:
        normalized_images = images / 255.0
    else:
        normalized_images = images.clone()
    
    # Define default scale factors if not provided
    if scale_factors is None:
        # Pattern: smaller patches use less aggressive downsampling
        scale_factors = [0.5, 0.5, 0.25]  # Adjust as needed
    
    # Ensure we have enough scale factors for the number of scales
    if len(scale_factors) < num_scales:
        scale_factors.extend([scale_factors[-1]] * (num_scales - len(scale_factors)))
    
    # Generate patch sizes
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
    
    # Initialize output dictionary
    batch_mse_maps = {}
    
    # Process each patch size with its corresponding scale factor
    for i, (ps, scale_factor) in enumerate(zip(patch_sizes, scale_factors)):
        # Downsample and upsample
        img_down = F.interpolate(
            normalized_images, 
            scale_factor=scale_factor, 
            mode='bilinear', 
            align_corners=False
        )
        img_up = F.interpolate(
            img_down, 
            scale_factor=1.0/scale_factor, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Handle potential size mismatch due to rounding in interpolation
        if img_up.shape[-2:] != (H, W):
            img_up = F.interpolate(img_up, size=(H, W), mode='bilinear', align_corners=False)
        
        # Compute MSE between original and reconstructed
        # Shape: (B, C, H, W)
        mse_per_pixel = F.mse_loss(normalized_images, img_up, reduction='none')
        
        # Average across channels to get per-pixel MSE
        # Shape: (B, H, W)
        mse_map = mse_per_pixel.mean(dim=1)
        
        # Calculate patch grid dimensions
        num_patches_h = (H + ps - 1) // ps  # Round up
        num_patches_w = (W + ps - 1) // ps  # Round up
        
        # Pad MSE map to ensure it fits into patches cleanly
        pad_h = num_patches_h * ps - H
        pad_w = num_patches_w * ps - W
        padded_mse = F.pad(mse_map, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Unfold into patches
        # Shape: (B, num_patches_h, num_patches_w, ps, ps)
        patches = padded_mse.unfold(1, ps, ps).unfold(2, ps, ps)
        
        # Aggregate MSE values within each patch
        if aggregate == 'mean':
            patch_mse = patches.mean(dim=(3, 4))
        elif aggregate == 'max':
            patch_mse = patches.amax(dim=(3, 4))
        elif aggregate == 'std':
            patch_mse = patches.std(dim=(3, 4))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        batch_mse_maps[ps] = patch_mse
    
    return batch_mse_maps


def visualize_selected_patches_cv2_non_overlapping(
    image_tensor, 
    masks, 
    patch_sizes,
    color=(255, 255, 255),  # BGR in OpenCV, but white is the same in BGR or RGB
    thickness=1
):
    """
    Draw rectangles (using cv2) where masks are 1 for patches of different sizes,
    avoiding overlapping boundaries to prevent thick lines.

    Args:
        image_tensor   (torch.Tensor): Grayscale or RGB image of shape (H, W) or (C, H, W).
        masks          (List[torch.Tensor]): List of 0/1 masks for different patch sizes.
        patch_sizes    (List[int]): List of patch sizes corresponding to each mask.
        color          (tuple): BGR color for rectangle outlines (default white).
        thickness      (int): Thickness of the rectangle outline.

    Returns:
        annotated_image_pil (PIL.Image): The original image with drawn rectangles (in white).
    """

    # 1. Convert the image tensor to a NumPy array for OpenCV
    if image_tensor.ndim == 3:
        # If image_tensor is (C, H, W) with channels first
        if image_tensor.shape[0] in [1, 3]:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            # Already (H, W, C)
            image_np = image_tensor.cpu().numpy()
    else:
        # (H, W) -> expand dimension for single-channel (H, W, 1)
        image_np = image_tensor.cpu().numpy()
        # Expand to 3 channels to draw colored rectangles
        image_np = np.stack([image_np]*3, axis=-1)

    # Convert to uint8 if needed
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    # Get full image dimensions
    img_h, img_w = image_np.shape[:2]
    
    # 2. Create a set to track which edges have been drawn
    # We'll use (y1, x1, y2, x2) tuples to represent line segments
    drawn_edges = set()
    
    # 3. Create a copy of the image to draw on
    annotated_np = image_np.copy()
    
    # Process masks from largest to smallest patch size to handle hierarchy
    for patch_size_idx in range(len(patch_sizes)):
        patch_size = patch_sizes[patch_size_idx]
        mask = masks[patch_size]
        
        H, W = mask.shape
        for i in range(H):
            for j in range(W):
                if mask[i, j] == 1:
                    # Calculate patch coordinates
                    y1 = i * patch_size
                    x1 = j * patch_size
                    y2 = min(y1 + patch_size, img_h)  # Ensure we don't go beyond image bounds
                    x2 = min(x1 + patch_size, img_w)
                    
                    # Draw top edge if not already drawn
                    if (y1, x1, y1, x2) not in drawn_edges:
                        cv2.line(annotated_np, (x1, y1), (x2, y1), color, thickness)
                        drawn_edges.add((y1, x1, y1, x2))
                    
                    # Draw bottom edge if not already drawn
                    if (y2, x1, y2, x2) not in drawn_edges:
                        cv2.line(annotated_np, (x1, y2), (x2, y2), color, thickness)
                        drawn_edges.add((y2, x1, y2, x2))
                    
                    # Draw left edge if not already drawn
                    if (y1, x1, y2, x1) not in drawn_edges:
                        cv2.line(annotated_np, (x1, y1), (x1, y2), color, thickness)
                        drawn_edges.add((y1, x1, y2, x1))
                    
                    # Draw right edge if not already drawn
                    if (y1, x2, y2, x2) not in drawn_edges:
                        cv2.line(annotated_np, (x2, y1), (x2, y2), color, thickness)
                        drawn_edges.add((y1, x2, y2, x2))
    
    # 4. Convert back to PIL image
    annotated_image_pil = Image.fromarray(annotated_np)
    
    return annotated_image_pil

if __name__ == '__main__':
    # Add visualizing!
    pass
