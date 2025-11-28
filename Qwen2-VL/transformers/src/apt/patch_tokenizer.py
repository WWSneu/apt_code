"""
Stand-alone module that computes the necessary inputs for the 
patch embedding. The goal is to compute only the parts 
that are needed, nothing more, and place them in a dictionary 
for the patch embedding to use. 
"""
from typing import Dict, List, Tuple, Union
import ipdb
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import math
from loguru import logger as eval_logger

from apt.entropy_utils import (
    select_patches_by_threshold,
    select_patches_by_threshold_v2,
    visualize_selected_patches_cv2,
    select_patches_by_budget,
    compute_patch_entropy_vectorized,
    compute_patch_entropy_batched,
    compute_patch_laplacian_vectorized,
    compute_patch_laplacian_batched,
    compute_patch_mse_batched
)


class PatchTokenizer(nn.Module):
    """Tokenizer for mixed-resolution patches.
    
    Args:
        num_scales (int): Number of scales to use
        base_patch_size (int): Base patch size
        image_size (int): Image size
        thresholds (List[float]): Entropy thresholds for patch selection at each scale
        mean (List[float]): Mean values for normalization
        std (List[float]): Standard deviation values for normalization
        method (str): Method to use for computing patch importance maps ('entropy' or 'laplacian')
        laplacian_aggregate (str): Method to aggregate Laplacian values ('mean', 'max', or 'std')
        patch_selection_method (str): Patch selection method ('v1' or 'v2', default: 'v1')
    """
    def __init__(
        self,
        num_scales: int,
        base_patch_size: int,
        image_size: int,
        thresholds: List[float],
        mean: List[float],
        std: List[float],
        method: str = 'entropy',
        laplacian_aggregate: str = 'mean',
        patch_selection_method: str = 'v1',
        alpha: float = 1.0,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.base_patch_size = base_patch_size
        self.image_size = image_size
        self.thresholds = thresholds
        self.method = method
        self.laplacian_aggregate = laplacian_aggregate
        self.patch_selection_method = patch_selection_method
        self.alpha = alpha
        # Filled when using budget mode: {'base_tokens': int, 'k': int, 'budget': int}
        self.last_budget_info = None

        if patch_selection_method not in ['v1', 'v2', 'budget']:
            raise ValueError(f"patch_selection_method must be 'v1', 'v2' or 'budget', got '{patch_selection_method}'")

        self.pos_embed16: Union[torch.Tensor, None] = None
        self.pos_embed32: Union[torch.Tensor, None] = None

        self.norm = transforms.Normalize(mean=mean, std=std)
        self.unnorm = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )


    def construct_masks(
        self,
        importance_maps: Dict[int, torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[int]]:
        """Constructs selection masks for patches based on importance maps.
        
        Args:
            importance_maps (Dict[int, torch.Tensor]): Dictionary mapping patch sizes to their
                importance maps. Shape of each map: (batch_size, height, width)
        
        Returns:
            Tuple containing:
                - masks (Dict[int, torch.Tensor]): Dictionary of selection masks for each
                  patch size
                - output_mask (torch.Tensor): Flattened mask with scale indicators
                  (-1 for class token)
                - seqlens (List[int]): Sequence lengths for each batch item
        """
        # Use selected patch selection method
        if self.patch_selection_method == 'v1':
            masks = select_patches_by_threshold(importance_maps, thresholds=self.thresholds)
        elif self.patch_selection_method == 'v2':
            # v2 uses dynamic thresholds based on alpha hyperparameter
            # Ignore self.thresholds and use alpha instead
            masks = select_patches_by_threshold_v2(importance_maps, thresholds=None, alpha=self.alpha)
        elif self.patch_selection_method == 'budget':
            # budget mode: interpret alpha as fraction of the total smallest-patch tokens
            # i.e., alpha = retained_small_fraction (0..1), so desired budget = alpha * (H1*W1)
            # base_tokens remains number of tokens if all kept at largest scale (L3)
            ps_sorted = sorted(list(importance_maps.keys()))
            smallest_ps = ps_sorted[0]
            largest_ps = ps_sorted[-1]

            # shapes
            smallest_map = importance_maps[smallest_ps]
            largest_map = importance_maps[largest_ps]
            H1, W1 = smallest_map.shape[1], smallest_map.shape[2]
            H3, W3 = largest_map.shape[1], largest_map.shape[2]
            
            base_small_tokens = H1 * W1
            base_tokens = H3 * W3
            eval_logger.info(f"!!!!!!!!!!!Acceptable minimum alpha for budget mode: {base_tokens / base_small_tokens:.4f}")

            # Desired budget = fraction * base_small_tokens
            budget = int(max(0, round(self.alpha * float(base_small_tokens))))

            # k: maximum splits allowed (each split increases tokens by 3, starting from all-L3)
            k = (budget - base_tokens) // 3
            eval_logger.info(f"!!!!!!!!!!!Budget mode: base_tokens={base_tokens}, base_small_tokens={base_small_tokens}, budget={budget}, k={k}")
            if k < 0:
                k = 0

            # record for later inspection by caller (visualize script)
            self.last_budget_info = {
                'base_tokens': int(base_tokens),
                'base_small_tokens': int(base_small_tokens),
                'k': int(k),
                'budget': int(budget)
            }

            masks = select_patches_by_budget(importance_maps, budget=budget)
        else:
            raise ValueError(f"Unknown patch_selection_method: {self.patch_selection_method}")
        
        batch_size = masks[self.base_patch_size].shape[0]
        # eval_logger.info(f"batch_size: {batch_size}")
        all_masks = []
        output_dict = {}
        
        # Set up output mask with class token (-1)
        device = importance_maps[self.base_patch_size].device
        temp_masks = [torch.ones((batch_size, 1), device=device) * -1]
        # eval_logger.info(f"temp_masks: {temp_masks}")

        seqlens = torch.ones((batch_size), device=device)
        # eval_logger.info(f"seqlens init: {seqlens}")
        
        for idx in range(0, self.num_scales):
            cur_patch_size = self.base_patch_size * 2 ** idx
            temp_mask = masks[cur_patch_size].flatten(1)
            # eval_logger.info(f"temp_mask for patch size {cur_patch_size}: {temp_mask.sum(1)}")
            seqlens += temp_mask.sum(1)
            temp_masks.append(temp_mask * (idx + 1))

        output_mask = torch.cat(temp_masks, dim=1)
        # eval_logger.info(f"output_mask before filtering: {output_mask}")
        output_mask = output_mask[output_mask != 0]
        # eval_logger.info(f"output_mask after filtering: {output_mask}")
        # eval_logger.info(f"seqlens before int: {seqlens}")
        seqlens = seqlens.int().tolist()
        # eval_logger.info(f"seqlens after int: {seqlens}")
        
        return masks, output_mask, seqlens

    def construct_patch_groups(
        self,
        images: torch.Tensor,
        masks: Dict[int, torch.Tensor],
        # pos_embeds: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Constructs groups of patches at different scales with their position embeddings.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, channels, height, width)
            masks (Dict[int, torch.Tensor]): Selection masks for each patch size
            pos_embeds (Dict[str, torch.Tensor]): Position embeddings for each patch size
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - full_patches_{size}: Original resolution patches for each size
                - resized_patches_{size}: Downsampled patches for each size
                - pos_embed_{size}: Position embeddings for selected patches
                - pos_embed_cls_token: Position embedding for class token
        """
        output_dict = {}
        B = images.shape[0]

        for idx in range(0, self.num_scales):
            # eval_logger.info(f"id: {idx}")

            cur_patch_size = self.base_patch_size * 2 ** idx
            cur_mask = masks[cur_patch_size].bool()

            # eval_logger.info(f"cur_mask for patch size {cur_patch_size} shape: {cur_mask.shape}, cur_mask {cur_mask}")
            
            # cur_pos_embed = pos_embeds[str(cur_patch_size)]
            # cur_pos_embed = cur_pos_embed[:, 1:].repeat(B, 1, 1)

            batch_size, channels, H, W = images.shape

            num_patches_h = (H + cur_patch_size - 1) // cur_patch_size  # Round up
            num_patches_w = (W + cur_patch_size - 1) // cur_patch_size  # Round up

            # Pad image to ensure it fits into patches cleanly
            pad_h = num_patches_h * cur_patch_size - H
            pad_w = num_patches_w * cur_patch_size - W
            padded_image = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='constant', value=0)
            scale_img = padded_image

            # eval_logger.info(f"padded_image shape: {padded_image.shape}")

            if idx > 0:
                scale_img = F.interpolate(
                    scale_img,
                    scale_factor=0.5 ** idx,
                    mode="bilinear"
                )

                # eval_logger.info(f"scale_img shape after interpolation: {scale_img.shape}")

                constituent_patches = einops.rearrange(
                    padded_image,
                    "b c (h n1 p3) (w n2 p4) -> b h w (n1 n2) c p3 p4",
                    h=num_patches_h,
                    w=num_patches_w,
                    n1=cur_patch_size // self.base_patch_size,
                    n2=cur_patch_size // self.base_patch_size,
                    p3=self.base_patch_size,
                    p4=self.base_patch_size
                )

                # eval_logger.info(f"constituent_patches shape: {constituent_patches.shape}")
                
                # eval_logger.info(f"constituent_patches: {constituent_patches}")
                # eval_logger.info(f"cur_mask shape: {cur_mask.shape}")
                # eval_logger.info(f"cur_mask: {cur_mask}")

                selected_constituent_patches = constituent_patches[cur_mask]
                output_dict[f"full_patches_{cur_patch_size}"] = selected_constituent_patches
                # eval_logger.info(f"full_patches {cur_patch_size} shape: {selected_constituent_patches.shape}")
            scaled_patches = einops.rearrange(
                scale_img, 
                "b c (h p1) (w p2) -> b h w c p1 p2",
                p1=self.base_patch_size,
                p2=self.base_patch_size
            )
            # eval_logger.info(f"scaled_patches shape: {scaled_patches.shape}")
            # eval_logger.info(f"scaled_patches: {scaled_patches}")

            selected_patches = scaled_patches[masks[cur_patch_size].bool()]

            # eval_logger.info(f"selected_patches shape: {selected_patches.shape}")
            # eval_logger.info(f"selected_patches: {selected_patches}")
            output_dict[f"resized_patches_{cur_patch_size}"] = selected_patches
            # eval_logger.info(f"resized_patches stored for patch size {cur_patch_size}")
            flat_mask = masks[cur_patch_size].flatten(1).bool()
            output_dict[f"pos_embed_mask_{cur_patch_size}"] = flat_mask
            # eval_logger.info(f"pos_embed_mask shape: {flat_mask.shape}")

        #output_dict["pos_embed_cls_token"] = pos_embeds[str(self.base_patch_size)][:, 0]
        return output_dict

    def compute_importance_maps(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Compute importance maps for images after unnormalizing them.
        Uses either entropy or Laplacian method based on the tokenizer configuration.
        
        Args:
            images: Normalized images tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary mapping patch sizes to importance maps with shape (B, H_p, W_p)
        """
        # Unnormalize the images using vectorized operations
        with torch.no_grad():
            eval_logger.info(f"images is normed? {images}")
            # Apply unnormalization directly to the batch
            unnormalized_images = self.unnorm(images)
            eval_logger.info(f"unnormalized_images: {unnormalized_images}")
            # Scale to [0, 255] range for computation
            unnormalized_images = torch.clamp(unnormalized_images * 255.0, 0, 255)
            
            # Compute maps based on selected method
            if self.method == 'entropy':
                # Compute entropy maps for the entire batch
                batch_maps = compute_patch_entropy_batched(
                    unnormalized_images, 
                    patch_size=self.base_patch_size, 
                    num_scales=self.num_scales
                )
            elif self.method == 'laplacian':
                # Compute Laplacian maps for the entire batch
                batch_maps = compute_patch_laplacian_batched(
                    unnormalized_images, 
                    patch_size=self.base_patch_size, 
                    num_scales=self.num_scales,
                    aggregate=self.laplacian_aggregate
                )
            elif self.method == 'upsample_mse':
                batch_maps = compute_patch_mse_batched(
                    unnormalized_images, 
                    patch_size=self.base_patch_size, 
                    num_scales=self.num_scales,
                    scale_factors=[1.0, 0.5, 0.25],
                    aggregate='mean'
                )
            else:
                raise ValueError(f"Unknown method: {self.method}. Choose 'entropy' or 'laplacian'")
            
        return batch_maps

    def forward(
        self,
        images: torch.Tensor,
        importance_maps: Dict[int, torch.Tensor] = None,
        pos_embeds: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, List[int]]]:
        """Forward pass of the patch tokenizer.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, channels, height, width)
            importance_maps (Dict[int, torch.Tensor]): Pre-computed importance maps for different patch sizes
            pos_embeds (Dict[str, torch.Tensor]): Position embeddings for different patch sizes
            
        Returns:
            Dictionary containing:
                - All outputs from construct_patch_groups
                - output_mask: Mask indicating patch scales
                - seqlens: Sequence lengths for each batch item
        """
        B, C, H, W = images.shape
        max_patches = B * H * W / (self.base_patch_size ** 2)
        
        # If importance maps are not provided, compute them

        # PRECOMPTUTE
        if self.method != "entropy":
            importance_maps = self.compute_importance_maps(images)
        
        masks, output_mask, seqlens = self.construct_masks(importance_maps)
        output_dict = self.construct_patch_groups(images, masks)
        output_dict["output_mask"] = output_mask
        output_dict["seqlens"] = seqlens
        
        # Compute cu_seqlens and max_seqlen for flash attention varlen
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=images.device),
                                torch.tensor(seqlens, dtype=torch.int32, device=images.device).cumsum(0)])
        output_dict["cu_seqlens"] = cu_seqlens
        output_dict["max_seqlen"] = max(seqlens)

        retained_patches = sum(seqlens)
        output_dict["retained_frac"] = retained_patches / max_patches

        return output_dict