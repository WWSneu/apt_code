# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for Qwen2-VL."""

import math
from typing import Optional, Union
import torch
import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, logging
from ...video_utils import VideoInput, make_batched_videos
from apt.patch_tokenizer import PatchTokenizer
from apt.entropy_utils import visualize_selected_patches_cv2_non_overlapping, select_patches_by_threshold
from loguru import logger as eval_logger

logger = logging.get_logger(__name__)

def save_entropy_map_as_heatmap(entropy_map, output_path, original_size):
    """Save entropy map as a matplotlib heatmap resized to the original image size."""
    # Convert entropy map to numpy array
    entropy_np = entropy_map.squeeze(0).cpu().numpy()  # 去掉 batch 维度
    
    # Create figure without axes
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    
    # Create heatmap with inferno colormap
    plt.imshow(entropy_np, cmap='inferno')
    plt.tight_layout(pad=0)
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # # Resize the saved image to match the original image size
    # saved_img = Image.open(output_path)
    # saved_img = saved_img.resize(original_size, Image.LANCZOS)
    # saved_img.save(output_path)

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
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


class Qwen2VLImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Qwen2-VL image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        size (`dict[str, int]`, *optional*, defaults to `{"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}`):
            Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
        patch_selection_method (`str`, *optional*, defaults to 'v1'):
            Patch selection method ('v1', 'v2', or 'budget').
        alpha (`float`, *optional*, defaults to 1.0):
            Hyperparameter for patch selection (v2: threshold scaling; budget: retention fraction).
    """

    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        patch_selection_method: str = 'v1',
        alpha: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        else:
            size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
        self.min_pixels = size["shortest_edge"]
        self.max_pixels = size["longest_edge"]
        self.size = size

        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb
        self.patch_selection_method = patch_selection_method
        self.alpha = alpha

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        patch_selection_method: Optional[str] = None,
        alpha: Optional[float] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`list[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """


        images = make_flat_list_of_images(images)
        device = images[0].device if hasattr(images[0], "device") else None
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]
        
        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        # eval_logger.info(f"processed_images shape: {np.array(processed_images).shape}")
        patches = processed_images
        # patches.to(device)
        
        # Set defaults for patch selection parameters
        patch_selection_method = patch_selection_method if patch_selection_method is not None else self.patch_selection_method
        alpha = alpha if alpha is not None else self.alpha
        
        pt= PatchTokenizer(
            num_scales=3,
            base_patch_size=14,
            image_size=(patches[0].shape[-2], patches[0].shape[-1]),
            thresholds=[3, 2],
            mean = [0.48145466, 0.4578275, 0.40821073],
            std = [0.26862954, 0.26130258, 0.27577711],
            patch_selection_method=patch_selection_method,
            alpha=alpha,
        )
        patches = np.array(patches)  # 转换为 NumPy 数组
        patches = torch.from_numpy(patches).to(device)        # torch.set_printoptions(threshold=10000, linewidth=200)  # threshold 设置显示元素数量，linewidth 设置行宽
        pt.to(device)
        batch_maps = pt.compute_importance_maps(patches)
        # eval_logger.info(f"Computed batch_maps 14: {batch_maps[14]}, Computed batch_maps 28: {batch_maps[28]}, Computed batch_maps 56: {batch_maps[56]}")
        # eval_logger.info(f"batch_maps 14 shape: {batch_maps[14].shape}, batch_maps 28 shape: {batch_maps[28].shape}, batch_maps 56 shape: {batch_maps[56].shape}")
        # output = []
        
        masks, _, seqlens = pt.construct_masks(batch_maps)

        # if patch_selection_method == 'budget':
        #      # Pad patches tensor to match mask dimensions
        #      # masks[0] is the label map with shape (B, H_p, W_p) corresponding to base_patch_size
        #      mask_h, mask_w = masks[0].shape[-2:]
        #      current_h, current_w = patches.shape[-2] // patch_size, patches.shape[-1] // patch_size
             
        #      if mask_h > current_h or mask_w > current_w:
        #          pad_h_pixels = (mask_h - current_h) * patch_size
        #          pad_w_pixels = (mask_w - current_w) * patch_size
        #          patches = torch.nn.functional.pad(patches, (0, pad_w_pixels, 0, pad_h_pixels), mode='constant', value=0)

        # output_dict = pt.construct_patch_groups(patches, masks)

        # output.append(output_dict)

        # # output_dict = construct_patch_masks(patches, masks)
        # # eval_logger.info(f"Output dict keys: {output_dict}")
        # eval_logger.info(f"{masks}, {seqlens}")
        # eval_logger.info(f"Constructed masks shape: {masks[14]}, seqlens: {seqlens}")
        # # Save importance maps as heatmaps
        # thresholds=[2.0, 1.0]
        # num_scales=3
        # for scale, importance_map in batch_maps.items():
        #     output_path = ("/data/wangwensong/dev/lmms-eval/lmms_eval/pics/heatmap/" + f"{self.cnt}_{scale}.jpg")
        #     save_entropy_map_as_heatmap(importance_map, output_path, (width, height))
        #     print(f"Saved map for scale {scale} to {output_path}")
        # # Prepare importance maps for visualization
        # for k, v in batch_maps.items():
        #     batch_maps[k] = v.unsqueeze(0)
        # Select patches based on threshold
        # all_masks = select_patches_by_threshold(batch_maps, thresholds)
        # Generate patch sizes list
        # patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
        # Prepare masks for visualization (remove batch dimension)
        # visualization_masks = {}
        # for scale, mask in masks.items():
        #     visualization_masks[scale] = mask.squeeze(0)
        # image_for_vis = patches[0].permute(1, 2, 0)  # 从 (C, H, W) 变为 (H, W, C)
        # image_for_vis = (image_for_vis * 255).clamp(0, 255).byte()  # 转换为 uint8
        # Draw patches on the image using cv2
        # vis_img = visualize_selected_patches_cv2_non_overlapping(
        #     image_tensor=image_for_vis,
        #     masks=visualization_masks,
        #     patch_sizes=patch_sizes,
        #     color=tuple((0, 255, 0)),
        #     thickness=1
        # )
        # vis_img.save(f"/data/wangwensong/dev/lmms-eval/lmms_eval/pics/selected_pics/apt_{self.cnt}.jpg")
        # self.cnt += 1

        patches = np.array(processed_images)
        # eval_logger.info(f"patchs shape patches = np.array(processed_images): {patches.shape}")
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        
        # Check if padding is needed based on masks[0] (for budget mode)
        if patch_selection_method == 'budget':
            mask_h, mask_w = masks[0].shape[-2:]
            current_h, current_w = patches.shape[-2] // patch_size, patches.shape[-1] // patch_size
            
            if mask_h > current_h or mask_w > current_w:
                pad_h_pixels = (mask_h - current_h) * patch_size
                pad_w_pixels = (mask_w - current_w) * patch_size
                
                # Pad patches (numpy array)
                # patches: (B, C, H, W)
                # Pad H and W with zeros
                # np.pad expects ((before_1, after_1), (before_2, after_2), ...)
                # Axis 0: Batch, Axis 1: Channel, Axis 2: Height, Axis 3: Width
                patches = np.pad(patches, ((0,0), (0,0), (0, pad_h_pixels), (0, pad_w_pixels)), mode='constant', constant_values=0)
                
                # Update resized_height/width for subsequent calculations
                resized_height = patches.shape[-2]
                resized_width = patches.shape[-1]

        if patches.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis], temporal_patch_size - (patches.shape[0] % temporal_patch_size), axis=0
            )
            patches = np.concatenate([patches, repeats], axis=0)
        # eval_logger.info(f"patches shape repeats = np.repeat(: {patches.shape}")

        # eval_logger.info(f"masks shape before reshape: {masks[0].shape}")

        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # eval_logger.info(f"masks before: {masks[0]}")
        masks[0] = masks[0].squeeze(0)
        cnt = {1:0, 2:0, 3:0}
        for i in range(masks[0].shape[0]):
            for j in range(masks[0].shape[1]):
                if masks[0][i, j].item() == 1:
                    cnt[1] += 1
                elif masks[0][i, j].item() == 2:
                    cnt[2] += 1
                elif masks[0][i, j].item() == 3:
                    cnt[3] += 1
        # for i in range(3):
        #     eval_logger.info(f"mask num {i+1} count: {cnt[i+1]}")
        sum = int(cnt[1] + cnt[2]/4 + cnt[3]/16)
        # eval_logger.info(f"sum in imageprocessor: {sum}")
        masks[1] = masks[0].reshape(
            grid_t,
            1,
            1,
            grid_h // merge_size,
            merge_size,
            1,
            grid_w // merge_size,
            merge_size,
            1,
        )
        # eval_logger.info(f"masks shape after reshape: {masks[0].shape}")
        # eval_logger.info(f"masks after: {masks[0]}")
        # eval_logger.info(f"patches shape after reshape: {patches.shape}")
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )
        # flatten_patches = patches.reshape(grid_t * grid_h * grid_w, -1)
        masks[1] = masks[1].permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        masks[1] = masks[1].reshape(
            grid_t * grid_h * grid_w, 1
        )
        # eval_logger.info(f"flatten_patches shape: {flatten_patches.shape}")
        # eval_logger.info(f"flatten_masks0 shape: {masks[0].shape}")
        # eval_logger.info(f"flatten_patches: {flatten_patches}")
        # eval_logger.info(f"flatten_masks0: {masks[0]}")
        # if sum%4 !=0:
        #     sum = sum + (4 - sum%4)       #flag: we ceil to multiple of 4. preparing for later padding in model forward.<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # # sum = sum - (sum%4)

        # #flag: DEBUG.>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # sum=sum*4
        
        # total = sum
        # eval_logger.info("sum in imageprocessor: {}", sum)
        # # flatten_patches = flatten_patches[:sum, :]
        # # eval_logger.info("total patches after merge: {}", total)
        # factors = []
        # for i in range(1, int(math.sqrt(total)) + 1):
        #     if total % i == 0:
        #         h, w = i, total // i 
        #         factors.append((h, w))
        # even_factors = [(h, w) for h, w in factors if h % 2 == 0 and w % 2 == 0]
        # best_pair = min(even_factors, key=lambda x: abs(x[0] - x[1]))
        # height, width = best_pair
    
        # # 返回 [1, h, w]
        # grid_new_thw = torch.tensor([1, height, width], dtype=torch.long, device=device).unsqueeze(0)
        grid_new_thw = []
        # Log stats
        try:
            orig_h, orig_w = get_image_size(images[0], channel_dim=input_data_format)
            orig_patches_14 = (orig_h // 14) * (orig_w // 14)
            resized_patches_14 = (resized_height // 14) * (resized_width // 14)
            eval_logger.info(
                f"Image Stats: "
                f"Original: {orig_h}x{orig_w} ({orig_patches_14} patches), "
                f"Resized: {resized_height}x{resized_width} ({resized_patches_14} patches), "
            )
        except Exception as e:
            eval_logger.warning(f"Failed to log image stats: {e}")

        return flatten_patches, (grid_t, grid_h, grid_w), processed_images, masks, grid_new_thw

    def preprocess(
        self,
        images: ImageInput,
        videos: Optional[VideoInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        patch_selection_method: Optional[str] = None,
        alpha: Optional[float] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            videos (`VideoInput`):
                Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
                passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            min_pixels (`int`, *optional*, defaults to `self.min_pixels`):
                The min pixels of the image to resize the image.
            max_pixels (`int`, *optional*, defaults to `self.max_pixels`):
                The max pixels of the image to resize the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            patch_selection_method (`str`, *optional*, defaults to `self.patch_selection_method`):
                Patch selection method ('v1', 'v2', or 'budget').
            alpha (`float`, *optional*, defaults to `self.alpha`):
                Hyperparameter for patch selection (v2: threshold scaling; budget: retention fraction).
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        
        """
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
        elif min_pixels is not None and max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        do_resize = do_resize if do_resize is not None else self.do_resize

        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        patch_selection_method = patch_selection_method if patch_selection_method is not None else self.patch_selection_method
        alpha = alpha if alpha is not None else self.alpha

        if images is not None:
            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        data = {}
        processed_images = []
        # pt = PatchTokenizer(
        #     num_scales=3,
        #     base_patch_size=14,
        #     thresholds=[4.0, 6.0],
        #     mean = [0.485, 0.456, 0.406],
        #     std = [0.229, 0.224, 0.225],
        # )
        if images is not None:
            pixel_values, vision_grid_thws, input = [], [], []
            for image in images:
                # eval_logger.info(f"Processing image 0: {image}, shape: {image.shape}")
                patches, image_grid_thw, processed_image, input_dict, grid_new_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    do_convert_rgb=do_convert_rgb,
                    patch_selection_method=patch_selection_method,
                    alpha=alpha,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                # eval_logger.info(f"Processed image 1: {image}, shape: {image.shape}")
                processed_images.append(processed_image)
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
                input.append(input_dict)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            input = np.array(input)
            data.update({"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws})

        # kept for BC only and should be removed after v5.0
        if videos is not None:
            logger.warning(
                "`Qwen2VLImageProcessor` works only with image inputs and doesn't process videos anymore. "
                "This is a deprecated behavior and will be removed in v5.0. "
                "Your videos should be forwarded to `Qwen2VLVideoProcessor`. "
            )
            videos = make_batched_videos(videos)
            pixel_values_videos, vision_grid_thws_videos = [], []
            for images in videos:
                patches, video_grid_thw = self._preprocess(
                    images,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values_videos.extend(patches)
                vision_grid_thws_videos.append(video_grid_thw)
            data.update(
                {
                    "pixel_values_videos": np.array(pixel_values_videos),
                    "video_grid_thw": np.array(vision_grid_thws_videos),
                }
            )

        return BatchFeature(data=data, tensor_type=return_tensors), input, processed_images, grid_new_thw

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        min_pixels = images_kwargs["min_pixels"] if "min_pixels" in images_kwargs else self.size["shortest_edge"]
        max_pixels = images_kwargs["max_pixels"] if "max_pixels" in images_kwargs else self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


__all__ = ["Qwen2VLImageProcessor"]
