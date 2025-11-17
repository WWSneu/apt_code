# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for Qwen2-VL."""

from typing import Optional, Union
import os
import torch
from torchvision.transforms.v2 import functional as F
import inspect
from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    logging,
)
from ...video_utils import VideoInput, make_batched_videos
from .image_processing_qwen2_vl import smart_resize
from loguru import logger as eval_logger
from apt.patch_tokenizer import PatchTokenizer
from apt.entropy_utils import visualize_selected_patches_cv2_non_overlapping, select_patches_by_threshold
import matplotlib.pyplot as plt

# from apt.patch_embed import PatchEmbed
logger = logging.get_logger(__name__)


class Qwen2VLFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
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
    """

    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]

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

@auto_docstring
class Qwen2VLImageProcessorFast(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_pixels = None
    max_pixels = None
    valid_kwargs = Qwen2VLFastImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]
    cnt = 0
    def __init__(self, **kwargs: Unpack[Qwen2VLFastImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        size = self.size if size is None else size
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        super().__init__(size=size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs)

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if min_pixels is not None and max_pixels is not None:
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        elif size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
            max_pixels = size["longest_edge"]
        else:
            size = {**self.size}

        return super()._further_process_kwargs(size=size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[Qwen2VLFastImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, videos, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        videos: VideoInput,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        To be overridden by subclasses when image-like inputs other than images should be processed.
        It can be used for segmentation maps, depth maps, etc.
        """
        caller = inspect.currentframe().f_back.f_code.co_name
        print(f"Called by: {caller}")
        # Prepare input images
        batch_feature = BatchFeature()
        if images is not None:
            images = self._prepare_image_like_inputs(
                images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
            )
            batch_feature = self._preprocess(images, **kwargs)
        if videos is not None:
            logger.warning(
                "`Qwen2VLImageProcessorFast` works only with image inputs and doesn't process videos anymore. "
                "This is a deprecated behavior and will be removed in v5.0. "
                "Your videos should be forwarded to `Qwen2VLVideoProcessor`. "
            )
            # Can't change _prepare_images_structure to work with videos because it also needs to work with images.
            videos = make_batched_videos(videos)
            videos = [
                torch.stack(self._prepare_image_like_inputs(video, do_convert_rgb, input_data_format, device))
                for video in videos
            ]
            video_outputs = self._preprocess(videos, **kwargs)
            batch_feature.update(
                {"pixel_values_videos": video_outputs.pixel_values, "video_grid_thw": video_outputs.image_grid_thw}
            )
        return batch_feature

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ):
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        # eval_logger.info(f"Grouped images keys: {grouped_images.keys()}, shapes: {[img.shape for img in grouped_images.values()]}")
        # eval_logger.info(f"indexes: {grouped_images_index}")
        
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            eval_logger.info(f"Processing shape: {shape}, patches shape before patchify: {patches.shape}")

            pt= PatchTokenizer(
                num_scales=3,
                base_patch_size=14,
                image_size=(patches.shape[-2], patches.shape[-1]),
                thresholds=[3, 2],
                mean = [0.48145466, 0.4578275, 0.40821073],
                std = [0.26862954, 0.26130258, 0.27577711],
            )
            
            # torch.set_printoptions(threshold=10000, linewidth=200)  # threshold 设置显示元素数量，linewidth 设置行宽
            pt.to(images[0].device)
            batch_maps = pt.compute_importance_maps(patches)
            eval_logger.info(f"Computed batch_maps 14: {batch_maps[14]}, Computed batch_maps 28: {batch_maps[28]}, Computed batch_maps 56: {batch_maps[56]}")
            
            masks, _, seqlens = pt.construct_masks(batch_maps)
            output_dict = pt.construct_patch_groups(patches, masks)
            output.append(output_dict)
            # output_dict = construct_patch_masks(patches, masks)
            # eval_logger.info(f"Output dict keys: {output_dict}")
            eval_logger.info(f"{masks}, {seqlens}")
            eval_logger.info(f"Constructed masks shape: {masks[14]}, seqlens: {seqlens}")
            # Save importance maps as heatmaps
            thresholds=[2.0, 1.0]
            num_scales=3
            for scale, importance_map in batch_maps.items():
                output_path = ("/data/wangwensong/dev/lmms-eval/lmms_eval/pics/heatmap/" + f"{self.cnt}_{scale}.jpg")
                save_entropy_map_as_heatmap(importance_map, output_path, (width, height))
                print(f"Saved map for scale {scale} to {output_path}")
            # # Prepare importance maps for visualization
            # for k, v in batch_maps.items():
            #     batch_maps[k] = v.unsqueeze(0)
            # Select patches based on threshold
            # all_masks = select_patches_by_threshold(batch_maps, thresholds)
            # Generate patch sizes list
            patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
            # Prepare masks for visualization (remove batch dimension)
            visualization_masks = {}
            for scale, mask in masks.items():
                visualization_masks[scale] = mask.squeeze(0)
            image_for_vis = patches[0].permute(1, 2, 0)  # 从 (C, H, W) 变为 (H, W, C)
            image_for_vis = (image_for_vis * 255).clamp(0, 255).byte()  # 转换为 uint8
            # Draw patches on the image using cv2
            vis_img = visualize_selected_patches_cv2_non_overlapping(
                image_tensor=image_for_vis,
                masks=visualization_masks,
                patch_sizes=patch_sizes,
                color=tuple((0, 255, 0)),
                thickness=1
            )
            vis_img.save(f"/data/wangwensong/dev/lmms-eval/lmms_eval/pics/selected_pics/apt_{self.cnt}.jpg")
            self.cnt += 1

            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
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
            # Reorder dimensions to group grid and patch information for subsequent flattening.
            # (batch, grid_t, grid_h, grid_w, merge_h, merge_w, channel, temp_patch_size, patch_h, patch_w)
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )
        # return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "output": output}
    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

        Note: Do not remove this method! It is used by vLLM to infer the number of patches and placeholders
        without an image input.

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


__all__ = ["Qwen2VLImageProcessorFast"]
