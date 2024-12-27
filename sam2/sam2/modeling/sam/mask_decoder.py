# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer, 256
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1 # 4
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder, 1 x 256 x 64 x 64
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings, 1 x 256 x 64 x 64, sin and cos
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes, 1x(N+1)x256, or Bx(2+1)x256, or Bx(2+N+1)x256
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs, 1/B x 256 x 64 x 64
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
          high_res_features (List[torch.Tensor]): the high resolution features, 1 x 32 x 256 x 256 -> 1 x 64 x 128 x 128

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        # masks: 1/B x 4 x 256 x 256
        # iou_pred: 1/B x 4, after sigmoid
        # mask_tokens_out: 1/B x 4 x 256
        # object_score_logits: 1/B x 1
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # image_embeddings: 1 x 256 x 64 x 64
        # image_pe: 1 x 256 x 64 x 64
        # sparse_prompt_embeddings: 1x(N+1)x256, or Bx(2+1)x256, or Bx(2+N+1)x256
        # dense_prompt_embeddings: 1/B x 256 x 64 x 64
        # repeat_image: whether it's multi object prediction
        # high_res_features: 1 x 32 x 256 x 256 -> 1 x 64 x 128 x 128
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight, # 1 x 256
                    self.iou_token.weight, # 1 x 256
                    self.mask_tokens.weight, # 4 x 256
                ],
                dim=0,
            ) # 6 x 256
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            ) # 5 x 256
        # multi-object prediction shares the same output tokens
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        ) # 1/B x 6/5 x 256, point prompts can only specify one object
        # 1/B x (6/5+(N+1)/(2+1)/(2+N+1)) x 256
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            # repeat the number of box times for each image
            # 1/B x 256 x 64 x 64
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        # if no mask prompts and multiple boxes,
        # dense_prompt_embeddings (1 x 256) are no_mask_embed.weight repeated B x 64 x 64 times
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        # 1/B x 256 x 64 x 64
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        # hs: 1/B x 6/5+(N+1)/(2+1)/(2+N+1) x 256
        # src: 1/B x 64*64 x 256
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :] # 1/B x 256
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :] # 1/B x 4 x 256

        # Upscale mask embeddings and predict masks using the mask tokens
        # 1/B x 256 x 64 x 64
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features # 1 x 32 x 256 x 256 -> 1 x 64 x 128 x 128
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1)) # 1/B x 64 x 128 x 128
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0) # 1/B x 32 x 256 x 256

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                # reduce 8x channels 
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) 
            )
        hyper_in = torch.stack(hyper_in_list, dim=1) # 1/B x 4 x 32
        b, c, h, w = upscaled_embedding.shape
        # 1/B x 4 x 256 x 256
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        # after sigmoid
        iou_pred = self.iou_prediction_head(iou_token_out) # 1/B x 4, 4 is the number of mask tokens
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :]) # 1/B x 1
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        # masks: 1/B x 4 x 256 x 256
        # iou_pred: 1/B x 4
        # mask_tokens_out: 1/B x 4 x 256
        # object_score_logits: 1/B x 1
        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2) # 1/B x 1 x 256*256
        stability_delta = self.dynamic_multimask_stability_delta
        # how many pixels are above stability_delta and -stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float() # 1/B x 1
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float() # 1/B x 1
        # the higher the stability_scores, the lower the number of pixels whose logits are between -stability_delta and stability_delta
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # all_mask_logits: 1/B x 4 x 256 x 256
        # all_iou_scores: 1/B x 4, after sigmoid

        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds] 
        best_multimask_logits = best_multimask_logits.unsqueeze(1) # 1/B x 1 x 256 x 256
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1) # 1/B x 1

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :] # 1/B x 1 x 256 x 256
        singlemask_iou_scores = all_iou_scores[:, 0:1] # 1/B x 1
        stability_scores = self._get_stability_scores(singlemask_logits) # 1/B x 1
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh # 1/B x 1

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits), # 1/B x 1 x 256 x 256
            singlemask_logits, # select between 0-th output mask token and best among 1-3 output mask tokens
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
