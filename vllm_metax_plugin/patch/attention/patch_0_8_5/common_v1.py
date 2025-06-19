# SPDX-License-Identifier: Apache-2.0

# TODO: remove this file

import vllm
from vllm_metax_plugin.patch.hook_registry import register_patch
from vllm import envs
from vllm.v1.attention.backends.mla.common import logger

import torch
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from flash_attn import flash_attn_varlen_func

def get_flash_attn_version():
    return None

def _flash_attn_varlen_diff_headdims(self,
                                        q,
                                        k,
                                        v,
                                        return_softmax_lse=False,
                                        softmax_scale=None,
                                        **kwargs):
    logger.info(f"[Plugin] Hooked _flash_attn_varlen_diff_headdims -> {_flash_attn_varlen_diff_headdims}")
    maybe_padded_v = v
    if self._pad_v:
        maybe_padded_v = torch.nn.functional.pad(
            v, [0, q.shape[-1] - v.shape[-1]], value=0)

    attn_out = self.flash_attn_varlen_func(
        q=q,
        k=k,
        v=maybe_padded_v,
        return_attn_probs=return_softmax_lse,
        softmax_scale=softmax_scale,
        **kwargs,
    )

    # Unpack the output if there is multiple results
    lse = None
    if isinstance(attn_out, tuple):
        attn_out, lse = attn_out[0], attn_out[1]

    # unpad if necessary
    if self._pad_v:
        attn_out = attn_out[..., :v.shape[-1]]

    # Remain consistent with old `flash_attn_varlen_func` where there
    # is only one output tensor if `return_softmax_lse` is False.
    if return_softmax_lse:
        return attn_out, lse
    return attn_out

def process_weights_after_loading(self, act_dtype: torch.dtype):
    logger.info(f"[Plugin] Hooked process_weights_after_loading -> {process_weights_after_loading}")

    def get_layer_weight(layer):
        WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
        for attr in WEIGHT_NAMES:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise AttributeError(
            f"Layer '{layer}' has no recognized weight attribute:"
            f" {WEIGHT_NAMES}.")

    def get_and_maybe_dequant_weights(layer: LinearBase):
        if not isinstance(layer.quant_method, UnquantizedLinearMethod):
            # NOTE: This should only be used offline, since it's O(N^3)
            eye = torch.eye(layer.input_size_per_partition,
                            dtype=act_dtype,
                            device=get_layer_weight(layer).device)
            dequant_weights = layer.quant_method.apply(layer,
                                                        eye,
                                                        bias=None)
            del eye
            # standardize to (output, input)
            return dequant_weights.T
        return layer.weight if not envs.MACA_VLLM_USE_TN_2_NN else layer.weight.T

    # we currently do not have quantized bmm's which are needed for
    # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
    # the bmm's in 16-bit, the extra memory overhead of this is fairly low
    kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
    assert kv_b_proj_weight.shape == (
        self.kv_lora_rank,
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}")
    kv_b_proj_weight = kv_b_proj_weight.view(
        self.kv_lora_rank,
        self.num_heads,
        self.qk_nope_head_dim + self.v_head_dim,
    )

    W_UK, W_UV = kv_b_proj_weight.split(
        [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    # Convert from (L, N, V) to (N, L, V)
    self.W_UV = W_UV.transpose(0, 1)
    # Convert from (L, N, P) to (N, P, L)
    self.W_UK_T = W_UK.permute(1, 2, 0)
    

from vllm.v1.attention.backends.mla import common

vllm.v1.attention.backends.mla.common.is_vllm_fa = False
vllm.v1.attention.backends.mla.common.get_flash_attn_version = get_flash_attn_version
vllm.v1.attention.backends.mla.common.MLACommonImpl._flash_attn_varlen_diff_headdims = _flash_attn_varlen_diff_headdims
vllm.v1.attention.backends.mla.common.MLACommonImpl.process_weights_after_loading = process_weights_after_loading
vllm.v1.attention.backends.mla.common.MLACommonImpl.flash_attn_varlen_func = flash_attn_varlen_func

register_patch("vllm.v1.attention.backends.mla.common", "get_flash_attn_version", get_flash_attn_version)
register_patch("vllm.v1.attention.backends.mla.common", "MLACommonImpl._flash_attn_varlen_diff_headdims", _flash_attn_varlen_diff_headdims)
register_patch("vllm.v1.attention.backends.mla.common", "MLACommonImpl.process_weights_after_loading", process_weights_after_loading)
register_patch("vllm.v1.attention.backends.mla.common", "MLACommonImpl.flash_attn_varlen_func", flash_attn_varlen_func)
