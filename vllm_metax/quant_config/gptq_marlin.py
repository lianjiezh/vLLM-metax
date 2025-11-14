# SPDX-License-Identifier: Apache-2.0

from typing import Optional, TYPE_CHECKING

from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig

from vllm_metax.patch.model_executor.hook_register import register_quantization_config

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods


@register_quantization_config("gptq_marlin")
class MacaGPTQMarlinConfig(GPTQMarlinConfig):
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> Optional["QuantizationMethods"]:
        return None
