# SPDX-License-Identifier: Apache-2.0

from vllm import ModelRegistry

def register_model():
    from .baichuan_moe import BaiChuanMoEForCausalLM
    from .telechat import TelechatForCausalLM
    from .deepseek import DeepseekForCausalLM
    from .deepseek_v2 import DeepseekV2ForCausalLM
    from .qwen import QWenLMHeadModel
    from .qwen3 import Qwen3ForCausalLM
    from .qwen3_moe import Qwen3MoeForCausalLM

    ModelRegistry.register_model(
        "baichuan_moe",
        "vllm_metax_plugin.models.baichuan_moe:BaiChuanMoEForCausalLM")

    ModelRegistry.register_model(
        "telechat",
        "vllm_metax_plugin.models.telechat:TelechatForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "vllm_metax_plugin.models.deepseek_v2:DeepseekV2ForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "vllm_metax_plugin.models.deepseek_v2:DeepseekV3ForCausalLM")

    ModelRegistry.register_model(
        "qwen",
        "vllm_metax_plugin.models.baichuan_moe:QWenLMHeadModel")

    ModelRegistry.register_model(
        "qwen3",
        "vllm_metax_plugin.models.qwen3:Qwen3ForCausalLM")
    
    ModelRegistry.register_model(
        "qwen3_moe",
        "vllm_metax_plugin.models.qwen3_moe:Qwen3MoeForCausalLM")
