# Supported Models

You could refer to [vllm's docs](https://docs.vllm.ai/en/stable/models/supported_models.html) for more details.

Here the plugin would list all the **tested** model on Maca.

## Feature Status Legend

- ✅︎ indicates that the feature is supported for the model.

- 🚧 indicates that the feature is planned but not yet supported for the model.

- ⚠️ indicates that the feature is available but may have known issues or limitations.

## List of Text-only Language Models

### Text Generative Models

| Architecture | Models | Example HF Models | [LoRA](https://docs.vllm.ai/en/stable/features/lora.html) | [PP](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `BaiChuanForCausalLM` | Baichuan2, Baichuan | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc. | ✅︎ | ✅︎ |
| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM | `zai-org/chatglm2-6b`, `zai-org/chatglm3-6b`, etc. | ✅︎ | ✅︎ |
| `DeepseekForCausalLM` | DeepSeek | `deepseek-ai/deepseek-llm-7b-chat`, etc. | ✅︎ | ✅︎ |
| `DeepseekV2ForCausalLM` | DeepSeek-V2 | `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat`, etc. | ✅︎ | ✅︎ |
| `DeepseekV3ForCausalLM` | DeepSeek-V3 | `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`, etc. | ✅︎ | ✅︎ |
| `Ernie4_5_MoeForCausalLM` | Ernie4.5MoE | `baidu/ERNIE-4.5-21B-A3B-PT`, etc. |✅︎| ✅︎ |
| `GlmForCausalLM` | GLM-4 | `zai-org/glm-4-9b-chat-hf`, etc. | ✅︎ | ✅︎ |
| `Glm4ForCausalLM` | GLM-4-0414 | `zai-org/GLM-4-32B-0414`, etc. | ✅︎ | ✅︎ |
| `Glm4MoeForCausalLM` | GLM-4.5, GLM-4.6 | `zai-org/GLM-4.5`, etc. | ✅︎ | ✅︎ |
| `InternLM3ForCausalLM` | InternLM3 | `internlm/internlm3-8b-instruct`, etc. | ✅︎ | ✅︎ |
| `LlamaForCausalLM` | Llama 3.1, Llama 3, Llama 2, LLaMA, Yi | `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, etc. | ✅︎ | ✅︎ |
| `MistralForCausalLM` | Mistral, Mistral-Instruct | `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc. | ✅︎ | ✅︎ |
| `MixtralForCausalLM` | Mixtral-8x7B, Mixtral-8x7B-Instruct | `mistralai/Mixtral-8x7B-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc. | ✅︎ | ✅︎ |
| `QWenLMHeadModel` | Qwen | `Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForCausalLM` | QwQ, Qwen2 | `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B`, etc. | ✅︎ | ✅︎ |
| `Qwen3ForCausalLM` | Qwen3 | `Qwen/Qwen3-8B`, etc. | ✅︎ | ✅︎ |
| `Qwen3MoeForCausalLM` | Qwen3MoE | `Qwen/Qwen3-30B-A3B`, etc. | ✅︎ | ✅︎ |
| `Qwen3NextForCausalLM` | Qwen3NextMoE | `Qwen/Qwen3-Next-80B-A3B-Instruct`, etc. | ✅︎ | ✅︎ |

## List of Multimodal Language Models

The following modalities are supported depending on the model:

- **T**ext
- **I**mage
- **V**ideo
- **A**udio

Any combination of modalities joined by `+` are supported.

- e.g.: `T + I` means that the model supports text-only, image-only, and text-with-image inputs.

On the other hand, modalities separated by `/` are mutually exclusive.

- e.g.: `T / I` means that the model supports text-only and image-only inputs, but not text-with-image inputs.

See [this page](https://docs.vllm.ai/en/stable/features/multimodal_inputs.html) on how to pass multi-modal inputs to the model.


### Text Generative Models

| Architecture | Models | Inputs | Example HF Models | [LoRA](https://docs.vllm.ai/en/stable/features/lora.html) | [PP](https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html) |
|--------------|--------|--------|-------------------|----------------------|---------------------------|
| `DeepseekVLV2ForCausalLM`<sup>^</sup> | DeepSeek-VL2 | T + I<sup>+</sup> | `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2`, etc. | | ✅︎ |
| `InternVLChatModel` | InternVL 3.5, InternVL 3.0, InternVideo 2.5, InternVL 2.5, Mono-InternVL, InternVL 2.0 | T + I<sup>E+</sup> + (V<sup>E+</sup>) | `OpenGVLab/InternVL3_5-14B`, `OpenGVLab/InternVL3-9B`, `OpenGVLab/InternVideo2_5_Chat_8B`, `OpenGVLab/InternVL2_5-4B`, `OpenGVLab/Mono-InternVL-2B`, `OpenGVLab/InternVL2-4B`, etc. | ✅︎ | ✅︎ |
| `QwenVLForConditionalGeneration`<sup>^</sup> | Qwen-VL | T + I<sup>E+</sup> | `Qwen/Qwen-VL`, `Qwen/Qwen-VL-Chat`, etc. | ✅︎ | ✅︎ |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen3VLMoeForConditionalGeneration` | Qwen3-VL-MOE | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-30B-A3B-Instruct`, etc. | ⚠️ | ✅︎ |