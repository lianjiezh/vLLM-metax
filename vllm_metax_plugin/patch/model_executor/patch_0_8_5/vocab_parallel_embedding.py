# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax_plugin.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

import torch
import inspect
from typing import Tuple
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, UnquantizedEmbeddingMethod

def get_masked_input_and_mask(
        input_: torch.Tensor, org_vocab_start_index: int,
        org_vocab_end_index: int, num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # torch.jit.script will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (
        input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index)
    added_offset = added_vocab_start_index - (
        org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
    valid_offset = (org_vocab_start_index *
                    org_vocab_mask) + (added_offset * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask

class MetaxVocabParallelEmbedding(VocabParallelEmbedding):
    def __init__(self, *args, **kwargs):

        # 1) 拿到父类 __init__ 的签名
        sig = inspect.signature(VocabParallelEmbedding.__init__)
        # bind_partial 可以让我们先把 args/kwargs 对应到形参名上
        bound = sig.bind_partial(self, *args, **kwargs)
        # 填入签名里的默认值
        bound.apply_defaults()
        
        params_dtype = bound.arguments['params_dtype']

        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            self.quant_method.create_weights(self,
                                            self.embedding_dim,
                                            [self.num_embeddings_per_partition],
                                            self.embedding_dim,
                                            self.num_embeddings_padded,
                                            params_dtype=params_dtype,
                                            weight_loader=self.weight_loader)
        else:
            self.quant_method.create_weights(self,
                                          self.num_embeddings_per_partition,
                                          [self.embedding_dim],
                                          self.embedding_dim,
                                          self.num_embeddings_padded,
                                          params_dtype=params_dtype,
                                          weight_loader=self.weight_loader) 

vllm.model_executor.layers.vocab_parallel_embedding.get_masked_input_and_mask = torch.jit.script(get_masked_input_and_mask)
vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding = MetaxVocabParallelEmbedding
register_patch("vllm.model_executor.layers.vocab_parallel_embedding", "get_masked_input_and_mask", get_masked_input_and_mask)
register_patch("vllm.model_executor.layers.vocab_parallel_embedding", "VocabParallelEmbedding", MetaxVocabParallelEmbedding)

