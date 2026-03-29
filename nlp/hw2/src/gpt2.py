"""
GPT-2 Implementation from Scratch

This module implements the GPT-2 transformer architecture using only PyTorch.
No HuggingFace dependencies are allowed in this file.
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GPT2Config:
    """Configuration class for GPT-2 small model."""
    # Total number of tokens in the vocabulary.
    # Note we use the same vocabulary and tokenizer as OpenAI GPT-2.
    vocab_size: int = 50257
    
    # The maximum context window length is 1024 tokens for GPT-2.
    max_ctx_len: int = 1024
    
    # The model dimension (hidden size) for GPT-2 Small is 768.
    d_model: int = 768
    
    # The dimension of each attention head is d_model / n_head = 768 / 12 = 64.
    d_head: int = 64
    
    # The intermediate dimension of the MLP in GPT-2 Small is 4 times the model dimension.
    # 4 * 768 = 3072
    d_mlp_intermediate: int = 3072
    
    # GPT-2 Small has 12 transformer blocks.
    n_layer: int = 12
    
    # GPT-2 Small has 12 attention heads per transformer block.
    n_head: int = 12
    
    # Total number of label classes for our classification dataset.
    num_labels: int = 20


@dataclass
class CausalLMOutput:
    """Output class for causal language modeling. Contains the logits for all input tokens."""
    logits: Tensor


@dataclass
class ModelOutput:
    """Output class for generation. Contains sequences of input and generated token IDs."""
    sequences: Tensor


@dataclass
class SequenceClassifierOutput:
    """Output class for sequence classification. Contains the logits for each label class."""
    logits: Tensor


class GPT2MLPBlock(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_mlp_intermediate)
        self.c_proj = nn.Linear(config.d_mlp_intermediate, config.d_model)

    def forward(self, x: Tensor) -> Tensor:

        mlp_intermediate = F.gelu(self.c_fc(x), approximate="tanh")
        res = self.c_proj(mlp_intermediate)

        return res


class GPT2AttentionBlock(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        # X -> Q, K, V
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)  # single weight: [d_model, 3*d_model] in checkpoint 
        
        # fuse the results of multi heads. Not just concatenate.
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.register_buffer(                                                                                                 
            "bias",                                                                                                           
            torch.ones(config.max_ctx_len, config.max_ctx_len).tril().view(1, 1, config.max_ctx_len, config.max_ctx_len)      
        )  

        self.n_head = config.n_head                                                                                           
        self.d_head = config.d_head   
                                                                                                                        

    def forward(
        self,
        x: Tensor,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, S, D = x.shape

        # Calculate q, k, v
        q, k, v = self.c_attn(x).split(D, dim=-1)  #[B, S, 768]

        # Reshape Q, K, V into multiple head
        q = q.view(B, S, self.n_head, self.d_head).transpose(1, 2)  # [B, 12, S, 64]                                  
        k = k.view(B, S, self.n_head, self.d_head).transpose(1, 2)                                                            
        v = v.view(B, S, self.n_head, self.d_head).transpose(1, 2)   

        # Compute attention scores
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)

        # Apply causal mask
        mask = self.bias[:, :, :S, :S]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over key dimension
        scores = F.softmax(scores, dim=-1)

        # Weighted sum over values
        out = scores @ v  # [B, n_head, S, d_head]

        out = out.transpose(1, 2).contiguous().view(B, S, D)                                                                  
        out = self.c_proj(out) 

        return out, (k, v)


class GPT2TransformerBlock(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attn = GPT2AttentionBlock(config)
        self.mlp = GPT2MLPBlock(config)
        self.ln_1 = nn.LayerNorm(config.d_model)                                                                              
        self.ln_2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: Tensor,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        

        attn, kv = self.attn(self.ln_1(x), past_kv)
        x = x + attn

        mlp = self.mlp(self.ln_2(x))
        x = x+mlp

        return x, kv


class GPT2LMHeadModel(nn.Module):
    """
    GPT-2 Language Model with a language modeling head.
    This corresponds to HF's GPT2LMHeadModel.
    """

    def __init__(self, config: GPT2Config = GPT2Config(), bin_path: Optional[str] = None):
        """
        Initialize GPT-2 Language Model.
        
        Args:
            config: GPT2Config object containing model configurations.
            bin_path: Path to the pytorch_model.bin file. If empty or None, 
                      weights will not be loaded from file.
        """
        super().__init__()

        self.config = config

        # Embedding
        self.wpe = nn.Embedding(config.max_ctx_len, config.d_model)
        self.wte = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer
        blocks = []
        for _ in range(config.n_layer):
            blocks.append(GPT2TransformerBlock(config))
        self.h = nn.ModuleList(blocks)

        # Layer normalization
        self.ln_f = nn.LayerNorm(config.d_model)

        # Load checkpoint weights
        if bin_path:
            state_dict = torch.load(bin_path, weights_only=False)
            for key in state_dict:
                if (state_dict[key].dim() == 2
                        and key.endswith(".weight")
                        and "wte" not in key
                        and "wpe" not in key
                        and "ln" not in key):
                    state_dict[key] = state_dict[key].T
            self.load_state_dict(state_dict)

    def forward(
        self, 
        input_ids: Tensor, 
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ) -> CausalLMOutput:
        """
        Forward pass of GPT-2.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            past_key_values: Optional list of past key-value pairs for KV caching

        Returns:
            CausalLMOutput with logits
        """

        wte = self.wte(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)                                              
        wpe = self.wpe(position_ids)  

        x = wte + wpe
        for i in range(self.config.n_layer):
            x, kv = self.h[i](x, past_key_values[i] if past_key_values else None)   
            if past_key_values is not None:
                past_key_values[i] = kv
        
        x = self.ln_f(x)

        logits = x @ self.wte.weight.T 

        return CausalLMOutput(logits=logits)
        
    def generate(
        self,
        input_ids: Tensor,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 128
    ) -> ModelOutput:
        """
        Generate tokens autoregressively using KV caching.
        
        Args:
            input_ids: [batch_size, seq_len] starting token IDs
            temperature: Sampling temperature. If 0.0, use greedy sampling.
            top_p: Top-p (nucleus) sampling threshold
            max_new_tokens: Maximum number of new tokens to generate
        
        Returns:
            ModelOutput with `sequences` containing the generated token IDs
        """        
        # TODO: implement the generation method here. 
        # You should use the `forward` method to compute logits and update KV cache at each step.
        # You can assume the input sequences are always padded to the same length,
        # and the total sequence length (input + generated) will not exceed 512 tokens.
        # GPT-2 does not have a stop token,
        # so you should always generate `max_new_tokens` new tokens 
        # for all the input sequences in the batch.
        
        return ModelOutput(sequences=input_ids)


class GPT2ForSequenceClassification(nn.Module):
    """
    GPT-2 Model with a classification head.
    """

    def __init__(self, 
                 config: GPT2Config = GPT2Config(), 
                 classifier_bin_path: Optional[str] = None,
                 lm_bin_path: Optional[str] = None):
        """
        Initialize GPT-2 Classification Model.
        
        Args:
            config: GPT2Config object containing model configurations,
                    including the number of labels.
            classifier_bin_path: Path to the 
                    This file should contain the weights for 
                    both the GPT-2 base model and the classification head.
                    If empty or None,
                    the classification head weights will be initialized randomly, 
                    and the base model weights may be initialized randomly 
                    or loaded from `lm_bin_path` if provided.
            lm_bin_path: Path to the pytorch_model.bin file for the language model.
                    This file should contain the weights for the GPT-2 base model.
                    If empty or None,
                    weights may be initialized randomly, 
                    or loaded from `classifier_bin_path` if provided.
        """
        super().__init__()

        # Only one of `classifier_bin_path` and `lm_bin_path` can be provided.
        assert not (classifier_bin_path and lm_bin_path), \
            "Only one of `classifier_bin_path` and `lm_bin_path` can be provided."

        # TODO: define and initialize the GPT-2 model that can be used for sequence classification.
        # You can reuse the GPT2LMHeadModel defined above as the base model,
        # and add a classification head on top of it.
        # You should also reuse GPT2LMHeadModel's weights to speed up training if possible.

    def forward(self, input_ids: Tensor) -> SequenceClassifierOutput:
        """
        Forward pass of GPT-2 for classification.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
        
        Returns:
            SequenceClassifierOutput with logits of shape (batch_size, num_labels)
        """
        
        # TODO: implement the forward pass for sequence classification here.
        # The output logits should be of shape (batch_size, num_labels),
        # where num_labels is specified in the GPT2Config,
        # and the logits contain the classification scores for each label class.
        
        return SequenceClassifierOutput(logits=logits)
