"""
vLLM Integration for Solar MoE Model

This module provides a clean and organized implementation of Solar MoE model
for vLLM inference engine. It includes proper type hints, documentation,
and follows Python best practices.

Author: AI Assistant
License: Apache 2.0
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union, ClassVar, Literal
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# Transformers imports
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache

# vLLM imports
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding
)
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    maybe_prefix,
)

# Local imports
from .solar_moe_updated import (
    SolarMoeForCausalLM,
    SolarMoeModel,
    SolarMoeConfig,
    SolarMoePreTrainedModel,
    SOLAR_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC
)

# Environment setup for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


def _convert_hf_config_to_solar(hf_config: Any) -> SolarMoeConfig:
    """
    Convert HuggingFace config to SolarMoeConfig instance.
    
    Args:
        hf_config: HuggingFace configuration object or dict-like object
        
    Returns:
        SolarMoeConfig: Properly configured Solar MoE configuration
        
    Raises:
        ValueError: If config conversion fails
    """
    if isinstance(hf_config, SolarMoeConfig):
        return hf_config
    
    try:
        data = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    except Exception as e:
        raise ValueError(f"Failed to convert config to dict: {e}")
    
    return SolarMoeConfig(**data)


class SolarMoeVllm(nn.Module, SupportsLoRA, SupportsPP):
    """
    vLLM-compatible Solar MoE model implementation.
    
    This class wraps the Solar MoE model to be compatible with vLLM's
    inference engine, providing necessary interfaces and methods for
    efficient inference.
    
    Attributes:
        config: Solar MoE configuration
        model: Core Solar MoE model
        vocab_size: Vocabulary size
        lm_head: Language modeling head
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        head_dim: Dimension of each attention head
        hidden_size: Hidden state dimension
        sliding_window: Sliding window size for attention
        max_model_len: Maximum sequence length
        num_experts: Number of experts in MoE layers
        num_experts_per_tok: Number of experts per token
    """
    
    _tied_weights_keys = ["lm_head.weight"]
    
    # Explicitly declare this as a text generation model for vLLM runner system
    # This helps vLLM 0.11.2+ recognize it supports the 'generate' runner
    # Note: This follows the pattern used by pooling models with is_pooling_model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """
        Initialize Solar MoE vLLM model.

        Args:
            vllm_config: vLLM configuration object
            prefix: Optional prefix for model weights
        """
        super().__init__()

        # Convert and store configuration
        self.config = _convert_hf_config_to_solar(vllm_config.model_config.hf_config)

        # Initialize core model components
        self.model = SolarMoeModel(self.config)
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Use the model's embedding layer (required by VllmModel protocol)
        # Wrap it in VocabParallelEmbedding for vLLM compatibility if needed
        # For now, we'll use the model's embed_tokens directly
        self.embed_tokens = self.model.embed_tokens

        # Set vLLM compatibility attributes
        self._setup_vllm_attributes()

        # KV cache storage (workaround for HF/vLLM cache mismatch)
        self._past_key_values = None
        self._cache_seq_len = 0  # Track total sequence length in cache

    def _setup_vllm_attributes(self) -> None:
        """Setup vLLM compatibility attributes."""
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        
        # Handle key-value heads configuration
        self.num_kv_heads = getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads)
        if self.num_kv_heads is None:
            self.num_kv_heads = self.config.num_attention_heads
            
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.sliding_window = getattr(self.config, 'sliding_window', None)
        self.max_model_len = getattr(self.config, 'max_position_embeddings', 2048)
        
        # MoE specific attributes
        self.num_experts = getattr(self.config, 'num_experts', 1)
        self.num_experts_per_tok = getattr(self.config, 'num_experts_per_tok', 1)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply token embeddings to input_ids.
        
        This method is required by the VllmModel protocol.
        
        Args:
            input_ids: Input token IDs tensor
            
        Returns:
            Embedded token tensor
        """
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """
        Forward pass through the Solar MoE model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len] or [seq_len]
            positions: Position indices [batch_size, seq_len] or [seq_len]
            intermediate_tensors: Optional intermediate tensor storage
            inputs_embeds: Optional pre-computed input embeddings

        Returns:
            Hidden states tensor compatible with vLLM indexing

        Example:
            >>> model = SolarMoeVllm(vllm_config=config)
            >>> input_ids = torch.tensor([[1, 2, 3]])
            >>> positions = torch.tensor([[0, 1, 2]])
            >>> hidden_states = model(input_ids, positions)
        """

        # Ensure proper tensor dimensions for vLLM compatibility
        input_ids, positions = self._prepare_input_tensors(input_ids, positions)

        batch_size, seq_len = input_ids.shape

        # WORKAROUND: Detect new sequence and reset cache
        # If position starts from beginning or has a gap, reset the cache
        first_position = positions[0, 0].item() if positions.dim() == 2 else positions[0].item()

        if first_position == 0 or first_position < self._cache_seq_len:
            # New sequence detected, reset cache
            self._past_key_values = None
            self._cache_seq_len = 0

        # Forward pass through the model with cache
        # QUICK FIX: Set attention_mask=None to let the model create its own proper 4D causal mask
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=None,  # Let model create proper causal mask internally
            position_ids=positions,
            inputs_embeds=inputs_embeds,
            past_key_values=self._past_key_values,  # Pass cached KV
            use_cache=True,  # Enable cache to get updated KV
        )

        # Update cache for next forward pass
        self._past_key_values = outputs.past_key_values
        self._cache_seq_len = self._cache_seq_len + seq_len

        # DIAGNOSTIC: Check cache health (disabled)
        # if self._past_key_values is not None and len(self._past_key_values) > 0:
        #     first_layer_k = self._past_key_values[0][0]  # First layer, K tensor
        #     print(f"[CACHE] seq_len={self._cache_seq_len}, K shape={first_layer_k.shape}, K stats: mean={first_layer_k.mean().item():.4f}, std={first_layer_k.std().item():.4f}")

        # Extract and reshape hidden states for vLLM compatibility
        hidden_states = self._extract_hidden_states(outputs)

        return hidden_states

    def _prepare_input_tensors(
        self, 
        input_ids: torch.Tensor, 
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input tensors for proper vLLM compatibility.
        
        Args:
            input_ids: Input token IDs
            positions: Position indices
            
        Returns:
            Tuple of prepared (input_ids, positions) tensors
        """
        # Ensure proper tensor dimensions
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        
        # Handle multi-sequence inputs
        if input_ids.dim() == 2 and input_ids.shape[0] > 1:
            if positions.shape[0] != input_ids.shape[0]:
                positions = positions.expand(input_ids.shape[0], -1)
        
        return input_ids, positions

    def _extract_hidden_states(self, outputs: Union[tuple, Any]) -> torch.Tensor:
        """
        Extract hidden states from model outputs.

        Args:
            outputs: Model outputs (tuple or ModelOutput object)

        Returns:
            Hidden states tensor reshaped for vLLM compatibility
        """
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            # ModelOutput object
            hidden_states = getattr(outputs, 'last_hidden_state', outputs[0])

        # Reshape for vLLM indexing compatibility
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            # Reshape to [seq_len, hidden_size] for vLLM indexing
            # Note: This assumes batch_size=1 for single request inference
            hidden_states = hidden_states.view(seq_len, hidden_size)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[Any] = None
    ) -> Optional[torch.Tensor]:
        """
        Compute logits from hidden states.

        Args:
            hidden_states: Hidden states tensor [num_tokens, hidden_size]
            sampling_metadata: Optional sampling metadata from vLLM

        Returns:
            Logits tensor [num_tokens, vocab_size]

        Raises:
            ValueError: If hidden_states has unexpected dimensions
        """
        # Compute logits for all hidden states
        logits = self.lm_head(hidden_states)

        # DIAGNOSTIC: Check logits health (disabled)
        # if logits is not None:
        #     print(f"[LOGITS] shape={logits.shape}, stats: min={logits.min().item():.2f}, max={logits.max().item():.2f}, mean={logits.mean().item():.2f}")
        #     if logits.shape[0] > 0:
        #         top_5 = logits[-1].topk(5)
        #         print(f"[LOGITS] Top 5 tokens: {top_5.indices.tolist()}, probs: {torch.softmax(logits[-1], dim=-1)[top_5.indices].tolist()}")

        return logits.float() if logits is not None else None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load model weights using vLLM's AutoWeightsLoader.

        Args:
            weights: Iterable of (name, tensor) weight tuples

        Returns:
            Set of loaded weight names
        """
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> List[Tuple[str, str, int, str]]:
        """
        Get expert mapping for MoE sharding (if needed).
        
        Returns:
            List of expert mapping tuples (empty for now)
        """
        # Not required unless using vLLM MoE sharding hints
        return []
    
    # vLLM interface methods
    def get_sliding_window(self) -> Optional[int]:
        """Get the sliding window size for attention."""
        return self.sliding_window
    
    def get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        return self.num_layers
    
    def get_num_heads(self) -> int:
        """Get the number of attention heads."""
        return self.num_heads
    
    def get_num_kv_heads(self) -> int:
        """Get the number of key-value heads."""
        return self.num_kv_heads
    
    def get_head_dim(self) -> int:
        """Get the dimension of each attention head."""
        return self.head_dim
    
    def get_hidden_size(self) -> int:
        """Get the hidden size of the model."""
        return self.hidden_size
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size
    
    def get_block_size(self) -> int:
        """
        Get the block size for KV cache.
        
        Returns:
            Block size (16) for vLLM compatibility
        """
        return 16  # Standard block size for vLLM
    
    def get_max_model_len(self) -> int:
        """Get the maximum model length."""
        return self.max_model_len