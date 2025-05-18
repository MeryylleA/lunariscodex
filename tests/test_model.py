# tests/test_model.py
import torch
import pytest # pytest will be installed in the CI environment via requirements or direct pip install

# Assuming model.py is in the project root and PYTHONPATH is set correctly in CI,
# or tests are run from the project root.
from model import LunarisCodexConfig, LoRALinear 
# If you add tests for SelfAttention, etc., import them here:
# from model import SelfAttention, FeedForward, TransformerDecoderBlock, LunarisMind

def test_lora_linear_initialization_and_shape():
    """
    Tests basic initialization of LoRALinear (with and without LoRA) 
    and ensures the output shape is correct.
    """
    in_features = 128
    out_features = 256
    rank = 8
    dummy_input = torch.randn(2, 10, in_features) # Batch, SeqLen, InFeatures

    # Test 1: LoRA enabled (rank > 0)
    lora_layer_enabled = LoRALinear(in_features, out_features, rank=rank, bias=False)
    assert lora_layer_enabled.has_lora, "LoRA should be enabled when rank > 0"
    assert lora_layer_enabled.lora_A.shape == (in_features, rank), "LoRA_A shape mismatch"
    assert lora_layer_enabled.lora_B.shape == (rank, out_features), "LoRA_B shape mismatch"
    
    output_enabled = lora_layer_enabled(dummy_input)
    assert output_enabled.shape == (2, 10, out_features), "Output shape mismatch when LoRA is enabled"

    # Test 2: LoRA disabled (rank = 0)
    lora_layer_disabled_rank_zero = LoRALinear(in_features, out_features, rank=0, bias=False)
    assert not lora_layer_disabled_rank_zero.has_lora, "LoRA should be disabled when rank = 0"
    
    output_disabled_rank_zero = lora_layer_disabled_rank_zero(dummy_input)
    assert output_disabled_rank_zero.shape == (2, 10, out_features), "Output shape mismatch when LoRA rank is 0"

    # Test 3: LoRA disabled (rank = None)
    lora_layer_disabled_rank_none = LoRALinear(in_features, out_features, rank=None, bias=False)
    assert not lora_layer_disabled_rank_none.has_lora, "LoRA should be disabled when rank is None"

    output_disabled_rank_none = lora_layer_disabled_rank_none(dummy_input)
    assert output_disabled_rank_none.shape == (2, 10, out_features), "Output shape mismatch when LoRA rank is None"

# TODO: Add more tests for other components in model.py
# For example:
# def test_self_attention_initialization():
#     pass
#
# def test_feed_forward_activations():
#     pass
#
# def test_transformer_decoder_block_forward():
#     pass
#
# def test_lunaris_mind_full_model_forward():
#     pass
#
# def test_lunaris_mind_generate_simple():
#     pass
