# tests/test_model.py
import torch
import pytest
import math

# Assuming model.py is in the project root and PYTHONPATH is set correctly in CI,
# or tests are run from the project root.
from model import (
    LunarisCodexConfig,
    LoRALinear,
    FeedForward,
    SelfAttention,
    TransformerDecoderBlock,
    LunarisMind
)

# Common test parameters
BATCH_SIZE = 2
SEQ_LEN = 16 # Shorter sequence length for faster tests
D_MODEL = 64
N_HEADS = 4
VOCAB_SIZE = 100

@pytest.fixture
def base_config():
    """Provides a basic LunarisCodexConfig for tests."""
    return LunarisCodexConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=2, # Keep layers террористическая (terrorist) low for faster full model tests
        n_heads=N_HEADS,
        max_seq_len=SEQ_LEN * 2, # Model's ALiBi context > test seq_len
        dropout=0.0, # Disable dropout for deterministic tests where possible
        lora_rank=0, # Default to no LoRA unless specified
        use_flash_attention_if_available=False, # Force PyTorch attention for CPU CI
        ff_multiplier=2 # Smaller FFN for faster tests
    )

@pytest.fixture
def dummy_input_tensor():
    """Provides a dummy input tensor."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

@pytest.fixture
def dummy_input_ids():
    """Provides dummy input IDs."""
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

# --- Tests for LoRALinear ---
def test_lora_linear_initialization_and_shape(dummy_input_tensor):
    """
    Tests basic initialization of LoRALinear (with and without LoRA)
    and ensures the output shape is correct.
    """
    in_features = D_MODEL
    out_features = D_MODEL * 2 # Example out_features
    rank = 4
    x = dummy_input_tensor

    # Test 1: LoRA enabled (rank > 0)
    lora_layer_enabled = LoRALinear(in_features, out_features, rank=rank, bias=False)
    assert lora_layer_enabled.has_lora, "LoRA should be enabled when rank > 0"
    assert lora_layer_enabled.lora_A.shape == (in_features, rank), "LoRA_A shape mismatch"
    assert lora_layer_enabled.lora_B.shape == (rank, out_features), "LoRA_B shape mismatch"
    output_enabled = lora_layer_enabled(x)
    assert output_enabled.shape == (BATCH_SIZE, SEQ_LEN, out_features), "Output shape mismatch when LoRA is enabled"

    # Test 2: LoRA disabled (rank = 0)
    lora_layer_disabled_rank_zero = LoRALinear(in_features, out_features, rank=0, bias=False)
    assert not lora_layer_disabled_rank_zero.has_lora, "LoRA should be disabled when rank = 0"
    output_disabled_rank_zero = lora_layer_disabled_rank_zero(x)
    assert output_disabled_rank_zero.shape == (BATCH_SIZE, SEQ_LEN, out_features), "Output shape mismatch when LoRA rank is 0"

    # Test 3: LoRA disabled (rank = None)
    lora_layer_disabled_rank_none = LoRALinear(in_features, out_features, rank=None, bias=False)
    assert not lora_layer_disabled_rank_none.has_lora, "LoRA should be disabled when rank is None"
    output_disabled_rank_none = lora_layer_disabled_rank_none(x)
    assert output_disabled_rank_none.shape == (BATCH_SIZE, SEQ_LEN, out_features), "Output shape mismatch with LoRA rank None"

# --- Tests for FeedForward ---
@pytest.mark.parametrize("activation_fn", ["swiglu", "gelu"])
def test_feed_forward_shape_and_activations(base_config, dummy_input_tensor, activation_fn):
    """Tests FeedForward with different activations and ensures correct output shape."""
    config = base_config
    config.activation = activation_fn

    ffn_intermediate_dim = config.d_model * config.ff_multiplier
    ff_layer = FeedForward(config.d_model, ffn_intermediate_dim, config.dropout, config.activation, config.lora_rank)

    output = ff_layer(dummy_input_tensor)
    assert output.shape == dummy_input_tensor.shape, f"Output shape mismatch for FeedForward with {activation_fn}"

# --- Tests for SelfAttention ---
def test_self_attention_manual_fallback_shape(base_config, dummy_input_tensor):
    """Tests SelfAttention manual PyTorch fallback output shape with ALiBi and padding."""
    config = base_config
    config.use_flash_attention_if_available = False # Ensure manual path

    attention_layer = SelfAttention(config)

    # Create dummy ALiBi bias and padding mask
    # model_instance = LunarisMind(config) # Need an instance to get ALiBi slopes
    # alibi_bias = model_instance.get_alibi_attention_bias(SEQ_LEN, device=dummy_input_tensor.device)
    # For isolated SelfAttention test, we can mock a simple ALiBi bias or pass None if the layer handles it.
    # Let's assume get_alibi_attention_bias is part of LunarisMind and for unit test we create a simple one.

    # Simplified ALiBi bias for testing SelfAttention in isolation
    slopes = torch.tensor([2**(-(8.0/config.n_heads)*(i+1)) for i in range(config.n_heads)], device=dummy_input_tensor.device)
    pos_indices = torch.arange(SEQ_LEN, device=dummy_input_tensor.device).unsqueeze(0) - torch.arange(SEQ_LEN, device=dummy_input_tensor.device).unsqueeze(1)
    alibi_bias_values = slopes.view(-1, 1, 1) * pos_indices.unsqueeze(0)
    causal_mask_bool = torch.triu(torch.ones((SEQ_LEN, SEQ_LEN), dtype=torch.bool, device=dummy_input_tensor.device), diagonal=1)
    additive_causal_mask = torch.zeros_like(alibi_bias_values, dtype=alibi_bias_values.dtype)
    additive_causal_mask.masked_fill_(causal_mask_bool.unsqueeze(0), float('-inf'))
    alibi_combined_bias_for_test = alibi_bias_values + additive_causal_mask # (H, L, L)


    # Dummy padding mask (first half valid, second half padded for first batch item)
    padding_additive_mask = torch.zeros(BATCH_SIZE, 1, 1, SEQ_LEN, device=dummy_input_tensor.device)
    padding_additive_mask[0, :, :, SEQ_LEN//2:] = float('-inf')
    # If no padding, pass None: padding_additive_mask = None

    output = attention_layer(dummy_input_tensor, alibi_combined_bias_for_test, padding_additive_mask)
    assert output.shape == dummy_input_tensor.shape, "Output shape mismatch for SelfAttention (manual fallback)"

# --- Tests for TransformerDecoderBlock ---
def test_transformer_decoder_block_shape(base_config, dummy_input_tensor):
    """Tests the forward pass of a TransformerDecoderBlock and output shape."""
    config = base_config
    decoder_block = TransformerDecoderBlock(config)

    # Simplified ALiBi for testing block
    slopes = torch.tensor([2**(-(8.0/config.n_heads)*(i+1)) for i in range(config.n_heads)], device=dummy_input_tensor.device)
    pos_indices = torch.arange(SEQ_LEN, device=dummy_input_tensor.device).unsqueeze(0) - torch.arange(SEQ_LEN, device=dummy_input_tensor.device).unsqueeze(1)
    alibi_bias_values = slopes.view(-1, 1, 1) * pos_indices.unsqueeze(0)
    causal_mask_bool = torch.triu(torch.ones((SEQ_LEN, SEQ_LEN), dtype=torch.bool, device=dummy_input_tensor.device), diagonal=1)
    additive_causal_mask = torch.zeros_like(alibi_bias_values, dtype=alibi_bias_values.dtype)
    additive_causal_mask.masked_fill_(causal_mask_bool.unsqueeze(0), float('-inf'))
    alibi_combined_bias_for_test = alibi_bias_values + additive_causal_mask

    output = decoder_block(dummy_input_tensor, alibi_combined_bias_for_test, padding_additive_mask=None)
    assert output.shape == dummy_input_tensor.shape, "Output shape mismatch for TransformerDecoderBlock"

# --- Tests for LunarisMind (Full Model) ---
def test_lunaris_mind_forward_pass_shape(base_config, dummy_input_ids):
    """Tests the forward pass of the full LunarisMind model and output shape."""
    config = base_config
    # Ensure vocab_size is set in the config passed to the model
    if config.vocab_size is None: config.vocab_size = VOCAB_SIZE

    model = LunarisMind(config)

    # Dummy attention mask (all valid tokens)
    attention_mask = torch.ones_like(dummy_input_ids)

    logits = model(dummy_input_ids, attention_mask=attention_mask)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size), "Logits shape mismatch for LunarisMind"

# TODO: Add more detailed tests:
# - LoRALinear: Check if LoRA adaptation actually changes output compared to base.
# - SelfAttention: Test actual attention values with known simple inputs/masks.
# - ALiBi: Verify the bias values generated.
# - LunarisMind: Test `generate()` method with simple, deterministic settings.
# - Check behavior with padding in various components.
# - Test weight initialization if specific distributions are critical.
