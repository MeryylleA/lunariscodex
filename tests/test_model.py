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

# Common test parameters used across multiple tests
BATCH_SIZE = 2
SEQ_LEN = 16 # Using a shorter sequence length for faster unit tests
D_MODEL = 64
N_HEADS = 4
VOCAB_SIZE = 100 # A small vocab size for testing purposes

@pytest.fixture
def base_config() -> LunarisCodexConfig:
    """
    Provides a basic, small LunarisCodexConfig for use in tests.
    Dropout is disabled for more deterministic behavior in tests.
    FlashAttention is disabled to ensure PyTorch fallback is tested on CPU CI runners.
    """
    return LunarisCodexConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=2, # Few layers for faster full model tests
        n_heads=N_HEADS,
        max_seq_len=SEQ_LEN * 2, # Ensure model's ALiBi context can handle test seq_len
        dropout=0.0, # Disable dropout for deterministic tests
        lora_rank=0, # Default to no LoRA unless a test specifically enables it
        use_flash_attention_if_available=False, # Force PyTorch attention path
        ff_multiplier=2, # Smaller FFN intermediate dim for speed
        activation="swiglu" # Default activation
    )

@pytest.fixture
def dummy_input_tensor(base_config: LunarisCodexConfig) -> torch.Tensor:
    """Provides a dummy input tensor of shape (BATCH_SIZE, SEQ_LEN, D_MODEL)."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, base_config.d_model)

@pytest.fixture
def dummy_input_ids(base_config: LunarisCodexConfig) -> torch.Tensor:
    """Provides dummy input token IDs of shape (BATCH_SIZE, SEQ_LEN)."""
    return torch.randint(0, base_config.vocab_size, (BATCH_SIZE, SEQ_LEN))

# --- Tests for LoRALinear ---
def test_lora_linear_initialization_and_shape(dummy_input_tensor: torch.Tensor):
    """
    Tests basic initialization of LoRALinear (with and without LoRA)
    and ensures the output tensor shape is correct.
    """
    in_features = D_MODEL
    out_features = D_MODEL * 2 # Example: FFN expansion
    rank = 4 # A small rank for LoRA testing
    x = dummy_input_tensor

    # Test Scenario 1: LoRA enabled (rank > 0)
    lora_layer_enabled = LoRALinear(in_features, out_features, rank=rank, bias=False)
    assert lora_layer_enabled.has_lora, "LoRA should be flagged as enabled when rank > 0"
    assert lora_layer_enabled.lora_A.shape == (in_features, rank), "LoRA_A matrix shape mismatch"
    assert lora_layer_enabled.lora_B.shape == (rank, out_features), "LoRA_B matrix shape mismatch"
    
    output_enabled = lora_layer_enabled(x)
    assert output_enabled.shape == (BATCH_SIZE, SEQ_LEN, out_features), \
        "Output shape mismatch when LoRA is enabled"

    # Test Scenario 2: LoRA disabled (rank = 0)
    lora_layer_disabled_rank_zero = LoRALinear(in_features, out_features, rank=0, bias=False)
    assert not lora_layer_disabled_rank_zero.has_lora, "LoRA should be flagged as disabled when rank = 0"
    
    output_disabled_rank_zero = lora_layer_disabled_rank_zero(x)
    assert output_disabled_rank_zero.shape == (BATCH_SIZE, SEQ_LEN, out_features), \
        "Output shape mismatch when LoRA rank is 0 (disabled)"

    # Test Scenario 3: LoRA disabled (rank = None)
    lora_layer_disabled_rank_none = LoRALinear(in_features, out_features, rank=None, bias=False)
    assert not lora_layer_disabled_rank_none.has_lora, "LoRA should be flagged as disabled when rank is None"

    output_disabled_rank_none = lora_layer_disabled_rank_none(x)
    assert output_disabled_rank_none.shape == (BATCH_SIZE, SEQ_LEN, out_features), \
        "Output shape mismatch when LoRA rank is None (disabled)"

# --- Tests for FeedForward ---
@pytest.mark.parametrize("activation_fn_name", ["swiglu", "gelu"])
def test_feed_forward_shape_and_activations(base_config: LunarisCodexConfig, dummy_input_tensor: torch.Tensor, activation_fn_name: str):
    """
    Tests the FeedForward layer with different activation functions (SwiGLU, GELU)
    and ensures the output tensor shape is correct.
    """
    config = base_config
    config.activation = activation_fn_name # Set activation from parametrize
    
    ffn_intermediate_dim = config.d_model * config.ff_multiplier
    ff_layer = FeedForward(
        d_model=config.d_model, 
        d_ff=ffn_intermediate_dim, 
        dropout=config.dropout, 
        activation=config.activation, 
        lora_rank=config.lora_rank
    )
    
    output = ff_layer(dummy_input_tensor)
    assert output.shape == dummy_input_tensor.shape, \
        f"Output shape mismatch for FeedForward layer with {activation_fn_name} activation"

# --- Tests for SelfAttention ---
def test_self_attention_manual_fallback_shapes_with_masks(base_config: LunarisCodexConfig, dummy_input_tensor: torch.Tensor):
    """
    Tests the SelfAttention layer's manual PyTorch fallback mechanism,
    ensuring correct output shapes when ALiBi/causal bias and padding masks are applied.
    This test forces the non-FlashAttention path.
    """
    config = base_config
    config.use_flash_attention_if_available = False # Crucial to test the manual path

    attention_layer = SelfAttention(config)
    
    batch_size, seq_len, d_model = dummy_input_tensor.shape
    device = dummy_input_tensor.device

    # Create a simplified ALiBi bias that includes causality for testing SelfAttention in isolation
    # This replicates the core idea of get_alibi_attention_bias from the model for unit testing.
    slopes = torch.tensor([2**(-(8.0/config.n_heads)*(i+1)) for i in range(config.n_heads)], device=device, dtype=torch.float)
    pos_indices = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(0) - \
                  torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(1)
    alibi_bias_values = slopes.view(-1, 1, 1) * pos_indices.unsqueeze(0) # Shape: (H, L, L)
    
    causal_mask_bool = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    # Initialize additive_causal_mask correctly to match model's get_alibi_attention_bias
    additive_causal_mask = torch.zeros((seq_len, seq_len), dtype=alibi_bias_values.dtype, device=device)
    additive_causal_mask.masked_fill_(causal_mask_bool, float('-inf'))
    
    alibi_combined_bias_for_test = alibi_bias_values + additive_causal_mask.unsqueeze(0) # Shape: (H, L, L)

    # Scenario 1: ALiBi/Causal bias only, no additional padding mask
    output_causal_only = attention_layer(dummy_input_tensor, alibi_combined_bias_for_test, padding_additive_mask=None)
    assert output_causal_only.shape == dummy_input_tensor.shape, \
        "Output shape mismatch for SelfAttention with ALiBi/causal bias only"

    # Scenario 2: ALiBi/Causal bias AND a padding mask
    # Dummy padding mask: first batch item no padding, second item pads the second half of the sequence
    padding_additive_mask = torch.zeros(batch_size, 1, 1, seq_len, device=device, dtype=dummy_input_tensor.dtype)
    if batch_size > 1:
        padding_start_idx = seq_len // 2
        padding_additive_mask[1, :, :, padding_start_idx:] = float('-inf') # Pad the second item in batch
    
    output_with_padding = attention_layer(dummy_input_tensor, alibi_combined_bias_for_test, padding_additive_mask)
    assert output_with_padding.shape == dummy_input_tensor.shape, \
        "Output shape mismatch for SelfAttention with ALiBi/causal and padding masks"

    # Basic check: output for a fully unpadded item should differ from an item with substantial padding
    # if batch_size > 1 and padding_start_idx < seq_len:
    #     assert not torch.allclose(output_with_padding[0], output_with_padding[1]), \
    #         "Outputs for unpadded and padded items in batch should differ if padding is effective."
    # This assertion is tricky without fixed weights; focusing on shape and non-crash is primary for now.

# --- Tests for TransformerDecoderBlock ---
def test_transformer_decoder_block_shape(base_config: LunarisCodexConfig, dummy_input_tensor: torch.Tensor):
    """Tests the forward pass of a TransformerDecoderBlock and its output shape."""
    config = base_config
    decoder_block = TransformerDecoderBlock(config)
    
    seq_len = dummy_input_tensor.shape[1]
    device = dummy_input_tensor.device

    # Re-create a simplified ALiBi/causal bias for this block test
    slopes = torch.tensor([2**(-(8.0/config.n_heads)*(i+1)) for i in range(config.n_heads)], device=device, dtype=torch.float)
    pos_indices = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(0) - \
                  torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(1)
    alibi_bias_values = slopes.view(-1, 1, 1) * pos_indices.unsqueeze(0)
    causal_mask_bool = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    additive_causal_mask = torch.zeros((seq_len, seq_len), dtype=alibi_bias_values.dtype, device=device)
    additive_causal_mask.masked_fill_(causal_mask_bool, float('-inf'))
    alibi_combined_bias_for_test = alibi_bias_values + additive_causal_mask.unsqueeze(0)

    # Test with ALiBi/causal bias and no additional padding
    output = decoder_block(dummy_input_tensor, alibi_combined_bias_for_test, padding_additive_mask=None)
    assert output.shape == dummy_input_tensor.shape, "Output shape mismatch for TransformerDecoderBlock"

# --- Tests for LunarisMind (Full Model) ---
def test_lunaris_mind_forward_pass_shape(base_config: LunarisCodexConfig, dummy_input_ids: torch.Tensor):
    """Tests the forward pass of the full LunarisMind model and output logits shape."""
    config = base_config
    # Ensure vocab_size is correctly set from the fixture for model instantiation
    assert config.vocab_size == VOCAB_SIZE, "Fixture vocab_size not matching constant"
    
    model = LunarisMind(config)
    model.eval() # Set to eval mode for testing forward pass without dropout effects if any were active
    
    # Create a dummy attention mask (all tokens are considered valid)
    attention_mask = torch.ones_like(dummy_input_ids)

    with torch.no_grad(): # Inference-like pass
        logits = model(dummy_input_ids, attention_mask=attention_mask)
    
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size), \
        f"Logits shape mismatch. Expected {(BATCH_SIZE, SEQ_LEN, config.vocab_size)}, got {logits.shape}"

def test_lunaris_mind_generate_deterministic_greedy(base_config: LunarisCodexConfig, dummy_input_ids: torch.Tensor):
    """
    Tests the model.generate() method with greedy decoding (approximated by low temperature and top_k=1)
    for basic execution and output shape correctness.
    """
    config = base_config
    assert config.vocab_size == VOCAB_SIZE
    config.dropout = 0.0 # Ensure no dropout during this generation test if not already 0

    model = LunarisMind(config)
    model.eval() 

    # Use a subset of dummy_input_ids as the prompt
    prompt_len = SEQ_LEN // 2
    prompt_ids = dummy_input_ids[:, :prompt_len] 
    
    max_new_tokens = 5

    torch.manual_seed(42) # Set seed for reproducibility of torch.multinomial
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    generated_ids = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.001, # Very low temperature to approximate greedy
        top_k=1,           # Explicitly greedy
        top_p=1.0,         # Disable top_p by setting to 1.0
        repetition_penalty=1.0, # No repetition penalty
        eos_token_id=None  # Do not stop on EOS for this shape test
    )

    expected_len = prompt_ids.shape[1] + max_new_tokens
    assert generated_ids.shape == (BATCH_SIZE, expected_len), \
        f"Generated IDs shape mismatch. Expected {(BATCH_SIZE, expected_len)}, got {generated_ids.shape}"
    
    # Check if the original prompt tokens are preserved at the beginning of the output
    assert torch.all(generated_ids[:, :prompt_ids.shape[1]] == prompt_ids), \
        "Original prompt IDs were not preserved in the generated sequence"

# TODO: Add more detailed tests in the future:
# - LoRALinear: Check if LoRA adaptation actually changes output values compared to base layer.
# - SelfAttention: Test actual attention score distributions with known simple inputs/masks.
# - ALiBi: More direct verification of the bias values generated by model.get_alibi_attention_bias.
# - LunarisMind.generate(): Test with actual EOS token, different sampling parameters, and repetition penalty.
# - Behavior with different padding scenarios in various components.
# - Test weight initialization if specific statistical properties of weights are critical.
