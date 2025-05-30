import torch
import pytest
import math
import copy
import os
import sys
from typing import Optional, Tuple # Added for type hinting

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model import (
    LunarisCodexConfig,
    LoRALinear,
    FeedForward,
    SelfAttention,
    TransformerDecoderBlock,
    LunarisMind,
    count_parameters
)

# Define constants for tests to make them easier to manage
BATCH_SIZE = 2
SEQ_LEN = 16 # Default sequence length for dummy tensors
D_MODEL_SMALL_FOR_BASE = 32
N_HEADS_SMALL_FOR_BASE = 2
VOCAB_SIZE_SMALL = 50

@pytest.fixture(scope="module")
def base_config() -> LunarisCodexConfig:
    """Base configuration for most model tests."""
    return LunarisCodexConfig(
        vocab_size=VOCAB_SIZE_SMALL,
        d_model=D_MODEL_SMALL_FOR_BASE,
        n_layers=2,
        n_heads=N_HEADS_SMALL_FOR_BASE,
        max_seq_len=SEQ_LEN * 2, 
        dropout=0.0, 
        lora_rank=0, 
        use_flash_attention_if_available=False, 
        ff_multiplier=4,
        activation="swiglu",
        pad_token_id=0 
    )

@pytest.fixture
def config_with_lora(base_config: LunarisCodexConfig) -> LunarisCodexConfig:
    """Configuration with LoRA enabled."""
    config = copy.deepcopy(base_config)
    config.lora_rank = 4
    return config

@pytest.fixture
def config_for_layerscale_test() -> LunarisCodexConfig:
    """Configuration designed to activate LayerScale."""
    return LunarisCodexConfig(
        vocab_size=50000, 
        d_model=512,      
        n_layers=6,      
        n_heads=8, 
        max_seq_len=SEQ_LEN,
        dropout=0.0,
        lora_rank=0,
        ff_multiplier=4, 
        activation="swiglu",
        pad_token_id=0
    )

@pytest.fixture
def config_no_layerscale_test(base_config: LunarisCodexConfig) -> LunarisCodexConfig:
    """Configuration designed to NOT activate LayerScale (matches base_config)."""
    return copy.deepcopy(base_config)


@pytest.fixture
def dummy_input_tensor(base_config: LunarisCodexConfig) -> torch.Tensor:
    """Dummy input tensor (batch_size, seq_len, d_model)."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, base_config.d_model)

@pytest.fixture
def dummy_input_ids(base_config: LunarisCodexConfig) -> torch.Tensor:
    """Dummy input IDs (batch_size, seq_len)."""
    return torch.randint(0, base_config.vocab_size, (BATCH_SIZE, SEQ_LEN))

# --- LunarisCodexConfig Tests ---
def test_config_dropout_adjustment_small_model():
    config = LunarisCodexConfig(
        vocab_size=1000, d_model=64, n_layers=1, n_heads=2, ff_multiplier=2,
        dropout=0.2 
    )
    assert config.dropout == 0.05

def test_config_dropout_no_adjustment_large_model():
    config = LunarisCodexConfig(
        vocab_size=50257, d_model=768, n_layers=12, n_heads=12, ff_multiplier=4,
        dropout=0.1
    )
    assert config.dropout == 0.1 

# --- LoRALinear Tests ---
def test_lora_linear_initialization_and_shape(dummy_input_tensor: torch.Tensor, base_config: LunarisCodexConfig):
    in_features = base_config.d_model
    out_features = in_features * 2
    rank = 4
    x = dummy_input_tensor

    lora_layer_enabled = LoRALinear(in_features, out_features, rank=rank, bias=True)
    assert lora_layer_enabled.has_lora
    assert lora_layer_enabled.lora_A.shape == (in_features, rank)
    assert lora_layer_enabled.lora_B.shape == (rank, out_features)
    assert torch.all(lora_layer_enabled.lora_B == 0) 
    output_enabled = lora_layer_enabled(x)
    assert output_enabled.shape == (BATCH_SIZE, SEQ_LEN, out_features)

    lora_layer_disabled_rank_zero = LoRALinear(in_features, out_features, rank=0, bias=True)
    assert not lora_layer_disabled_rank_zero.has_lora
    output_disabled_rank_zero = lora_layer_disabled_rank_zero(x)
    assert output_disabled_rank_zero.shape == (BATCH_SIZE, SEQ_LEN, out_features)

    lora_layer_disabled_rank_none = LoRALinear(in_features, out_features, rank=None, bias=True)
    assert not lora_layer_disabled_rank_none.has_lora
    output_disabled_rank_none = lora_layer_disabled_rank_none(x)
    assert output_disabled_rank_none.shape == (BATCH_SIZE, SEQ_LEN, out_features)

def test_lora_linear_effect_on_output(dummy_input_tensor: torch.Tensor, base_config: LunarisCodexConfig):
    torch.manual_seed(42) 
    in_features = base_config.d_model
    out_features = base_config.d_model 
    rank = 4
    x = dummy_input_tensor

    base_layer = LoRALinear(in_features, out_features, rank=0, bias=False)
    base_output = base_layer(x)

    lora_layer = LoRALinear(in_features, out_features, rank=rank, bias=False)
    lora_layer.weight.data = base_layer.weight.data.clone() 

    output_lora_before_training_delta = lora_layer(x)
    assert torch.allclose(base_output, output_lora_before_training_delta, atol=1e-6)

    with torch.no_grad():
        lora_layer.lora_A.data.uniform_(-0.1, 0.1) 
        lora_layer.lora_B.data.uniform_(-0.1, 0.1) 

    output_lora_after_simulated_training_delta = lora_layer(x)
    assert not torch.allclose(base_output, output_lora_after_simulated_training_delta, atol=1e-6)

# --- ALiBi Tests ---
@pytest.mark.parametrize("n_heads_test", [1, 2, 4, 8, 16])
def test_alibi_slopes_generation(base_config: LunarisCodexConfig, n_heads_test: int):
    config = copy.deepcopy(base_config)
    config.n_heads = n_heads_test
    # Ensure d_model is compatible with n_heads for the test
    config.d_model = D_MODEL_SMALL_FOR_BASE * (n_heads_test // N_HEADS_SMALL_FOR_BASE if N_HEADS_SMALL_FOR_BASE > 0 and n_heads_test % N_HEADS_SMALL_FOR_BASE == 0 else n_heads_test)
    if config.d_model < n_heads_test or config.d_model % n_heads_test != 0 : 
        config.d_model = n_heads_test * (D_MODEL_SMALL_FOR_BASE // N_HEADS_SMALL_FOR_BASE) # Make it a multiple
        if config.d_model == 0: config.d_model = n_heads_test # Fallback if N_HEADS_SMALL_FOR_BASE was 0

    model = LunarisMind(config) 
    assert model.alibi_slopes is not None
    assert model.alibi_slopes.shape == (n_heads_test,)
    # Slopes are now negative and should be increasing (becoming less negative)
    # e.g., -0.5, -0.25, -0.125 ...
    for i in range(n_heads_test - 1):
        # Check that slope[i] <= slope[i+1] (with a small tolerance for floating point)
        # A slope is "smaller" if it's more negative.
        assert model.alibi_slopes[i] <= model.alibi_slopes[i+1] + 1e-6, \
            f"ALiBi slopes not increasing for n_heads={n_heads_test}: slope[{i}]={model.alibi_slopes[i].item()} vs slope[{i+1}]={model.alibi_slopes[i+1].item()}"

def test_get_alibi_attention_bias(base_config: LunarisCodexConfig):
    model = LunarisMind(base_config)
    test_seq_len = 8
    device = torch.device("cpu")
    model.to(device)

    alibi_bias = model.get_alibi_attention_bias(test_seq_len, device)
    assert alibi_bias.shape == (1, base_config.n_heads, test_seq_len, test_seq_len)

    for h_idx in range(base_config.n_heads):
        for i in range(test_seq_len): 
            for j in range(test_seq_len): 
                current_bias_val = alibi_bias[0, h_idx, i, j].item()
                if j > i:
                    assert current_bias_val == float('-inf'), f"Future pos ({i},{j}) not -inf"
                else:
                    assert current_bias_val != float('-inf'), f"Past/current pos ({i},{j}) is -inf"
                    # With negative slopes and (i-j) for relative_positions:
                    # if j < i (past), i-j > 0, slope * (i-j) is negative.
                    # if j = i (current), i-j = 0, slope * (i-j) is 0.
                    if j < i:
                        assert current_bias_val < 0, f"Past pos ({i},{j}) bias {current_bias_val} not negative"
            assert alibi_bias[0, h_idx, i, i].item() == 0.0, f"Diagonal ({i},{i}) not 0.0"

# ... (o resto do test_model.py continua igual) ...

# --- FeedForward Tests ---
@pytest.mark.parametrize("activation_fn_name", ["swiglu", "gelu"])
def test_feed_forward_shape_and_activations(base_config: LunarisCodexConfig, dummy_input_tensor: torch.Tensor, activation_fn_name: str):
    config = copy.deepcopy(base_config)
    config.activation = activation_fn_name
    d_ff_internal = config.d_model * config.ff_multiplier

    ff_layer = FeedForward(
        d_model=config.d_model,
        d_ff=d_ff_internal,
        dropout=config.dropout,
        activation=config.activation,
        lora_rank=config.lora_rank 
    )
    output = ff_layer(dummy_input_tensor)
    assert output.shape == dummy_input_tensor.shape

# --- SelfAttention Tests ---
def test_self_attention_manual_fallback_shapes_with_masks(base_config: LunarisCodexConfig, dummy_input_tensor: torch.Tensor):
    config = copy.deepcopy(base_config)
    config.use_flash_attention_if_available = False 
    attention_layer = SelfAttention(config)
    attention_layer.eval() 

    batch_size, seq_len, _ = dummy_input_tensor.shape
    device = dummy_input_tensor.device
    attention_layer.to(device)

    temp_model_for_alibi = LunarisMind(config)
    temp_model_for_alibi.to(device)
    alibi_combined_bias_for_test = temp_model_for_alibi.get_alibi_attention_bias(seq_len, device)

    output_causal_only, _ = attention_layer(
        dummy_input_tensor, 
        alibi_combined_bias=alibi_combined_bias_for_test, 
        padding_additive_mask=None,
        past_key_value=None 
    )
    assert output_causal_only.shape == dummy_input_tensor.shape

    padding_additive_mask = torch.zeros(batch_size, 1, 1, seq_len, device=device, dtype=dummy_input_tensor.dtype)
    if batch_size > 0 and seq_len > 1: 
        padding_additive_mask[0, :, :, seq_len // 2:] = float('-inf') 

    output_with_padding, _ = attention_layer(
        dummy_input_tensor, 
        alibi_combined_bias=alibi_combined_bias_for_test, 
        padding_additive_mask=padding_additive_mask,
        past_key_value=None
    )
    assert output_with_padding.shape == dummy_input_tensor.shape

# --- TransformerDecoderBlock Tests ---
def test_transformer_decoder_block_shape(base_config: LunarisCodexConfig, dummy_input_tensor: torch.Tensor):
    config = copy.deepcopy(base_config)
    decoder_block = TransformerDecoderBlock(config)
    decoder_block.eval() 

    seq_len = dummy_input_tensor.shape[1]
    device = dummy_input_tensor.device
    decoder_block.to(device)

    temp_model_for_alibi = LunarisMind(config)
    temp_model_for_alibi.to(device)
    alibi_combined_bias_for_test = temp_model_for_alibi.get_alibi_attention_bias(seq_len, device)

    output, cache = decoder_block(
        dummy_input_tensor, 
        alibi_combined_bias=alibi_combined_bias_for_test, 
        padding_additive_mask=None,
        past_key_value=None, 
        use_cache=False      
    )
    assert output.shape == dummy_input_tensor.shape
    assert cache is None

def _calculate_expected_params_for_layerscale(config: LunarisCodexConfig) -> int:
    """Helper function to calculate estimated params exactly as in TransformerDecoderBlock."""
    if config.vocab_size is None:
        # This case should ideally not happen if config is properly set for the test
        # but if it does, LayerScale would be disabled in the block.
        return 0 
    d_ff = config.d_model * config.ff_multiplier
    if config.activation == "swiglu":
        ffn_params_per_layer = config.d_model * (d_ff * 2) + d_ff * config.d_model
    else: # gelu or other
        ffn_params_per_layer = config.d_model * d_ff + d_ff * config.d_model
    
    # This is the formula from TransformerDecoderBlock.__init__
    estimated_params = (
        config.vocab_size * config.d_model + # Token Embedding (part of total model, not just one block)
        config.n_layers * (                   # This is for the whole model, so the block uses it
            4 * config.d_model * config.d_model + # Self-Attention (QKV+Output Proj, simplified)
            ffn_params_per_layer +                # FeedForward
            4 * config.d_model                    # Layer Norms (approx. for 2 LN per block * 2 params each)
        )
    )
    return estimated_params

def test_transformer_decoder_block_activates_layerscale(config_for_layerscale_test: LunarisCodexConfig):
    config = config_for_layerscale_test
    # Fixture should provide a valid vocab_size
    assert config.vocab_size is not None, "config_for_layerscale_test must have vocab_size"

    x = torch.randn(BATCH_SIZE, SEQ_LEN, config.d_model)
    device = torch.device("cpu")
    x = x.to(device)

    decoder_block = TransformerDecoderBlock(config)
    decoder_block.to(device)
    decoder_block.eval()

    expected_params_in_block = _calculate_expected_params_for_layerscale(config)
    expected_use_layerscale_in_block = expected_params_in_block > 50_000_000
    
    assert decoder_block.use_layerscale == expected_use_layerscale_in_block, \
        f"LayerScale activation mismatch. Block says: {decoder_block.use_layerscale}, Expected: {expected_use_layerscale_in_block} (Params: {expected_params_in_block/1e6:.1f}M)"

    assert expected_use_layerscale_in_block == True, \
        f"Test setup error: config_for_layerscale_test (params: {expected_params_in_block/1e6:.1f}M) did not result in expected LayerScale activation. It resulted in {expected_params_in_block} params."

    if decoder_block.use_layerscale: 
        assert hasattr(decoder_block, 'ls_gamma_1') and decoder_block.ls_gamma_1 is not None
        assert hasattr(decoder_block, 'ls_gamma_2') and decoder_block.ls_gamma_2 is not None

    temp_model_for_alibi = LunarisMind(config)
    temp_model_for_alibi.to(device)
    alibi_combined_bias_for_test = temp_model_for_alibi.get_alibi_attention_bias(SEQ_LEN, device)
    
    output, _ = decoder_block(x, alibi_combined_bias_for_test, padding_additive_mask=None, use_cache=False)
    assert output.shape == x.shape

def test_transformer_decoder_block_deactivates_layerscale(config_no_layerscale_test: LunarisCodexConfig):
    config = config_no_layerscale_test
    assert config.vocab_size is not None, "config_no_layerscale_test must have vocab_size"

    x = torch.randn(BATCH_SIZE, SEQ_LEN, config.d_model)
    device = torch.device("cpu")
    x = x.to(device)

    decoder_block = TransformerDecoderBlock(config)
    decoder_block.to(device)
    decoder_block.eval()
    
    expected_params_in_block = _calculate_expected_params_for_layerscale(config)
    expected_use_layerscale_in_block = expected_params_in_block > 50_000_000

    assert decoder_block.use_layerscale == expected_use_layerscale_in_block, \
        f"LayerScale activation mismatch. Block says: {decoder_block.use_layerscale}, Expected: {expected_use_layerscale_in_block} (Params: {expected_params_in_block/1e6:.1f}M)"
    
    assert expected_use_layerscale_in_block == False, \
        f"Test setup error: config_no_layerscale_test (params: {expected_params_in_block/1e6:.1f}M) did not result in expected LayerScale deactivation."

    if not decoder_block.use_layerscale:
        assert not hasattr(decoder_block, 'ls_gamma_1')
        assert not hasattr(decoder_block, 'ls_gamma_2')

    temp_model_for_alibi = LunarisMind(config)
    temp_model_for_alibi.to(device)
    alibi_combined_bias_for_test = temp_model_for_alibi.get_alibi_attention_bias(SEQ_LEN, device)
    output, _ = decoder_block(x, alibi_combined_bias_for_test, padding_additive_mask=None, use_cache=False)
    assert output.shape == x.shape

# --- LunarisMind Tests ---
def test_lunaris_mind_forward_pass_shape_and_tied_weights(base_config: LunarisCodexConfig, dummy_input_ids: torch.Tensor):
    config = base_config
    model = LunarisMind(config)
    model.eval() 

    assert model.lm_head.weight is model.token_embedding.weight 

    attention_mask = torch.ones_like(dummy_input_ids, dtype=torch.long) # Ensure dtype is long

    with torch.no_grad():
        logits_output = model(dummy_input_ids, attention_mask=attention_mask, use_cache=False)
    
    assert isinstance(logits_output, torch.Tensor)
    assert logits_output.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)


def test_lunaris_mind_generate_methods(base_config: LunarisCodexConfig, dummy_input_ids: torch.Tensor):
    config = copy.deepcopy(base_config)
    config.vocab_size = VOCAB_SIZE_SMALL 
    model = LunarisMind(config)
    model.eval()

    prompt_len = SEQ_LEN // 2
    prompt_ids = dummy_input_ids[:, :prompt_len]
    max_new_tokens_greedy = 5
    eos_token_id_test = config.vocab_size - 1 

    torch.manual_seed(42)
    initial_attention_mask_greedy = torch.ones_like(prompt_ids, dtype=torch.long)
    if config.pad_token_id is not None and config.pad_token_id >= 0:
        initial_attention_mask_greedy[prompt_ids == config.pad_token_id] = 0


    generated_greedy = model.generate(
        input_ids=prompt_ids,
        attention_mask=initial_attention_mask_greedy, 
        max_new_tokens=max_new_tokens_greedy,
        temperature=0.001, 
        top_k=1,
        pad_token_id=config.pad_token_id 
    )
    expected_len_greedy = prompt_len + max_new_tokens_greedy
    assert generated_greedy.shape == (BATCH_SIZE, expected_len_greedy)
    assert torch.all(generated_greedy[:, :prompt_len] == prompt_ids) 

    torch.manual_seed(123)
    class MockModelEOS(LunarisMind):
        def __init__(self, config, eos_token_to_inject, inject_at_step):
            super().__init__(config)
            self.eos_token_to_inject = eos_token_to_inject
            self.inject_at_step = inject_at_step
            self.current_generation_step_in_loop = 0 
            self._is_in_generate_loop_flag = False 

        # Ensure the signature matches the base class exactly
        def forward(self,
                    input_ids: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                    use_cache: bool = False
                    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] | torch.Tensor: # Match return type
            
            # Call super().forward and unpack its return value correctly based on use_cache
            if use_cache:
                logits, new_past_key_values = super().forward(
                    input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache 
                )
            else:
                logits = super().forward(
                    input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values, # Will be None
                    use_cache=use_cache
                )
                new_past_key_values = None # Explicitly None if not using cache


            if not self.training and self._is_in_generate_loop_flag:
                # This mocking logic assumes generate calls forward with use_cache=True and single token inputs after prompt
                if self.current_generation_step_in_loop == (self.inject_at_step - 1): 
                    logits[:, -1, self.eos_token_to_inject] = 1e9 
                self.current_generation_step_in_loop += 1
            
            if use_cache:
                return logits, new_past_key_values
            else:
                return logits


        # Override generate to set flags for the mock forward
        def generate(self, *args, **kwargs):
            self._is_in_generate_loop_flag = True
            self.current_generation_step_in_loop = 0 
            output = super().generate(*args, **kwargs)
            self._is_in_generate_loop_flag = False
            return output

    mock_model_eos = MockModelEOS(config, eos_token_id_test, inject_at_step=1) 
    mock_model_eos.eval()

    max_new_tokens_for_eos_test = 10
    initial_attention_mask_eos = torch.ones_like(prompt_ids, dtype=torch.long)
    if config.pad_token_id is not None and config.pad_token_id >= 0:
        initial_attention_mask_eos[prompt_ids == config.pad_token_id] = 0


    generated_with_eos = mock_model_eos.generate(
        input_ids=prompt_ids,
        attention_mask=initial_attention_mask_eos, 
        max_new_tokens=max_new_tokens_for_eos_test,
        temperature=0.7, 
        top_k=0, 
        top_p=1.0, 
        eos_token_id=eos_token_id_test,
        pad_token_id=config.pad_token_id
    )

    expected_len_with_eos = prompt_len + 1 
    assert generated_with_eos.shape[1] == expected_len_with_eos, \
        f"Generated length {generated_with_eos.shape[1]} not {expected_len_with_eos}"
    assert generated_with_eos[0, -1] == eos_token_id_test 

    torch.manual_seed(45)
    initial_attention_mask_rep = torch.ones_like(prompt_ids, dtype=torch.long)
    if config.pad_token_id is not None and config.pad_token_id >= 0:
        initial_attention_mask_rep[prompt_ids == config.pad_token_id] = 0

    _ = model.generate(
        input_ids=prompt_ids,
        attention_mask=initial_attention_mask_rep,
        max_new_tokens=max_new_tokens_greedy,
        repetition_penalty=1.5,
        pad_token_id=config.pad_token_id
    )

# --- count_parameters Test ---
def test_count_parameters_function(base_config: LunarisCodexConfig):
    config = copy.deepcopy(base_config)
    # Ensure vocab_size is set for this calculation, as base_config has a small one
    if config.vocab_size is None or config.vocab_size < 1000: 
        config.vocab_size = VOCAB_SIZE_SMALL # Use the small vocab from constants
    
    model = LunarisMind(config)
    # count_parameters in model.py counts all parameters if requires_grad is True.
    # When a model is freshly initialized, all nn.Parameters have requires_grad=True by default.
    params_from_func = count_parameters(model)

    # Calculate expected parameters by summing numel() of all nn.Parameters
    expected_params = 0
    for param in model.parameters(): # This iterates over all nn.Parameters
        expected_params += param.numel()
    
    assert params_from_func == expected_params, \
        f"Parameter count mismatch. Function: {params_from_func}, Expected (all params): {expected_params}"

