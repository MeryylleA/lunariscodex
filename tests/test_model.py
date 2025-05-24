# tests/test_model.py
import torch
import pytest
import math
import copy

from model import (
    LunarisCodexConfig,
    LoRALinear,
    FeedForward,
    SelfAttention,
    TransformerDecoderBlock,
    LunarisMind,
    count_parameters
)

# --- Test Constants ---
BATCH_SIZE = 2
SEQ_LEN = 16
D_MODEL_SMALL = 32
N_HEADS_SMALL = 2
VOCAB_SIZE_SMALL = 50
FF_MULTIPLIER_SMALL = 2

# --- Fixtures ---
@pytest.fixture(scope="module")
def base_config() -> LunarisCodexConfig:
    return LunarisCodexConfig(
        vocab_size=VOCAB_SIZE_SMALL,
        d_model=D_MODEL_SMALL,
        n_layers=2,
        n_heads=N_HEADS_SMALL,
        max_seq_len=SEQ_LEN * 2,
        dropout=0.0,
        lora_rank=0,
        use_flash_attention_if_available=False,
        ff_multiplier=FF_MULTIPLIER_SMALL,
        activation="swiglu"
    )

@pytest.fixture
def config_with_lora(base_config: LunarisCodexConfig) -> LunarisCodexConfig:
    config = copy.deepcopy(base_config)
    config.lora_rank = 4
    return config

@pytest.fixture
def config_with_layerscale(base_config: LunarisCodexConfig) -> LunarisCodexConfig:
    config = copy.deepcopy(base_config)
    config.d_model = 1024
    config.n_layers = 12
    config.ff_multiplier = 4
    return config

@pytest.fixture
def dummy_input_tensor(base_config: LunarisCodexConfig) -> torch.Tensor:
    return torch.randn(BATCH_SIZE, SEQ_LEN, base_config.d_model)

@pytest.fixture
def dummy_input_ids(base_config: LunarisCodexConfig) -> torch.Tensor:
    return torch.randint(0, base_config.vocab_size, (BATCH_SIZE, SEQ_LEN))

# --- Tests for LunarisCodexConfig ---
def test_config_dropout_adjustment_small_model():
    config = LunarisCodexConfig(
        vocab_size=100, d_model=32, n_layers=1, n_heads=2, ff_multiplier=2,
        dropout=0.2
    )
    assert config.dropout == 0.05

def test_config_dropout_no_adjustment_large_model():
    config = LunarisCodexConfig(
        vocab_size=50000, d_model=768, n_layers=12, n_heads=12,
        dropout=0.1
    )
    assert config.dropout == 0.1

# --- Tests for LoRALinear ---
def test_lora_linear_initialization_and_shape(dummy_input_tensor: torch.Tensor):
    in_features = D_MODEL_SMALL
    out_features = D_MODEL_SMALL * 2
    rank = 4
    x = dummy_input_tensor[:, :, :in_features]

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

def test_lora_linear_effect_on_output(dummy_input_tensor: torch.Tensor):
    torch.manual_seed(42)
    in_features = D_MODEL_SMALL
    out_features = D_MODEL_SMALL
    rank = 4
    x = dummy_input_tensor[:, :, :in_features]

    base_layer = LoRALinear(in_features, out_features, rank=0, bias=False)
    base_output = base_layer(x)

    lora_layer = LoRALinear(in_features, out_features, rank=rank, bias=False)
    lora_layer.weight.data = base_layer.weight.data.clone()

    output_lora_before_training = lora_layer(x)
    assert torch.allclose(base_output, output_lora_before_training, atol=1e-6)

    with torch.no_grad():
        lora_layer.lora_A.data.uniform_(-0.1, 0.1)
        lora_layer.lora_B.data.uniform_(-0.1, 0.1)

    output_lora_after_training = lora_layer(x)
    assert not torch.allclose(base_output, output_lora_after_training, atol=1e-6)

# --- Tests for ALiBi ---
@pytest.mark.parametrize("n_heads_test", [1, 2, 4, 8, 16])
def test_alibi_slopes_generation(base_config: LunarisCodexConfig, n_heads_test: int):
    config = copy.deepcopy(base_config)
    config.n_heads = n_heads_test
    if config.d_model % n_heads_test != 0:
        pytest.skip(f"Skipping n_heads={n_heads_test} as d_model={config.d_model} is not divisible by it.")

    model = LunarisMind(config)
    assert model.alibi_slopes is not None
    assert model.alibi_slopes.shape == (n_heads_test,)
    for i in range(n_heads_test - 1):
        assert model.alibi_slopes[i] >= model.alibi_slopes[i+1] * 0.99

def test_get_alibi_attention_bias(base_config: LunarisCodexConfig):
    model = LunarisMind(base_config)
    test_seq_len = 8
    device = torch.device("cpu")
    model.to(device)
    alibi_bias = model.get_alibi_attention_bias(test_seq_len, device)

    assert alibi_bias.shape == (base_config.n_heads, test_seq_len, test_seq_len)
    for h in range(base_config.n_heads):
        for i in range(test_seq_len):
            for j in range(test_seq_len):
                if j > i:
                    assert alibi_bias[h, i, j] == float('-inf')
                else:
                    assert alibi_bias[h, i, j] != float('-inf')
            assert alibi_bias[h, i, i] == 0.0
            if i > 0:
                assert alibi_bias[h, i, i-1] < 0.0
                if i > 1:
                    assert alibi_bias[h, i, i-2] <= alibi_bias[h, i, i-1]

# --- Tests for FeedForward ---
@pytest.mark.parametrize("activation_fn_name", ["swiglu", "gelu"])
def test_feed_forward_shape_and_activations(base_config: LunarisCodexConfig, dummy_input_tensor: torch.Tensor, activation_fn_name: str):
    config = copy.deepcopy(base_config)
    config.activation = activation_fn_name
    ffn_intermediate_dim = config.d_model * config.ff_multiplier
    ff_layer = FeedForward(
        d_model=config.d_model, d_ff=ffn_intermediate_dim, dropout=config.dropout,
        activation=config.activation, lora_rank=config.lora_rank
    )
    output = ff_layer(dummy_input_tensor)
    assert output.shape == dummy_input_tensor.shape

# --- Tests for SelfAttention ---
def test_self_attention_manual_fallback_shapes_with_masks(base_config: LunarisCodexConfig, dummy_input_tensor: torch.Tensor):
    config = base_config
    attention_layer = SelfAttention(config)
    batch_size, seq_len, _ = dummy_input_tensor.shape
    device = dummy_input_tensor.device

    temp_model_for_alibi = LunarisMind(config)
    temp_model_for_alibi.to(device)
    alibi_combined_bias_for_test = temp_model_for_alibi.get_alibi_attention_bias(seq_len, device)

    output_causal_only = attention_layer(dummy_input_tensor, alibi_combined_bias_for_test, padding_additive_mask=None)
    assert output_causal_only.shape == dummy_input_tensor.shape

    padding_additive_mask = torch.zeros(batch_size, 1, 1, seq_len, device=device, dtype=dummy_input_tensor.dtype)
    if batch_size > 1:
        padding_additive_mask[1, :, :, seq_len // 2:] = float('-inf')

    output_with_padding = attention_layer(dummy_input_tensor, alibi_combined_bias_for_test, padding_additive_mask)
    assert output_with_padding.shape == dummy_input_tensor.shape

# --- Tests for TransformerDecoderBlock ---
def test_transformer_decoder_block_shape(base_config: LunarisCodexConfig, dummy_input_tensor: torch.Tensor):
    config = base_config
    decoder_block = TransformerDecoderBlock(config)
    seq_len = dummy_input_tensor.shape[1]
    device = dummy_input_tensor.device

    temp_model_for_alibi = LunarisMind(config)
    temp_model_for_alibi.to(device)
    alibi_combined_bias_for_test = temp_model_for_alibi.get_alibi_attention_bias(seq_len, device)

    output = decoder_block(dummy_input_tensor, alibi_combined_bias_for_test, padding_additive_mask=None)
    assert output.shape == dummy_input_tensor.shape

def test_transformer_decoder_block_with_layerscale(config_with_layerscale: LunarisCodexConfig):
    config = config_with_layerscale
    x = torch.randn(BATCH_SIZE, SEQ_LEN, config.d_model)
    device = torch.device("cpu")
    x = x.to(device)

    decoder_block = TransformerDecoderBlock(config)
    decoder_block.to(device)

    # For debugging, print details if assert fails
    block_ffn_intermediate_dim = config.d_model * config.ff_multiplier
    block_estimated_params = config.d_model * config.n_layers * (config.d_model * 4 + block_ffn_intermediate_dim)
    print(f"Testing LayerScale: d_model={config.d_model}, n_layers={config.n_layers}, ff_mult={config.ff_multiplier}, block_est_params={block_estimated_params / 1_000_000:.1f}M, use_layerscale={decoder_block.use_layerscale}")

    assert decoder_block.use_layerscale, f"LayerScale should be enabled for config with d_model={config.d_model}, n_layers={config.n_layers}"
    assert hasattr(decoder_block, 'ls_gamma_1') and decoder_block.ls_gamma_1 is not None
    assert hasattr(decoder_block, 'ls_gamma_2') and decoder_block.ls_gamma_2 is not None

    temp_model_for_alibi = LunarisMind(config)
    temp_model_for_alibi.to(device)
    alibi_combined_bias_for_test = temp_model_for_alibi.get_alibi_attention_bias(SEQ_LEN, device)

    output = decoder_block(x, alibi_combined_bias_for_test, padding_additive_mask=None)
    assert output.shape == x.shape

# --- Tests for LunarisMind (Full Model) ---
def test_lunaris_mind_forward_pass_shape_and_tied_weights(base_config: LunarisCodexConfig, dummy_input_ids: torch.Tensor):
    config = base_config
    model = LunarisMind(config)
    model.eval()

    assert model.lm_head.weight is model.token_embedding.weight
    attention_mask = torch.ones_like(dummy_input_ids)
    with torch.no_grad():
        logits = model(dummy_input_ids, attention_mask=attention_mask)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)

def test_lunaris_mind_generate_methods(base_config: LunarisCodexConfig, dummy_input_ids: torch.Tensor):
    config = copy.deepcopy(base_config)
    config.vocab_size = VOCAB_SIZE_SMALL
    model = LunarisMind(config)
    model.eval()

    prompt_len = SEQ_LEN // 2
    prompt_ids = dummy_input_ids[:, :prompt_len]
    max_new_tokens_greedy = 5
    eos_token_id_test = VOCAB_SIZE_SMALL - 1

    # Test 1: Greedy
    torch.manual_seed(42)
    generated_greedy = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens_greedy,
        temperature=0.001,
        top_k=1
    )
    expected_len_greedy = prompt_len + max_new_tokens_greedy
    assert generated_greedy.shape == (BATCH_SIZE, expected_len_greedy)
    assert torch.all(generated_greedy[:, :prompt_len] == prompt_ids)

    # Test 2: Generation with EOS
    torch.manual_seed(123) # Consistent seed for this specific test part

    class MockModelEOS(LunarisMind):
        def __init__(self, config, eos_token_to_inject, inject_at_step):
            super().__init__(config)
            self.eos_token_to_inject = eos_token_to_inject
            self.inject_at_step = inject_at_step
            self.current_generation_step = 0
            self._is_in_generate_loop = False

        def generate(self, *args, **kwargs):
            self._is_in_generate_loop = True
            self.current_generation_step = 0 # Reset step for each generate call
            output = super().generate(*args, **kwargs)
            self._is_in_generate_loop = False
            return output

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
            logits = super().forward(input_ids, attention_mask)
            if not self.training and self._is_in_generate_loop:
                # This logic applies to the prediction for the *next* token
                if self.current_generation_step == (self.inject_at_step - 1):
                    logits[:, -1, self.eos_token_to_inject] = 1e9
                self.current_generation_step += 1
            return logits

    # Inject EOS as the first new token (inject_at_step=1)
    mock_model_eos = MockModelEOS(config, eos_token_id_test, inject_at_step=1)
    mock_model_eos.eval()

    max_new_tokens_for_eos_test = 10

    generated_with_eos = mock_model_eos.generate(
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens_for_eos_test,
        temperature=0.7,
        top_k=0,
        top_p=1.0,
        eos_token_id=eos_token_id_test
    )

    expected_len_with_eos = prompt_len + 1 # Prompt + 1 EOS token

    assert generated_with_eos.shape[1] == expected_len_with_eos, \
        f"Generation should stop after EOS. Expected len {expected_len_with_eos}, got {generated_with_eos.shape[1]}"
    assert generated_with_eos[0, -1] == eos_token_id_test, \
        "Last token of the generated sequence should be the EOS token"

    # Test 3: Repetition Penalty (execution check)
    torch.manual_seed(45)
    _ = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens_greedy,
        repetition_penalty=1.5
    )

def test_count_parameters_function(base_config: LunarisCodexConfig):
    model = LunarisMind(base_config)
    num_params = count_parameters(model)
    expected_params = 22400
    assert num_params == expected_params, f"Parameter count mismatch. Expected {expected_params}, got {num_params}"
