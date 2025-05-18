# tests/test_model.py
import torch
import pytest # Você precisará de pytest instalado no seu ambiente de CI

# Supondo que model.py está na raiz do projeto ou seu PYTHONPATH está configurado
# Se model.py estiver na raiz, e você rodar pytest da raiz, isso deve funcionar.
# Se não, você pode precisar ajustar o caminho de importação ou configurar o PYTHONPATH no CI.
# Para simplicidade, vamos assumir que model.py é importável diretamente.
from model import LunarisCodexConfig, LoRALinear

def test_lora_linear_initialization_and_shape():
    """
    Tests basic initialization of LoRALinear and output shape.
    """
    in_features = 128
    out_features = 256
    rank = 8

    # Test with LoRA enabled
    lora_layer_with_lora = LoRALinear(in_features, out_features, rank=rank, bias=False)
    assert lora_layer_with_lora.has_lora, "LoRA should be enabled when rank > 0"
    assert lora_layer_with_lora.lora_A.shape == (in_features, rank)
    assert lora_layer_with_lora.lora_B.shape == (rank, out_features)

    dummy_input = torch.randn(2, 10, in_features) # Batch, SeqLen, Features
    output_with_lora = lora_layer_with_lora(dummy_input)
    assert output_with_lora.shape == (2, 10, out_features), "Output shape mismatch with LoRA"

    # Test with LoRA disabled (rank=0)
    lora_layer_no_lora = LoRALinear(in_features, out_features, rank=0, bias=False)
    assert not lora_layer_no_lora.has_lora, "LoRA should be disabled when rank = 0"

    output_no_lora = lora_layer_no_lora(dummy_input)
    assert output_no_lora.shape == (2, 10, out_features), "Output shape mismatch without LoRA"

    # Test with LoRA disabled (rank=None)
    lora_layer_none_lora = LoRALinear(in_features, out_features, rank=None, bias=False)
    assert not lora_layer_none_lora.has_lora, "LoRA should be disabled when rank is None"

    output_none_lora = lora_layer_none_lora(dummy_input)
    assert output_none_lora.shape == (2, 10, out_features), "Output shape mismatch with LoRA rank None"

# Você pode adicionar mais testes aqui para outras partes do model.py
# def test_another_component():
#     pass
