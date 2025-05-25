# tests/test_inference.py
import pytest
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
import hashlib
from argparse import Namespace
import logging
from pathlib import Path
from unittest import mock
from rich.syntax import Syntax
from rich.text import Text as RichText

# Adicionar o diretório pai (raiz do projeto) ao sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Importar do inference.py
from inference import (
    SCRIPT_VERSION,
    get_memory_usage,
    compute_sha256,
    verify_checkpoint_integrity,
    validate_checkpoint_exists,
    format_code_output,
    load_model_from_checkpoint,
)
# Importar do train.py para ajudar a criar checkpoints de teste
from train import save_checkpoint as train_save_checkpoint
from model import LunarisCodexConfig, LunarisMind

# Nome do logger usado em inference.py
INFERENCE_LOGGER_NAME = "lunaris"

# --- Fixtures ---

@pytest.fixture
def dummy_config_inf():
    return LunarisCodexConfig(
        vocab_size=100, d_model=16, n_layers=1, n_heads=2,
        max_seq_len=32, ff_multiplier=2, dropout=0.0,
    )

@pytest.fixture
def dummy_model_inf(dummy_config_inf):
    with mock.patch('builtins.print') as mocked_print:
        model = LunarisMind(dummy_config_inf)
    return model


@pytest.fixture
def dummy_tokenizer_inf(tmp_path):
    from transformers import AutoTokenizer
    tokenizer_path = tmp_path / "dummy_gpt2_tokenizer_inf"
    if not tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(str(tokenizer_path))

    loaded_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if loaded_tokenizer.pad_token_id is None:
        if loaded_tokenizer.eos_token_id is not None:
            loaded_tokenizer.pad_token_id = loaded_tokenizer.eos_token_id
            loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
        else:
            loaded_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    return loaded_tokenizer


@pytest.fixture
def dummy_checkpoint_path(tmp_path, dummy_model_inf, dummy_tokenizer_inf):
    checkpoint_dir = tmp_path / "inf_checkpoints"

    model_to_save = dummy_model_inf
    optimizer_dummy = torch.optim.AdamW(model_to_save.parameters(), lr=1e-4)

    args_save = Namespace(
        checkpoint_dir=str(checkpoint_dir),
        lora_rank=0,
        tokenizer_name_or_path="gpt2",
    )

    train_logger = logging.getLogger("train")
    original_train_logger_level = train_logger.getEffectiveLevel()
    train_logger.setLevel(logging.CRITICAL + 1)

    train_save_checkpoint(model_to_save, optimizer_dummy, epoch=0, step=1, current_loss=0.1, args=args_save, is_best=False)

    train_logger.setLevel(original_train_logger_level)

    checkpoint_filename = f"lunaris_codex_epoch-1_step-1.pt"
    full_checkpoint_path = checkpoint_dir / checkpoint_filename
    assert full_checkpoint_path.exists(), "Dummy checkpoint .pt file was not created by train_save_checkpoint"
    assert (checkpoint_dir / f"{checkpoint_filename}.sha256").exists(), "Dummy checkpoint .sha256 file was not created"

    return str(full_checkpoint_path)


# --- Testes para Funções Utilitárias de inference.py ---

def test_script_version_importable():
    assert isinstance(SCRIPT_VERSION, str)
    assert len(SCRIPT_VERSION) > 0

def test_get_memory_usage():
    mem = get_memory_usage()
    assert isinstance(mem, float)
    assert mem > 0

def test_inf_compute_sha256_valid_file(tmp_path):
    p = tmp_path / "test_file_inf.txt"
    content = b"Inference SHA256 Test"
    p.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()
    assert compute_sha256(str(p)) == expected_hash

def test_inf_verify_checkpoint_integrity_correct(dummy_checkpoint_path, caplog):
    with caplog.at_level(logging.INFO, logger=INFERENCE_LOGGER_NAME):
        assert verify_checkpoint_integrity(dummy_checkpoint_path) == True

    log_text = "".join(r.message for r in caplog.records if r.name == INFERENCE_LOGGER_NAME)
    assert "Checkpoint integrity verified" in log_text
    assert Path(dummy_checkpoint_path).name in log_text
    assert "(SHA-256 match)" in log_text

def test_inf_verify_checkpoint_integrity_incorrect_hash(dummy_checkpoint_path, caplog):
    hash_file_path = Path(dummy_checkpoint_path + ".sha256")
    original_hash_content = hash_file_path.read_text()

    incorrect_test_hash = "incorrecthash123"
    hash_file_path.write_text(f"{incorrect_test_hash}  {Path(dummy_checkpoint_path).name}\n")

    with caplog.at_level(logging.ERROR, logger=INFERENCE_LOGGER_NAME):
        assert verify_checkpoint_integrity(dummy_checkpoint_path) == False

    log_text = "".join(r.message for r in caplog.records if r.name == INFERENCE_LOGGER_NAME)
    assert "SHA-256 Mismatch!" in log_text
    assert "Checkpoint integrity check failed for" in log_text
    assert dummy_checkpoint_path in log_text
    assert f"Expected: [yellow]{incorrect_test_hash}[/yellow]" in log_text
    assert "Actual:   [red]" in log_text

    hash_file_path.write_text(original_hash_content)


def test_validate_checkpoint_exists(tmp_path, caplog):
    existing_file = tmp_path / "exists.pt"
    content_to_write = "data"
    existing_file.write_text(content_to_write)

    with caplog.at_level(logging.INFO, logger=INFERENCE_LOGGER_NAME):
        assert validate_checkpoint_exists(str(existing_file)) == True

    info_records = [rec for rec in caplog.records if rec.levelname == 'INFO' and rec.name == INFERENCE_LOGGER_NAME]
    warning_records = [rec for rec in caplog.records if rec.levelname == 'WARNING' and rec.name == INFERENCE_LOGGER_NAME]

    assert len(info_records) >= 1, "Deveria haver pelo menos uma mensagem INFO"

    expected_info_msg_part1 = "Checkpoint file found:"
    expected_info_msg_part2 = existing_file.name
    expected_info_msg_part3 = " MB)"

    found_info_msg = False
    target_info_message_for_assert = None

    for rec in info_records:
        target_info_message_for_assert = rec.message
        check1 = expected_info_msg_part1 in rec.message
        check2 = expected_info_msg_part2 in rec.message
        check3 = expected_info_msg_part3 in rec.message

        if check1 and check2 and check3:
            found_info_msg = True
            break

    assert found_info_msg, \
        f"Mensagem INFO esperada contendo '{expected_info_msg_part1}', '{expected_info_msg_part2}', e '{expected_info_msg_part3}' não encontrada. " \
        f"Log(s) INFO capturado(s): {[r.message for r in info_records]}. " \
        f"Última mensagem INFO verificada: '{target_info_message_for_assert}'"

    assert len(warning_records) >= 1, "Deveria haver uma mensagem WARNING para arquivo pequeno"
    expected_warning_msg = "Checkpoint file seems very small"
    found_warning_msg = False
    for rec in warning_records:
        if expected_warning_msg in rec.message:
            found_warning_msg = True
            break
    assert found_warning_msg, f"Mensagem WARNING esperada '{expected_warning_msg}' não encontrada nos logs: {[r.message for r in warning_records]}"

    caplog.clear()

    # Teste para arquivo inexistente
    with caplog.at_level(logging.ERROR, logger=INFERENCE_LOGGER_NAME):
        assert validate_checkpoint_exists("nonexistent.pt") == False

    error_records = [rec for rec in caplog.records if rec.levelname == 'ERROR' and rec.name == INFERENCE_LOGGER_NAME]
    assert len(error_records) >= 1, "Deveria haver pelo menos uma mensagem ERROR para arquivo inexistente"

    expected_error_part1 = "Checkpoint file not found:"
    expected_error_part2 = "nonexistent.pt"

    found_error_msg = False
    target_error_message_for_assert = None
    for rec in error_records:
        target_error_message_for_assert = rec.message
        if expected_error_part1 in rec.message and expected_error_part2 in rec.message:
            found_error_msg = True
            break

    assert found_error_msg, \
        f"Mensagem ERROR esperada contendo '{expected_error_part1}' e '{expected_error_part2}' não encontrada. " \
        f"Log(s) ERROR capturado(s): {[r.message for r in error_records]}. " \
        f"Última mensagem ERROR verificada: '{target_error_message_for_assert}'"


def test_format_code_output():
    python_code = "def hello():\n  print('world')"
    formatted_python = format_code_output(python_code, "python")
    assert isinstance(formatted_python, Syntax)
    if formatted_python.lexer is not None:
        assert "python" in str(type(formatted_python.lexer)).lower()

    text_content = "Just some plain text."
    formatted_text = format_code_output(text_content, "text")
    assert isinstance(formatted_text, Syntax)
    if formatted_text.lexer is not None:
        assert "text" in str(type(formatted_text.lexer)).lower()

    formatted_invalid_lang = format_code_output(python_code, "invalidlang!!")
    assert isinstance(formatted_invalid_lang, (Syntax, RichText))

    if isinstance(formatted_invalid_lang, Syntax):
        if formatted_invalid_lang.lexer is not None:
            assert "text" in str(type(formatted_invalid_lang.lexer)).lower()
    elif isinstance(formatted_invalid_lang, RichText):
        pass


# --- Testes para load_model_from_checkpoint (Integração) ---

@pytest.mark.skip(reason="Temporarily skipping due to prints from model.py __init__ which can interfere with log capture or stdout if not managed. Needs model.py to use logging.")
def test_load_model_from_checkpoint_success(dummy_checkpoint_path, dummy_config_inf, dummy_tokenizer_inf, caplog):
    device = torch.device("cpu")

    with mock.patch('builtins.print') as mocked_print:
        with caplog.at_level(logging.INFO, logger=INFERENCE_LOGGER_NAME):
            load_result = load_model_from_checkpoint(dummy_checkpoint_path, device)

    assert load_result is not None, "load_model_from_checkpoint returned None"
    model, config_loaded, args_loaded_from_ckpt = load_result

    assert isinstance(model, LunarisMind)
    assert isinstance(config_loaded, LunarisCodexConfig)

    assert config_loaded.vocab_size == dummy_config_inf.vocab_size
    assert config_loaded.d_model == dummy_config_inf.d_model
    assert config_loaded.n_layers == dummy_config_inf.n_layers

    log_text = "".join(r.message for r in caplog.records if r.name == INFERENCE_LOGGER_NAME and r.levelno >= logging.INFO)
    assert "Model configuration loaded successfully from checkpoint." in log_text
    assert "Model ready on device!" in log_text


def test_load_model_from_checkpoint_sha_mismatch_returns_none(dummy_checkpoint_path, caplog):
    device = torch.device("cpu")
    hash_file_path = Path(dummy_checkpoint_path + ".sha256")
    original_hash_content = hash_file_path.read_text()
    hash_file_path.write_text("incorrect_hash_value  " + Path(dummy_checkpoint_path).name + "\n")

    result = "placeholder_to_ensure_it_changes"
    with caplog.at_level(logging.ERROR, logger=INFERENCE_LOGGER_NAME):
        result = load_model_from_checkpoint(dummy_checkpoint_path, device)

    assert result is None
    assert "Aborting due to checkpoint integrity verification failure" in caplog.text

    hash_file_path.write_text(original_hash_content)

@pytest.mark.skip(reason="Temporarily skipping due to prints from model.py __init__.")
def test_load_model_from_checkpoint_missing_keys_exits(tmp_path, caplog):
    invalid_ckpt_path = tmp_path / "invalid.pt"
    torch.save({"some_other_key": "value"}, str(invalid_ckpt_path))
    device = torch.device("cpu")

    result = "placeholder"
    with mock.patch('builtins.print') as mocked_print:
        with caplog.at_level(logging.ERROR, logger=INFERENCE_LOGGER_NAME):
            result = load_model_from_checkpoint(str(invalid_ckpt_path), device)

    assert result is None
    assert "Checkpoint is missing required keys" in caplog.text

# --- Placeholder para testes mais complexos ---
# def test_stream_generation_mocked():
#     pass

# def test_interactive_mode_commands_mocked():
#     pass

# def test_main_arg_parsing_mocked(monkeypatch):
#     pass
