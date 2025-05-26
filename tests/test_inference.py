# tests/test_inference.py
import pytest
import os
import sys
import torch
import torch.nn as nn # Was missing, though not directly used in failing tests, good practice
import numpy as np    # Was missing
import random         # Was missing
import hashlib        # Was missing
from argparse import Namespace
import logging        # Was missing
from pathlib import Path
from unittest import mock
from rich.syntax import Syntax # Was missing
from rich.text import Text as RichText # Was missing

# Adicionar o diretório pai (raiz do projeto) ao sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from inference import (
    SCRIPT_VERSION,
    get_memory_usage,
    compute_sha256,
    verify_checkpoint_integrity,
    validate_checkpoint_exists,
    format_code_output,
    load_model_from_checkpoint,
)
from train import save_checkpoint as train_save_checkpoint
from model import LunarisCodexConfig, LunarisMind

INFERENCE_LOGGER_NAME = "lunaris_inference"
MODEL_LOGGER_NAME = "lunaris_model"
TRAIN_UTILS_LOGGER_NAME = "train"

# --- Fixtures ---
@pytest.fixture
def dummy_config_inf():
    return LunarisCodexConfig(
        vocab_size=100, d_model=16, n_layers=1, n_heads=2,
        max_seq_len=32, ff_multiplier=2, dropout=0.0,
        pad_token_id=0
    )

@pytest.fixture
def dummy_model_inf(dummy_config_inf):
    model_logger = logging.getLogger(MODEL_LOGGER_NAME)
    original_level = model_logger.getEffectiveLevel()
    model_logger.setLevel(logging.CRITICAL + 1)
    model = LunarisMind(dummy_config_inf)
    model_logger.setLevel(original_level)
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
        loaded_tokenizer.pad_token_id = loaded_tokenizer.eos_token_id if loaded_tokenizer.eos_token_id is not None else 50257
        loaded_tokenizer.pad_token = loaded_tokenizer.decode([loaded_tokenizer.pad_token_id])
    return loaded_tokenizer

@pytest.fixture
def dummy_checkpoint_path(tmp_path, dummy_model_inf):
    checkpoint_dir = tmp_path / "inf_checkpoints"
    model_to_save = dummy_model_inf
    optimizer_dummy = torch.optim.AdamW(model_to_save.parameters(), lr=1e-4)
    args_save = Namespace(
        checkpoint_dir=str(checkpoint_dir),
        tokenizer_name_or_path="gpt2",
        lora_rank=dummy_model_inf.config.lora_rank,
        learning_rate = 1e-4
    )
    train_save_logger = logging.getLogger(TRAIN_UTILS_LOGGER_NAME)
    original_train_save_level = train_save_logger.getEffectiveLevel()
    train_save_logger.setLevel(logging.CRITICAL + 1)
    train_save_checkpoint(model_to_save, optimizer_dummy, epoch=0, step=1, current_loss=0.1, args=args_save, is_best=False)
    train_save_logger.setLevel(original_train_save_level)
    checkpoint_filename = f"lunaris_codex_epoch-1_step-1.pt"
    full_checkpoint_path = checkpoint_dir / checkpoint_filename
    assert full_checkpoint_path.exists()
    assert (checkpoint_dir / f"{checkpoint_filename}.sha256").exists()
    return str(full_checkpoint_path)

# --- Testes ---
def test_script_version_importable():
    assert isinstance(SCRIPT_VERSION, str) and len(SCRIPT_VERSION) > 0

def test_get_memory_usage():
    mem = get_memory_usage()
    assert isinstance(mem, float) and mem > 0

def test_inf_compute_sha256_valid_file(tmp_path):
    p = tmp_path / "test_file_inf.txt"
    content = b"Inference SHA256 Test"
    p.write_bytes(content)
    assert compute_sha256(str(p)) == hashlib.sha256(content).hexdigest()

def test_inf_verify_checkpoint_integrity_correct(dummy_checkpoint_path, caplog):
    with caplog.at_level(logging.INFO, logger=INFERENCE_LOGGER_NAME):
        assert verify_checkpoint_integrity(dummy_checkpoint_path) is True
    assert any("Checkpoint integrity verified" in rec.message and "(SHA-256 match)" in rec.message
               for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME)

def test_inf_verify_checkpoint_integrity_incorrect_hash(dummy_checkpoint_path, caplog):
    hash_file_path = Path(dummy_checkpoint_path + ".sha256")
    original_hash_content = hash_file_path.read_text()
    checkpoint_basename = Path(dummy_checkpoint_path).name
    incorrect_hash = "manual_incorrect_hash"
    hash_file_path.write_text(f"{incorrect_hash}  {checkpoint_basename}\n")
    with caplog.at_level(logging.ERROR, logger=INFERENCE_LOGGER_NAME):
        assert verify_checkpoint_integrity(dummy_checkpoint_path) is False
    error_logs = [rec.getMessage() for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME and rec.levelno >= logging.ERROR]
    assert any("SHA-256 Mismatch!" in log for log in error_logs)
    assert any(f"Expected: [yellow]{incorrect_hash}[/yellow]" in log for log in error_logs)
    hash_file_path.write_text(original_hash_content)

def test_validate_checkpoint_exists(tmp_path, caplog):
    small_file = tmp_path / "small.pt"
    content_small = b"data" * 10
    small_file.write_bytes(content_small)
    with caplog.at_level(logging.INFO, logger=INFERENCE_LOGGER_NAME):
        assert validate_checkpoint_exists(str(small_file)) is True
    assert any("Checkpoint file found:" in rec.message and small_file.name in rec.message
               for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME and rec.levelno == logging.INFO)
    assert any("Checkpoint file seems very small" in rec.message
               for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME and rec.levelno == logging.WARNING)
    caplog.clear()
    large_file = tmp_path / "large.pt"
    content_large = b"data" * 1024 * 110
    large_file.write_bytes(content_large)
    with caplog.at_level(logging.INFO, logger=INFERENCE_LOGGER_NAME):
        assert validate_checkpoint_exists(str(large_file)) is True
    assert any("Checkpoint file found:" in rec.message and large_file.name in rec.message
               for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME and rec.levelno == logging.INFO)
    assert not any("Checkpoint file seems very small" in rec.message
                   for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME and rec.levelno == logging.WARNING)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=INFERENCE_LOGGER_NAME):
        assert validate_checkpoint_exists("nonexistent.pt") is False
    assert any("Checkpoint file not found:" in rec.message and "nonexistent.pt" in rec.message
               for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME and rec.levelno == logging.ERROR)

def test_format_code_output():
    python_code = "def hello():\n  print('world')"
    formatted_python = format_code_output(python_code, "python")
    assert isinstance(formatted_python, Syntax)
    if formatted_python.lexer: assert "python" in str(type(formatted_python.lexer)).lower()
    text_content = "Just some plain text."
    formatted_text = format_code_output(text_content, "text")
    assert isinstance(formatted_text, Syntax)
    if formatted_text.lexer: assert "text" in str(type(formatted_text.lexer)).lower()
    formatted_auto = format_code_output(python_code, "auto")
    assert isinstance(formatted_auto, Syntax)
    if formatted_auto.lexer: assert "text" in str(type(formatted_auto.lexer)).lower()
    formatted_invalid_lang = format_code_output(python_code, "invalidlang!!")
    assert isinstance(formatted_invalid_lang, (Syntax, RichText))
    if isinstance(formatted_invalid_lang, Syntax) and formatted_invalid_lang.lexer:
        assert "text" in str(type(formatted_invalid_lang.lexer)).lower()

def test_load_model_from_checkpoint_success(dummy_checkpoint_path, dummy_config_inf, caplog):
    device = torch.device("cpu")
    model_logger = logging.getLogger(MODEL_LOGGER_NAME)
    original_level = model_logger.getEffectiveLevel()
    model_logger.setLevel(logging.CRITICAL + 1)
    with caplog.at_level(logging.INFO, logger=INFERENCE_LOGGER_NAME):
        load_result = load_model_from_checkpoint(dummy_checkpoint_path, device)
    model_logger.setLevel(original_level)
    assert load_result is not None
    model, config_loaded, train_args_loaded = load_result
    assert isinstance(model, LunarisMind)
    assert isinstance(config_loaded, LunarisCodexConfig)
    assert config_loaded.vocab_size == dummy_config_inf.vocab_size
    assert config_loaded.d_model == dummy_config_inf.d_model
    assert config_loaded.pad_token_id == dummy_config_inf.pad_token_id
    log_messages = [rec.getMessage() for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME]
    assert any("Model configuration loaded successfully" in msg for msg in log_messages)
    assert any("Initial memory usage after model load" in msg for msg in log_messages)
    assert "tokenizer_name_or_path" in train_args_loaded
    assert train_args_loaded.get("tokenizer_name_or_path") == "gpt2"

def test_load_model_from_checkpoint_sha_mismatch_returns_none(dummy_checkpoint_path, caplog):
    device = torch.device("cpu")
    hash_file_path = Path(dummy_checkpoint_path + ".sha256")
    original_hash_content = hash_file_path.read_text()
    checkpoint_basename = Path(dummy_checkpoint_path).name
    hash_file_path.write_text(f"mismatch_hash  {checkpoint_basename}\n")
    with caplog.at_level(logging.ERROR, logger=INFERENCE_LOGGER_NAME):
        result = load_model_from_checkpoint(dummy_checkpoint_path, device)
    assert result is None
    assert any("Aborting due to checkpoint integrity verification failure" in rec.getMessage()
               for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME)
    hash_file_path.write_text(original_hash_content)

def test_load_model_from_checkpoint_missing_config_key_exits(tmp_path, caplog):
    invalid_ckpt_path = tmp_path / "invalid_config.pt"
    dummy_state_dict = {"some_param": torch.tensor(1.0)}
    incomplete_config = {"d_model": 16, "n_layers": 1, "n_heads": 1, "max_seq_len": 10, "pad_token_id": 0}
    torch.save({"config_args": incomplete_config, "model_state_dict": dummy_state_dict}, str(invalid_ckpt_path))
    device = torch.device("cpu")
    model_logger = logging.getLogger(MODEL_LOGGER_NAME)
    original_level = model_logger.getEffectiveLevel()
    model_logger.setLevel(logging.CRITICAL + 1)
    with caplog.at_level(logging.ERROR, logger=INFERENCE_LOGGER_NAME):
        result = load_model_from_checkpoint(str(invalid_ckpt_path), device)
    model_logger.setLevel(original_level)
    assert result is None
    assert any("config (key: 'config_args') is missing required field: 'vocab_size'" in rec.getMessage()
               for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME)

def test_load_model_from_checkpoint_missing_state_dict_exits(tmp_path, caplog):
    invalid_ckpt_path = tmp_path / "invalid_state.pt"
    valid_config = {"vocab_size":100, "d_model":16, "n_layers":1, "n_heads":2, "max_seq_len":32, "pad_token_id":0}
    torch.save({"config_args": valid_config}, str(invalid_ckpt_path))
    device = torch.device("cpu")
    model_logger = logging.getLogger(MODEL_LOGGER_NAME)
    original_level = model_logger.getEffectiveLevel()
    model_logger.setLevel(logging.CRITICAL + 1)
    with caplog.at_level(logging.ERROR, logger=INFERENCE_LOGGER_NAME):
        result = load_model_from_checkpoint(str(invalid_ckpt_path), device)
    model_logger.setLevel(original_level)
    assert result is None
    assert any("Checkpoint is missing 'model_state_dict'." in rec.getMessage()
               for rec in caplog.records if rec.name == INFERENCE_LOGGER_NAME)
