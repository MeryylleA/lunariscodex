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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from train import (
    set_seed,
    compute_sha256,
    verify_checkpoint_integrity,
    compute_metrics,
    save_checkpoint,
    load_checkpoint
)
from model import LunarisCodexConfig, LunarisMind

TRAIN_LOGGER_NAME = "train"

@pytest.fixture
def dummy_config():
    return LunarisCodexConfig(
        vocab_size=100,
        d_model=16,
        n_layers=1,
        n_heads=2,
        max_seq_len=32,
        ff_multiplier=2,
        dropout=0.0
    )

@pytest.fixture
def dummy_model(dummy_config):
    return LunarisMind(dummy_config)

@pytest.fixture
def dummy_optimizer(dummy_model):
    return torch.optim.AdamW(dummy_model.parameters(), lr=1e-4)

@pytest.fixture
def dummy_args_for_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints_test"
    args = Namespace(
        checkpoint_dir=str(checkpoint_dir),
        resume_from_checkpoint=None,
        lora_rank=0,
        device='cpu',
        learning_rate=1e-4
    )
    return args

def test_set_seed():
    seed1_val = 42
    set_seed(seed1_val)
    rand1_py = random.random()
    rand1_np = np.random.rand()
    rand1_torch = torch.rand(1).item()

    set_seed(seed1_val)
    rand2_py = random.random()
    rand2_np = np.random.rand()
    rand2_torch = torch.rand(1).item()

    assert rand1_py == rand2_py
    assert rand1_np == rand2_np
    assert rand1_torch == rand2_torch

    seed2_val = 123
    set_seed(seed2_val)
    rand3_py = random.random()
    assert rand1_py != rand3_py

def test_compute_sha256_valid_file(tmp_path):
    p = tmp_path / "test_file.txt"
    content = b"Hello SHA256 Test"
    p.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    assert compute_sha256(str(p)) == expected_hash

def test_compute_sha256_nonexistent_file(caplog):
    with caplog.at_level(logging.WARNING, logger=TRAIN_LOGGER_NAME):
        assert compute_sha256("nonexistent_file.txt") is None
    assert "Failed to compute SHA-256 for nonexistent_file.txt" in caplog.text

def test_verify_checkpoint_integrity(tmp_path, caplog):
    checkpoint_file = tmp_path / "model.pt"
    checkpoint_content = b"dummy model data"
    checkpoint_file.write_bytes(checkpoint_content)

    hash_file = tmp_path / "model.pt.sha256"
    correct_hash = hashlib.sha256(checkpoint_content).hexdigest()

    hash_file.write_text(f"{correct_hash}  model.pt\n")
    with caplog.at_level(logging.INFO, logger=TRAIN_LOGGER_NAME):
        assert verify_checkpoint_integrity(str(checkpoint_file)) == True
        assert f"Checkpoint integrity verified: {str(checkpoint_file)}" in caplog.text
    caplog.clear()

    hash_file.write_text(f"incorrecthash123  model.pt\n")
    with caplog.at_level(logging.ERROR, logger=TRAIN_LOGGER_NAME):
        assert verify_checkpoint_integrity(str(checkpoint_file)) == False
    assert f"Checkpoint integrity check failed: {str(checkpoint_file)}" in caplog.text
    caplog.clear()

    hash_file.unlink()
    with caplog.at_level(logging.WARNING, logger=TRAIN_LOGGER_NAME):
        assert verify_checkpoint_integrity(str(checkpoint_file)) == True
        assert f"No hash file found for {str(checkpoint_file)}" in caplog.text
    caplog.clear()

    hash_file.write_text("")
    with caplog.at_level(logging.WARNING, logger=TRAIN_LOGGER_NAME):
        assert verify_checkpoint_integrity(str(checkpoint_file)) == True
        assert "Could not verify checkpoint integrity" in caplog.text
        assert "list index out of range" in caplog.text
    caplog.clear()

def test_compute_metrics_basic():
    vocab_size = 10
    seq_len = 5
    batch_size = 2

    logits1 = torch.randn(batch_size, seq_len, vocab_size)
    targets1 = torch.randint(0, vocab_size, (batch_size, seq_len))
    for b in range(batch_size):
        for s in range(seq_len -1):
             logits1[b, s, targets1[b, s+1]] = 10.0
    attention_mask1 = torch.ones(batch_size, seq_len, dtype=torch.long)

    loss1, ppl1, acc1 = compute_metrics(logits1, targets1, attention_mask1)
    assert loss1 < 0.1
    assert acc1 == 1.0

    logits2 = torch.randn(batch_size, seq_len, vocab_size)
    targets2 = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask2 = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask2[0, 3:] = 0
    attention_mask2[1, 2:] = 0

    for b in range(batch_size):
        for s in range(seq_len -1):
            if attention_mask2[b, s+1] == 1:
                 logits2[b, s, targets2[b, s+1]] = 10.0

    loss2, ppl2, acc2 = compute_metrics(logits2, targets2, attention_mask2)
    assert acc2 == 1.0
    assert loss2 < 0.1

    logits3 = torch.randn(batch_size, seq_len, vocab_size)
    targets3 = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask3 = torch.zeros(batch_size, seq_len, dtype=torch.long)

    loss3, ppl3, acc3 = compute_metrics(logits3, targets3, attention_mask3)
    assert loss3 == 0.0
    assert torch.isinf(ppl3)
    assert acc3 == 0.0

def test_save_and_load_checkpoint_cycle(dummy_model, dummy_optimizer, dummy_args_for_checkpoint, tmp_path, caplog):
    args = dummy_args_for_checkpoint
    epoch = 0
    step = 100
    current_loss = 0.5

    with caplog.at_level(logging.INFO, logger=TRAIN_LOGGER_NAME):
        save_checkpoint(dummy_model, dummy_optimizer, epoch, step, current_loss, args, is_best=True)

    saved_checkpoint_dir = Path(args.checkpoint_dir)
    expected_checkpoint_filename = f"lunaris_codex_epoch-{epoch+1}_step-{step}.pt"
    saved_checkpoint_path = saved_checkpoint_dir / expected_checkpoint_filename
    saved_best_checkpoint_path = saved_checkpoint_dir / "best_model.pt"

    assert saved_checkpoint_path.exists()
    assert (saved_checkpoint_dir / f"{expected_checkpoint_filename}.sha256").exists()
    assert saved_best_checkpoint_path.exists()
    assert (saved_checkpoint_dir / "best_model.pt.sha256").exists()

    log_text = "".join(r.message for r in caplog.records if r.name == TRAIN_LOGGER_NAME and r.levelno >= logging.INFO)
    assert f"Checkpoint saved: {str(saved_checkpoint_path)}" in log_text
    assert f"Best checkpoint saved: {str(saved_best_checkpoint_path)}" in log_text
    caplog.clear()

    new_model = LunarisMind(dummy_model.config)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-5)

    original_model_state = {k: v.clone() for k, v in dummy_model.state_dict().items()}

    args_load = Namespace(
        resume_from_checkpoint=str(saved_checkpoint_path),
        checkpoint_dir=str(saved_checkpoint_dir),
        lora_rank=args.lora_rank,
        device=args.device
    )

    with caplog.at_level(logging.INFO, logger=TRAIN_LOGGER_NAME):
        start_epoch, start_step, min_val_loss = load_checkpoint(
            new_model, new_optimizer, args_load, device=torch.device(args_load.device)
        )

    log_text_load = "".join(r.message for r in caplog.records if r.name == TRAIN_LOGGER_NAME and r.levelno >= logging.INFO)
    assert f"Loading checkpoint: {str(saved_checkpoint_path)}" in log_text_load
    assert "Optimizer state loaded" in log_text_load

    assert start_epoch == epoch
    assert start_step == step
    assert abs(min_val_loss - current_loss) < 1e-6

    for key in original_model_state:
        assert torch.equal(new_model.state_dict()[key], original_model_state[key])

    if new_optimizer.param_groups:
         assert abs(new_optimizer.param_groups[0]['lr'] - dummy_optimizer.param_groups[0]['lr']) < 1e-7
    caplog.clear()

    args_load_best = Namespace(
        resume_from_checkpoint=None,
        checkpoint_dir=str(saved_checkpoint_dir),
        lora_rank=args.lora_rank,
        device=args.device
    )
    model_for_best = LunarisMind(dummy_model.config)
    optimizer_for_best = torch.optim.AdamW(model_for_best.parameters(), lr=1e-3)
    with caplog.at_level(logging.INFO, logger=TRAIN_LOGGER_NAME):
        s_epoch, s_step, s_loss = load_checkpoint(model_for_best, optimizer_for_best, args_load_best, device=torch.device(args_load_best.device))

    log_text_best = "".join(r.message for r in caplog.records if r.name == TRAIN_LOGGER_NAME and r.levelno >= logging.INFO)
    assert "Loading best_model.pt" in log_text_best
    for key in original_model_state:
        assert torch.equal(model_for_best.state_dict()[key], original_model_state[key])
    assert abs(s_loss - current_loss) < 1e-6
