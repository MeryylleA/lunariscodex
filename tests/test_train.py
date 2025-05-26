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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from train import (
    set_seed,
    compute_sha256 as train_compute_sha256,
    verify_checkpoint_integrity as train_verify_checkpoint_integrity,
    compute_metrics,
    save_checkpoint,
    load_checkpoint
)
from model import LunarisCodexConfig, LunarisMind

TRAIN_UTILS_LOGGER_NAME = "train"
MODEL_LOGGER_NAME = "lunaris_model"

@pytest.fixture
def dummy_config():
    return LunarisCodexConfig(
        vocab_size=100, d_model=16, n_layers=1, n_heads=2,
        max_seq_len=32, ff_multiplier=2, dropout=0.0,
        pad_token_id=0
    )

@pytest.fixture
def dummy_model(dummy_config):
    model_logger = logging.getLogger(MODEL_LOGGER_NAME)
    original_level = model_logger.getEffectiveLevel()
    model_logger.setLevel(logging.CRITICAL + 1)
    model = LunarisMind(dummy_config)
    model_logger.setLevel(original_level)
    return model

@pytest.fixture
def dummy_optimizer(dummy_model):
    return torch.optim.AdamW(dummy_model.parameters(), lr=1e-4)

@pytest.fixture
def dummy_args_for_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints_test_train"
    return Namespace(
        checkpoint_dir=str(checkpoint_dir),
        resume_from_checkpoint=None,
        lora_rank=0,
        tokenizer_name_or_path="gpt2_dummy_for_train_test",
        learning_rate=1e-4,
        batch_size=1
    )

def test_set_seed():
    seed1 = 42; set_seed(seed1)
    r1p, r1n, r1t = random.random(), np.random.rand(), torch.rand(1).item()
    set_seed(seed1); r2p, r2n, r2t = random.random(), np.random.rand(), torch.rand(1).item()
    assert (r1p, r1n, r1t) == (r2p, r2n, r2t)
    seed2 = 123; set_seed(seed2); r3p = random.random()
    assert r1p != r3p

def test_compute_sha256_valid_file(tmp_path):
    p = tmp_path / "test_file_train.txt"; content = b"Train SHA256 Test"
    p.write_bytes(content)
    assert train_compute_sha256(str(p)) == hashlib.sha256(content).hexdigest()

def test_compute_sha256_nonexistent_file(caplog):
    with caplog.at_level(logging.WARNING, logger=TRAIN_UTILS_LOGGER_NAME):
        assert train_compute_sha256("nonexistent_file.txt") is None
    assert any("Failed to compute SHA-256 for nonexistent_file.txt" in rec.message for rec in caplog.records)

def test_verify_checkpoint_integrity(tmp_path, caplog):
    ckpt_file = tmp_path / "model.pt"; ckpt_content = b"dummy data for train test"
    ckpt_file.write_bytes(ckpt_content)
    hash_file = tmp_path / "model.pt.sha256"
    correct_hash = hashlib.sha256(ckpt_content).hexdigest()
    base_name = ckpt_file.name

    hash_file.write_text(f"{correct_hash}  {base_name}\n")
    with caplog.at_level(logging.INFO, logger=TRAIN_UTILS_LOGGER_NAME):
        assert train_verify_checkpoint_integrity(str(ckpt_file)) is True
    assert any(f"Checkpoint integrity verified: {str(ckpt_file)}" in rec.message
               for rec in caplog.records if rec.name == TRAIN_UTILS_LOGGER_NAME)
    caplog.clear()

    hash_file.write_text(f"bad_hash_123  {base_name}\n")
    with caplog.at_level(logging.ERROR, logger=TRAIN_UTILS_LOGGER_NAME):
        assert train_verify_checkpoint_integrity(str(ckpt_file)) is False
    assert any(f"Checkpoint integrity check FAILED: {str(ckpt_file)}" in rec.getMessage() and "Expected bad_hash_123" in rec.getMessage()
               for rec in caplog.records if rec.name == TRAIN_UTILS_LOGGER_NAME and rec.levelno == logging.ERROR)
    caplog.clear()

    hash_file.unlink()
    with caplog.at_level(logging.WARNING, logger=TRAIN_UTILS_LOGGER_NAME):
        assert train_verify_checkpoint_integrity(str(ckpt_file)) is True
    assert any(f"No hash file found for {str(ckpt_file)}" in rec.message
               for rec in caplog.records if rec.name == TRAIN_UTILS_LOGGER_NAME)
    caplog.clear()

    hash_file.write_text("")
    with caplog.at_level(logging.WARNING, logger=TRAIN_UTILS_LOGGER_NAME):
        assert train_verify_checkpoint_integrity(str(ckpt_file)) is True
    assert any(f"Could not verify checkpoint integrity for {str(ckpt_file)}" in rec.message and "list index out of range" in rec.message
               for rec in caplog.records if rec.name == TRAIN_UTILS_LOGGER_NAME and rec.levelno == logging.WARNING)

def test_compute_metrics_basic():
    V, S, B = 10, 5, 2
    logits1 = torch.rand(B, S, V) * 0.1; targets1 = torch.randint(0,V,(B,S))
    for b in range(B):
        for s in range(S-1): logits1[b,s,targets1[b,s+1]] += 10
    mask1 = torch.ones(B,S,dtype=torch.long)
    loss1,_,acc1 = compute_metrics(logits1,targets1,mask1)
    assert loss1.item() < 0.01 and acc1.item() == 1.0

    logits2 = torch.rand(B,S,V)*0.1; targets2 = torch.randint(0,V,(B,S))
    mask2 = torch.ones(B,S,dtype=torch.long); mask2[0,3:]=0; mask2[1,2:]=0
    for b in range(B):
        for s in range(S-1):
            if mask2[b,s+1]==1: logits2[b,s,targets2[b,s+1]] += 10
    loss2,_,acc2 = compute_metrics(logits2,targets2,mask2)
    assert loss2.item() < 0.01 and acc2.item() == 1.0

    mask3 = torch.zeros(B,S,dtype=torch.long)
    loss3,ppl3,acc3 = compute_metrics(torch.rand(B,S,V), torch.randint(0,V,(B,S)), mask3)
    assert loss3.item()==0.0 and torch.isinf(ppl3) and acc3.item()==0.0

def test_save_and_load_checkpoint_cycle(dummy_model, dummy_optimizer, dummy_args_for_checkpoint, caplog):
    args = dummy_args_for_checkpoint
    epoch_saved, step_saved, loss_saved = 0, 100, 0.5
    model_logger = logging.getLogger(MODEL_LOGGER_NAME)
    train_utils_logger = logging.getLogger(TRAIN_UTILS_LOGGER_NAME) # Not used directly but good to define
    m_orig_lvl = model_logger.getEffectiveLevel()
    model_logger.setLevel(logging.CRITICAL + 1)

    with caplog.at_level(logging.INFO, logger=TRAIN_UTILS_LOGGER_NAME): # Capture from train logger
        save_checkpoint(dummy_model, dummy_optimizer, epoch_saved, step_saved, loss_saved, args, is_best=True)

    save_logs = [rec.getMessage() for rec in caplog.records if rec.name == TRAIN_UTILS_LOGGER_NAME and rec.levelno >= logging.INFO]
    ckpt_dir = Path(args.checkpoint_dir)
    expected_fname = f"lunaris_codex_epoch-{epoch_saved+1}_step-{step_saved}.pt"
    saved_path = ckpt_dir / expected_fname
    best_path = ckpt_dir / "best_model.pt"
    assert any(f"Checkpoint saved: {str(saved_path)}" in log for log in save_logs)
    assert any(f"Best checkpoint saved: {str(best_path)}" in log for log in save_logs)
    caplog.clear()

    model_logger.setLevel(m_orig_lvl) # Restore after save

    assert saved_path.exists() and (ckpt_dir / f"{expected_fname}.sha256").exists()
    assert best_path.exists() and (ckpt_dir / "best_model.pt.sha256").exists()

    new_model = LunarisMind(dummy_model.config)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=dummy_optimizer.defaults['lr'] * 0.1)
    orig_model_state = {k: v.clone() for k,v in dummy_model.state_dict().items()}
    args_load = Namespace(resume_from_checkpoint=str(saved_path), checkpoint_dir=str(ckpt_dir), lora_rank=args.lora_rank)
    device = torch.device('cpu')

    model_logger.setLevel(logging.CRITICAL + 1) # Suppress for new_model init if any logs
    with caplog.at_level(logging.INFO, logger=TRAIN_UTILS_LOGGER_NAME):
        res_epoch, res_step, res_loss = load_checkpoint(new_model, new_optimizer, args_load, device)
    model_logger.setLevel(m_orig_lvl)

    load_logs = [rec.getMessage() for rec in caplog.records if rec.name == TRAIN_UTILS_LOGGER_NAME]
    assert any(f"Loading checkpoint: {str(saved_path)}" in log for log in load_logs)
    assert any("Optimizer state loaded" in log for log in load_logs)
    assert res_epoch == epoch_saved + 1
    assert res_step == step_saved
    assert abs(res_loss - loss_saved) < 1e-6
    for k in orig_model_state: assert torch.equal(new_model.state_dict()[k], orig_model_state[k])
    if new_optimizer.param_groups: assert abs(new_optimizer.param_groups[0]['lr'] - dummy_optimizer.param_groups[0]['lr']) < 1e-7
    caplog.clear()

    args_load_best = Namespace(resume_from_checkpoint=None, checkpoint_dir=str(ckpt_dir), lora_rank=args.lora_rank)
    model_best = LunarisMind(dummy_model.config)
    opt_best = torch.optim.AdamW(model_best.parameters(), lr=1e-3)
    model_logger.setLevel(logging.CRITICAL + 1)
    with caplog.at_level(logging.INFO, logger=TRAIN_UTILS_LOGGER_NAME):
        _, _, loss_best_loaded = load_checkpoint(model_best, opt_best, args_load_best, device)
    model_logger.setLevel(m_orig_lvl)
    best_load_logs = [rec.getMessage() for rec in caplog.records if rec.name == TRAIN_UTILS_LOGGER_NAME]
    assert any("Found 'best_model.pt'" in log for log in best_load_logs)
    assert any(f"Loading checkpoint: {str(best_path)}" in log for log in best_load_logs)
    assert abs(loss_best_loaded - loss_saved) < 1e-6
    for k in orig_model_state: assert torch.equal(model_best.state_dict()[k], orig_model_state[k])
