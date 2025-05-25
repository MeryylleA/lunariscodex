import pytest
import os
import glob
import numpy as np
from transformers import AutoTokenizer, GPT2TokenizerFast
import tempfile
import logging
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prepare_data import (
    get_tokenizer,
    yield_examples_from_text_files_lines,
    yield_examples_from_text_file_chunks,
    process_and_save_dataset
)

LOGGER_NAME = "prepare_data"

@pytest.fixture
def gpt2_tokenizer() -> GPT2TokenizerFast:
    return get_tokenizer("gpt2", trust_remote_code_flag=False)

@pytest.fixture
def temp_text_file_lines(tmp_path):
    file_content = "Primeira linha.\n\nSegunda linha.\nTerceira linha.\n  \nQuarta linha com espacos.  \n"
    file_path = tmp_path / "sample_lines.txt"
    file_path.write_text(file_content, encoding="utf-8")
    return str(file_path)

@pytest.fixture
def temp_text_file_for_chunks(tmp_path):
    file_content = "Este é um texto de exemplo um pouco mais longo para testar a funcionalidade de chunking. " * 10
    file_content += "Ele será dividido em vários pedaços pelo tokenizer. " * 10
    file_content += "Esperamos que cada pedaço tenha o tamanho máximo definido ou menos, se for o último." * 10
    file_path = tmp_path / "sample_chunks.txt"
    file_path.write_text(file_content, encoding="utf-8")
    return str(file_path)

def test_get_tokenizer_gpt2_basic(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        tokenizer = get_tokenizer("gpt2", trust_remote_code_flag=False)
    assert tokenizer is not None
    assert isinstance(tokenizer, GPT2TokenizerFast)
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.pad_token == tokenizer.eos_token
    assert "Loading tokenizer from: gpt2" in caplog.text
    assert "Effective Pad token:" in caplog.text
    caplog.clear()

def test_get_tokenizer_missing_pad_adds_new(caplog):
    pass

def test_get_tokenizer_uses_eos_if_pad_missing(gpt2_tokenizer, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        current_tokenizer = get_tokenizer("gpt2", trust_remote_code_flag=False)
    assert f"Effective Pad token: '{current_tokenizer.eos_token}' (ID: {current_tokenizer.eos_token_id})" in caplog.text
    caplog.clear()

def test_yield_examples_from_text_files_lines_single_file(temp_text_file_lines):
    iterator = yield_examples_from_text_files_lines(temp_text_file_lines)
    examples = list(iterator)
    assert len(examples) == 4
    assert examples[0] == "Primeira linha."
    assert examples[1] == "Segunda linha."
    assert examples[2] == "Terceira linha."
    assert examples[3] == "Quarta linha com espacos."

def test_yield_examples_from_text_files_lines_glob_pattern(tmp_path):
    file1 = tmp_path / "file1.txt"
    file1.write_text("Linha do arquivo 1\nOutra linha file1")
    file2 = tmp_path / "file2.txt"
    file2.write_text("Linha do arquivo 2")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.txt"
    file3.write_text("Linha do subdir")

    pattern = str(tmp_path / "*.txt")
    iterator = yield_examples_from_text_files_lines(pattern)
    examples = list(iterator)
    assert len(examples) == 3
    assert "Linha do arquivo 1" in examples
    assert "Outra linha file1" in examples
    assert "Linha do arquivo 2" in examples
    assert "Linha do subdir" not in examples

    pattern_recursive = str(tmp_path / "**/*.txt")
    iterator_recursive = yield_examples_from_text_files_lines(pattern_recursive)
    examples_recursive = list(iterator_recursive)
    assert len(examples_recursive) == 4
    assert "Linha do subdir" in examples_recursive

def test_yield_examples_from_text_files_lines_no_files_found(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        iterator = yield_examples_from_text_files_lines("./non_existent_path/*.txt")
        examples = list(iterator)
    assert len(examples) == 0
    assert "No files found matching pattern: ./non_existent_path/*.txt" in caplog.text
    caplog.clear()

def test_yield_examples_from_text_files_lines_empty_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    iterator = yield_examples_from_text_files_lines(str(empty_file))
    examples = list(iterator)
    assert len(examples) == 0

def test_yield_examples_from_text_file_chunks_basic(temp_text_file_for_chunks, gpt2_tokenizer):
    max_length = 50
    iterator = yield_examples_from_text_file_chunks(temp_text_file_for_chunks, gpt2_tokenizer, max_length)
    chunks = list(iterator)

    assert len(chunks) > 1

    original_text = Path(temp_text_file_for_chunks).read_text(encoding="utf-8")
    reconstructed_text_from_chunks = "".join(chunks)

    original_tokens = gpt2_tokenizer.encode(original_text, add_special_tokens=False)
    reconstructed_tokens = gpt2_tokenizer.encode(reconstructed_text_from_chunks, add_special_tokens=False)

    assert abs(len(original_tokens) - len(reconstructed_tokens)) < max_length * 2

    for i, chunk_text in enumerate(chunks):
        chunk_tokens = gpt2_tokenizer.encode(chunk_text, add_special_tokens=False)
        assert len(chunk_tokens) <= max_length

def test_yield_examples_from_text_file_chunks_empty_file(tmp_path, gpt2_tokenizer, caplog):
    empty_file_path = tmp_path / "empty_for_chunks.txt"
    empty_file_path.write_text("   \n\t   ")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        iterator = yield_examples_from_text_file_chunks(str(empty_file_path), gpt2_tokenizer, 50)
        chunks = list(iterator)
    assert len(chunks) == 0
    assert f"File {str(empty_file_path)} is empty or contains only whitespace. No chunks to yield." in caplog.text
    caplog.clear()

def test_yield_examples_from_text_file_chunks_file_not_found(gpt2_tokenizer, caplog):
    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        iterator = yield_examples_from_text_file_chunks("non_existent_file.txt", gpt2_tokenizer, 50)
        chunks = list(iterator)
    assert len(chunks) == 0
    assert "File not found for chunking: non_existent_file.txt" in caplog.text
    caplog.clear()

@pytest.fixture
def dummy_example_iterator():
    return iter([
        "Este é o primeiro exemplo de texto.",
        "Um segundo exemplo, um pouco mais longo para testar o padding.",
        "Exemplo três.",
        "Quarto exemplo com muitos tokens para testar truncamento " * 10,
        "",
        "Quinto exemplo."
    ])

def test_process_and_save_dataset_basic(dummy_example_iterator, gpt2_tokenizer, tmp_path, caplog):
    output_file = tmp_path / "test_dataset.memmap"
    max_length = 20
    max_examples = 3

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        num_saved = process_and_save_dataset(
            dummy_example_iterator,
            gpt2_tokenizer,
            str(output_file),
            max_length=max_length,
            max_examples=max_examples,
            add_special_tokens=False,
            dtype_str="int32",
            overwrite_output=True
        )

    assert num_saved == max_examples
    assert output_file.exists()

    log_text = "".join(record.message for record in caplog.get_records(when='call'))
    assert f"Successfully tokenized {max_examples} valid examples." in log_text
    assert f"Successfully saved {max_examples} sequences to {str(output_file)}" in log_text

    loaded_memmap = np.memmap(str(output_file), dtype=np.int32, mode='r', shape=(num_saved, max_length))

    tokens1 = gpt2_tokenizer.encode("Este é o primeiro exemplo de texto.", add_special_tokens=False)
    expected_padded1 = tokens1[:max_length] + [gpt2_tokenizer.pad_token_id] * (max_length - len(tokens1[:max_length]))
    assert np.array_equal(loaded_memmap[0], expected_padded1)

    tokens2 = gpt2_tokenizer.encode("Um segundo exemplo, um pouco mais longo para testar o padding.", add_special_tokens=False)
    expected_padded2 = tokens2[:max_length] + [gpt2_tokenizer.pad_token_id] * (max_length - len(tokens2[:max_length]))
    assert np.array_equal(loaded_memmap[1], expected_padded2)

    tokens3 = gpt2_tokenizer.encode("Exemplo três.", add_special_tokens=False)
    expected_padded3 = tokens3[:max_length] + [gpt2_tokenizer.pad_token_id] * (max_length - len(tokens3[:max_length]))
    assert np.array_equal(loaded_memmap[2], expected_padded3)

    del loaded_memmap
    caplog.clear()

def test_process_and_save_dataset_overwrite_false_raises_error(dummy_example_iterator, gpt2_tokenizer, tmp_path):
    output_file = tmp_path / "overwrite_test.memmap"
    output_file.touch()

    with pytest.raises(FileExistsError):
        process_and_save_dataset(
            dummy_example_iterator, gpt2_tokenizer, str(output_file),
            max_length=10, overwrite_output=False
        )

def test_process_and_save_dataset_overwrite_true_works(dummy_example_iterator, gpt2_tokenizer, tmp_path):
    output_file = tmp_path / "overwrite_true.memmap"
    output_file.write_text("dummy content")

    num_saved = process_and_save_dataset(
        dummy_example_iterator, gpt2_tokenizer, str(output_file),
        max_length=10, max_examples=1, overwrite_output=True
    )
    assert num_saved == 1
    assert output_file.exists()
    assert output_file.stat().st_size == 1*10*np.dtype(np.int32).itemsize

def test_process_and_save_dataset_int16_dtype(dummy_example_iterator, gpt2_tokenizer, tmp_path, caplog):
    output_file = tmp_path / "int16_test.memmap"
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
      with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        if gpt2_tokenizer.vocab_size > 32767:
            process_and_save_dataset(
                dummy_example_iterator, gpt2_tokenizer, str(output_file),
                max_length=10, max_examples=1, dtype_str="int16", overwrite_output=True
            )
            log_text = "".join(record.message for record in caplog.get_records(when='call'))
            assert "Requested int16 dtype, but tokenizer vocab size" in log_text
            assert "Using int32 instead" in log_text
            loaded_memmap = np.memmap(str(output_file), dtype=np.int32, mode='r')
            assert loaded_memmap.dtype == np.int32
            del loaded_memmap
        else:
            pytest.skip("Skipping int16 specific test as gpt2_tokenizer vocab is too small for direct int16.")
    caplog.clear()

def test_process_and_save_dataset_no_valid_examples(gpt2_tokenizer, tmp_path, caplog):
    empty_iterator = iter(["", "   ", "\n"])
    output_file = tmp_path / "no_valid.memmap"

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        num_saved = process_and_save_dataset(
            empty_iterator, gpt2_tokenizer, str(output_file),
            max_length=10, overwrite_output=True
        )
    assert num_saved == 0
    assert "No valid examples were processed or tokenized." in caplog.text
    assert not output_file.exists()
    caplog.clear()
