# prepare_data.py
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SCRIPT VERSION ---
SCRIPT_VERSION = "0.3.0" # Updated version for line-buffered chunking feature

def get_tokenizer(tokenizer_name_or_path: str, trust_remote_code_flag: bool = True) -> AutoTokenizer:
    """
    Loads a tokenizer using AutoTokenizer from Hugging Face.
    Handles common cases for setting pad_token_id if not already defined.
    Also logs detailed information about the loaded tokenizer.
    """
    logger.info(f"Loading tokenizer from: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=trust_remote_code_flag)

    # --- Enhanced Tokenizer Logging ---
    logger.info(f"--- Tokenizer Info ({type(tokenizer).__name__}) ---")
    logger.info(f"  Name or Path: {tokenizer.name_or_path}")
    logger.info(f"  Vocabulary Size (len(tokenizer)): {len(tokenizer)}") # More accurate for actual vocab size
    logger.info(f"  Model Max Length: {tokenizer.model_max_length}")

    # Log special token IDs and their string representations
    special_tokens_info = {
        "PAD": {"id": tokenizer.pad_token_id, "token": tokenizer.pad_token},
        "EOS": {"id": tokenizer.eos_token_id, "token": tokenizer.eos_token},
        "BOS": {"id": tokenizer.bos_token_id, "token": tokenizer.bos_token},
        "UNK": {"id": tokenizer.unk_token_id, "token": tokenizer.unk_token},
        "SEP": {"id": tokenizer.sep_token_id, "token": tokenizer.sep_token},
        "CLS": {"id": tokenizer.cls_token_id, "token": tokenizer.cls_token},
        "MASK": {"id": tokenizer.mask_token_id, "token": tokenizer.mask_token},
    }
    for name, info in special_tokens_info.items():
        if info["id"] is not None or info["token"] is not None: # Log if either ID or token string exists
             logger.info(f"  {name} token: '{info['token']}' (ID: {info['id']})")
    # --- End of Enhanced Tokenizer Logging ---

    if tokenizer.pad_token_id is None:
        original_vocab_size_before_pad_add = len(tokenizer) # Get vocab size before potential additions
        pad_token_source_info = ""

        if tokenizer.eos_token_id is not None:
            logger.info(f"Tokenizer lacks a pad_token_id. Attempting to use eos_token_id ({tokenizer.eos_token_id}) as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            pad_token_source_info = f"eos_token ('{tokenizer.eos_token}')"
        elif tokenizer.bos_token_id is not None:
            logger.info(f"Tokenizer lacks a pad_token_id and eos_token_id. Attempting to use bos_token_id ({tokenizer.bos_token_id}) as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.bos_token_id
            if tokenizer.pad_token is None and tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
            pad_token_source_info = f"bos_token ('{tokenizer.bos_token}')"
        else:
            new_pad_token_str = '<|PAD|>'
            logger.warning(f"Tokenizer lacked pad, eos, or bos token. Adding new special pad_token: '{new_pad_token_str}'.")
            tokenizer.add_special_tokens({'pad_token': new_pad_token_str})
            pad_token_source_info = f"newly added token '{new_pad_token_str}'"
            # Log only if vocab size actually changed
            if len(tokenizer) != original_vocab_size_before_pad_add:
                logger.info(f"  New PAD token ID {tokenizer.pad_token_id} assigned. Vocab size changed from {original_vocab_size_before_pad_add} to {len(tokenizer)}.")
            else:
                logger.info(f"  New PAD token ID {tokenizer.pad_token_id} assigned (vocab size unchanged).")


        # This block attempts to ensure tokenizer.pad_token (string) is set if pad_token_id was set from eos/bos
        if tokenizer.pad_token is None and tokenizer.pad_token_id is not None:
            logger.warning(f"Pad token string was not automatically set from {pad_token_source_info} for pad_token_id {tokenizer.pad_token_id}. "
                           f"Attempting to decode pad_token_id or using a placeholder.")
            decoded_pad_token = tokenizer.decode([tokenizer.pad_token_id], skip_special_tokens=False)
            if decoded_pad_token and decoded_pad_token.strip(): # Ensure it's a meaningful string
                tokenizer.pad_token = decoded_pad_token
                logger.info(f"  Inferred pad_token string as '{decoded_pad_token}' by decoding pad_token_id {tokenizer.pad_token_id}.")
            else:
                placeholder_pad_token_str = f"[PAD_ID_{tokenizer.pad_token_id}]"
                tokenizer.pad_token = placeholder_pad_token_str
                logger.warning(f"  Could not decode pad_token_id {tokenizer.pad_token_id} to a non-empty string. Using placeholder string: '{placeholder_pad_token_str}'.")


    if tokenizer.pad_token_id is None:
        logger.error("CRITICAL: Could not set a pad_token_id for the tokenizer. Padding will fail. Please check the tokenizer's configuration.")
        raise ValueError("pad_token_id could not be determined or set. This is required for padding sequences.")

    # Log final state of essential tokens after all modifications
    logger.info(f"--- Final Tokenizer State ---")
    logger.info(f"  Effective Vocab Size (len(tokenizer)): {len(tokenizer)}")
    logger.info(f"  Effective Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    if tokenizer.eos_token_id is not None:
        logger.info(f"  Effective EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    # --- End of Final Tokenizer State Logging ---
    return tokenizer

def yield_examples_from_hf_dataset(
    dataset_name_or_path: str,
    dataset_config_name: str = None,
    dataset_data_dir: str = None,
    split: str = "train",
    input_column: str = None,
    target_column: str = None,
    formatting_template: str = None,
    single_content_column: str = "text",
    streaming: bool = True,
    trust_remote_code_hf_dataset: bool = True
):
    """Yields examples from a Hugging Face dataset, applying formatting if specified."""
    logger.info(f"Loading Hugging Face dataset: '{dataset_name_or_path}', Config: '{dataset_config_name}', Data dir: '{dataset_data_dir}', Split: '{split}', Streaming: {streaming}")

    if input_column and target_column and formatting_template:
        if "{input}" not in formatting_template or "{target}" not in formatting_template:
            logger.warning(
                f"Formatting template ('{formatting_template}') might be missing '{{input}}' or '{{target}}' "
                "placeholders. Ensure these are correctly specified if you intend to use them."
            )

    dataset = load_dataset(
        dataset_name_or_path,
        name=dataset_config_name,
        data_dir=dataset_data_dir,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code_hf_dataset
    )

    processed_count = 0
    skipped_count = 0
    dataset_iterator = dataset

    for example_idx, example in enumerate(dataset_iterator):
        text_to_yield = None
        try:
            if input_column and target_column and formatting_template:
                if input_column not in example or target_column not in example or \
                   not example[input_column] or not example[target_column]:
                    skipped_count +=1
                    continue
                text_to_yield = formatting_template.format(input=example[input_column], target=example[target_column])
            elif single_content_column:
                if single_content_column not in example or not example[single_content_column]:
                    skipped_count +=1
                    continue
                text_to_yield = example[single_content_column]
            else:
                logger.warning(f"Example {example_idx}: Misconfiguration for HF dataset. Provide (input_column, target_column, formatting_template) or single_content_column. Skipping.")
                skipped_count +=1
                continue
        except KeyError as e:
            logger.warning(f"KeyError while formatting example {example_idx} with template. Error: {e}. Example keys: {list(example.keys())}. Ensure template placeholders match column names. Skipping.")
            skipped_count +=1
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing example {example_idx} from HF dataset: {e}. Skipping.")
            skipped_count +=1
            continue

        if text_to_yield and isinstance(text_to_yield, str) and text_to_yield.strip():
            yield text_to_yield
            processed_count +=1
        elif text_to_yield is not None:
            skipped_count += 1

    logger.info(f"Finished yielding {processed_count} examples from Hugging Face dataset '{dataset_name_or_path}' (split: {split}).")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} examples due to missing/empty content, formatting errors, or configuration issues.")

def yield_examples_from_text_files_lines(file_pattern: str):
    """Yields examples from local text files, where each non-empty line is an example."""
    logger.info(f"Attempting to read examples from local text files (one line per example) matching pattern: {file_pattern}")
    filepaths = glob.glob(file_pattern, recursive=True)
    if not filepaths:
        logger.warning(f"No files found matching pattern: {file_pattern}. Returning empty iterator.")
        return iter([])

    logger.info(f"Found {len(filepaths)} files to process: {filepaths}")
    overall_lines_yielded = 0
    overall_empty_lines_skipped = 0
    overall_files_with_errors = 0

    for file_path in filepaths:
        logger.info(f"Processing file: {file_path}")
        lines_yielded_in_file = 0
        empty_lines_skipped_in_file = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    stripped_line = line.strip()
                    if not stripped_line:
                        empty_lines_skipped_in_file += 1
                        continue
                    yield stripped_line
                    lines_yielded_in_file += 1
            overall_lines_yielded += lines_yielded_in_file
            overall_empty_lines_skipped += empty_lines_skipped_in_file
            if lines_yielded_in_file > 0 or empty_lines_skipped_in_file > 0:
                 logger.info(f"  File '{file_path}': Yielded {lines_yielded_in_file} lines, skipped {empty_lines_skipped_in_file} empty/whitespace-only lines.")
            else:
                 logger.info(f"  File '{file_path}' contained no processable lines (all lines were empty or whitespace-only).")
        except Exception as e:
            logger.error(f"Error reading or processing file {file_path}: {e}. Skipping this file.")
            overall_files_with_errors += 1
            continue
    logger.info(f"Finished yielding a total of {overall_lines_yielded} non-empty lines from {len(filepaths) - overall_files_with_errors} successfully processed file(s).")
    logger.info(f"Skipped a total of {overall_empty_lines_skipped} empty/whitespace-only lines across all processed files.")
    if overall_files_with_errors > 0:
        logger.warning(f"{overall_files_with_errors} file(s) could not be processed due to errors.")

def yield_examples_from_text_file_chunks(file_path: str, tokenizer: AutoTokenizer, max_length: int):
    """Yields examples by tokenizing a large text file and splitting it into chunks."""
    logger.info(f"Reading examples from local text file (chunking mode): {file_path}")
    if not os.path.isfile(file_path):
        logger.error(f"File not found for chunking: {file_path}. Returning empty iterator.")
        return iter([])
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path} for chunking: {e}. Returning empty iterator.")
        return iter([])
    if not full_text.strip():
        logger.warning(f"File {file_path} is empty or contains only whitespace. No chunks to yield.")
        return iter([])
    logger.info("Tokenizing the full text content for chunking...")
    # For this older chunking method, special tokens are generally not added at this global tokenization stage.
    # They might be added per-chunk later if the main `add_special_tokens` flag is true.
    all_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    logger.info(f"Full text tokenized into {len(all_tokens)} tokens.")
    num_chunks = 0
    if not all_tokens:
        logger.warning(f"Tokenization of {file_path} resulted in zero tokens. No chunks to yield.")
        return iter([])
    for i in range(0, len(all_tokens), max_length):
        chunk_tokens = all_tokens[i : i + max_length]
        # skip_special_tokens=True is typical for decoding intermediate chunks for pre-training.
        yield tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        num_chunks +=1
    logger.info(f"Finished yielding {num_chunks} chunks of max_length {max_length} from {file_path}.")


def yield_examples_from_text_file_chunks_line_buffered(
    file_path: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    line_buffer_char_target: int = 500000,
    add_special_tokens_for_chunking: bool = False
):
    """
    Yields examples by reading a large text file line by line, accumulating lines into
    a buffer, tokenizing this buffer once it reaches a target character size,
    and then splitting the resulting tokens into chunks of `max_length`.
    This approach avoids loading the entire file into memory.

    Args:
        file_path (str): Path to the text file.
        tokenizer (AutoTokenizer): The tokenizer to use.
        max_length (int): The maximum number of tokens for each yielded chunk.
        line_buffer_char_target (int): Approximate number of characters to buffer
                                       before tokenizing the accumulated lines.
        add_special_tokens_for_chunking (bool): Whether to add special tokens when
                                                tokenizing the intermediate text_block.
                                                Typically False for pre-training, as special
                                                tokens (like EOS) are often added per final sequence.
    """
    logger.info(f"Streaming and chunking file (line buffered): {file_path} with char_target: {line_buffer_char_target}, max_length: {max_length}")
    if not os.path.isfile(file_path):
        logger.error(f"File not found for chunking: {file_path}. Returning empty iterator.")
        return iter([])

    current_line_buffer = []
    current_char_count = 0
    all_buffered_tokens = [] # Stores tokens from multiple text_blocks before chunking
    num_chunks_yielded = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                current_line_buffer.append(line)
                current_char_count += len(line)

                if current_char_count >= line_buffer_char_target:
                    text_block = "".join(current_line_buffer)
                    if text_block.strip():
                        logger.debug(f"Tokenizing text_block of ~{current_char_count} chars ending at line index {line_num}")
                        block_tokens = tokenizer.encode(text_block, add_special_tokens=add_special_tokens_for_chunking)
                        all_buffered_tokens.extend(block_tokens)

                    current_line_buffer = []
                    current_char_count = 0

                    while len(all_buffered_tokens) >= max_length:
                        chunk_to_yield_tokens = all_buffered_tokens[:max_length]
                        yield tokenizer.decode(chunk_to_yield_tokens, skip_special_tokens=True)
                        num_chunks_yielded += 1
                        all_buffered_tokens = all_buffered_tokens[max_length:]

            if current_line_buffer:
                text_block = "".join(current_line_buffer)
                if text_block.strip():
                    logger.debug(f"Tokenizing final text_block of ~{current_char_count} chars (remaining lines up to EOF)")
                    block_tokens = tokenizer.encode(text_block, add_special_tokens=add_special_tokens_for_chunking)
                    all_buffered_tokens.extend(block_tokens)
                current_line_buffer = []
                current_char_count = 0

            while len(all_buffered_tokens) >= max_length:
                chunk_to_yield_tokens = all_buffered_tokens[:max_length]
                yield tokenizer.decode(chunk_to_yield_tokens, skip_special_tokens=True)
                num_chunks_yielded += 1
                all_buffered_tokens = all_buffered_tokens[max_length:]

            if all_buffered_tokens:
                logger.debug(f"Yielding final partial chunk of {len(all_buffered_tokens)} tokens.")
                yield tokenizer.decode(all_buffered_tokens, skip_special_tokens=True)
                num_chunks_yielded += 1
                all_buffered_tokens = []

        logger.info(f"Finished yielding {num_chunks_yielded} chunks from {file_path} using line_buffered strategy.")

    except Exception as e:
        logger.error(f"Error reading or processing file {file_path} with line_buffered chunking: {e}", exc_info=True)
        return iter([])

def process_and_save_dataset(example_iterator, tokenizer: AutoTokenizer, output_path: str,
                             max_length: int, max_examples: int = None, add_special_tokens: bool = False,
                             dtype_str: str = "int32", overwrite_output: bool = False):
    """Processes text examples from an iterator, tokenizes them, and saves to a memmap file."""
    if os.path.exists(output_path) and not overwrite_output:
        logger.error(f"Output file {output_path} already exists and --overwrite_output is False. Halting to prevent data loss.")
        raise FileExistsError(f"Output file {output_path} already exists. Use --overwrite_output to replace it.")
    elif os.path.exists(output_path) and overwrite_output:
        logger.info(f"Output file {output_path} exists and will be overwritten as per --overwrite_output flag.")

    max_examples_str = str(max_examples) if max_examples is not None else "all available"
    logger.info(f"Starting processing of up to {max_examples_str} examples.")
    logger.info(f"Tokenization settings for final sequences: max_length={max_length}, add_special_tokens (globally for final sequences)={add_special_tokens}")

    tokenized_sequences = []
    total_tokens_in_valid_examples = 0
    examples_from_iterator_count = 0
    valid_examples_tokenized_count = 0

    target_dtype = np.int32
    if dtype_str == "int16":
        if len(tokenizer) <= 32767:
            target_dtype = np.int16
        else:
            logger.warning(
                f"Requested int16 dtype, but tokenizer vocab size ({len(tokenizer)}) "
                f"exceeds the safe limit for int16 (32767). Using int32 instead to ensure all token IDs fit."
            )
    logger.info(f"Using NumPy dtype '{np.dtype(target_dtype).name}' for storing tokens in the memmap file.")

    for text_content in tqdm(example_iterator,
                             total=max_examples if max_examples is not None else None,
                             desc="Processing examples",
                             unit="example"):
        examples_from_iterator_count += 1
        if not text_content or not isinstance(text_content, str) or not text_content.strip():
            continue

        # This is the final tokenization pass for each example yielded by the iterators.
        # `add_special_tokens` here refers to the global flag.
        tokens = tokenizer.encode(
            text_content,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        if not tokens:
            continue
        tokenized_sequences.append(tokens)
        total_tokens_in_valid_examples += len(tokens)
        valid_examples_tokenized_count += 1
        if max_examples is not None and valid_examples_tokenized_count >= max_examples:
            logger.info(f"Limit of {max_examples} valid examples to process and tokenize has been reached.")
            break
        if max_examples is not None and examples_from_iterator_count > max_examples * 2 and examples_from_iterator_count > 1000:
             logger.warning(f"Attempted to read {examples_from_iterator_count} items from iterator but only found {valid_examples_tokenized_count} valid examples. "
                            f"Stopping early to prevent potential infinite loop for max_examples={max_examples}.")
             break

    num_sequences_to_save = len(tokenized_sequences)
    if num_sequences_to_save == 0:
        logger.warning("No valid examples were processed or tokenized. The output memmap file will not be created or will be empty if it previously existed and was overwritten.")
        if os.path.exists(output_path) and overwrite_output:
             try: # Attempt to create an empty file or clear if overwriting
                open(output_path, 'w').close()
                logger.info(f"Empty output file {output_path} created/cleared due to no sequences to save and overwrite_output=True.")
             except Exception as e_file:
                logger.error(f"Could not create/clear empty output file {output_path}: {e_file}")
        return 0

    logger.info(f"Read {examples_from_iterator_count} items from the example iterator.")
    logger.info(f"Successfully tokenized {valid_examples_tokenized_count} valid examples.")
    logger.info(f"Preparing to save {num_sequences_to_save} sequences to memmap file.")

    bytes_per_token = np.dtype(target_dtype).itemsize
    estimated_size_bytes = num_sequences_to_save * max_length * bytes_per_token
    logger.info(f"Estimated memmap file size: {estimated_size_bytes / (1024 * 1024):.2f} MB (Shape: ({num_sequences_to_save}, {max_length}), Dtype: {np.dtype(target_dtype).name})")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Output directory created: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            raise

    memmap_data = np.memmap(output_path, dtype=target_dtype, mode='w+', shape=(num_sequences_to_save, max_length))
    logger.info("Populating memmap file with tokenized sequences (padding/truncating as needed)...")
    pad_token_id_to_use = tokenizer.pad_token_id
    if pad_token_id_to_use is None:
        logger.error("CRITICAL: tokenizer.pad_token_id is None at the saving stage. Cannot pad sequences.")
        # This should ideally be caught earlier by get_tokenizer, but as a safeguard:
        raise ValueError("pad_token_id is None, cannot perform padding. This should have been set during tokenizer loading.")

    for i, tokens in tqdm(enumerate(tokenized_sequences), total=num_sequences_to_save, desc="Saving sequences to memmap", unit="sequence"):
        current_len = len(tokens)
        if current_len > max_length: # Should not happen if truncation=True in encode, but defensive.
            padded_tokens = tokens[:max_length]
        elif current_len < max_length:
            padded_tokens = tokens + [pad_token_id_to_use] * (max_length - current_len)
        else:
            padded_tokens = tokens
        try:
            memmap_data[i, :] = padded_tokens
        except ValueError as e:
            logger.error(f"ValueError while assigning to memmap for sequence {i}. Tokens length: {len(padded_tokens)}, Expected: {max_length}. Error: {e}")
            logger.error(f"Problematic tokens (first 10): {padded_tokens[:10]}")
            raise
    memmap_data.flush()
    del memmap_data
    logger.info(f"Successfully saved {num_sequences_to_save} sequences to {output_path} (final shape: ({num_sequences_to_save}, {max_length})).")
    logger.info(f"Total tokens from valid examples (sum of lengths before padding/truncating to max_length): {total_tokens_in_valid_examples:,}")
    pad_token_str = tokenizer.decode([pad_token_id_to_use], skip_special_tokens=False)
    logger.info(f"Pad token ID used for padding: {pad_token_id_to_use} (decoded as: '{pad_token_str}')")
    return num_sequences_to_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Prepares and tokenizes text/code datasets for Lunaris Codex (v{SCRIPT_VERSION}), saving to a memory-mapped NumPy file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    source_group = parser.add_argument_group('Data Source Configuration')
    source_group.add_argument("--data_source_type", type=str, default="hf_dataset",
                        choices=["hf_dataset", "text_file_lines", "text_file_chunks", "text_file_chunks_line_buffered"],
                        help="The type of data source to use.")
    source_group.add_argument("--dataset_name_or_path", type=str, required=True,
                        help="Identifier for Hugging Face dataset (e.g., 'username/dataset_name') or a path/glob pattern for local files (e.g., './data/*.txt'). For chunking modes, this must be a single file path.")

    hf_group = parser.add_argument_group('Hugging Face Dataset Specific Arguments (used if data_source_type is hf_dataset)')
    hf_group.add_argument("--hf_dataset_config_name", type=str, default=None,
                        help="Specific configuration name for the Hugging Face dataset (e.g., 'wikitext-103-raw-v1' for 'wikitext').")
    hf_group.add_argument("--hf_dataset_data_dir", type=str, default=None,
                        help="Subdirectory within the Hugging Face dataset repository that contains data files (e.g., 'data' for meryyllebr543/lunaris-data).")
    hf_group.add_argument("--hf_dataset_split", type=str, default="train",
                        help="The dataset split to load (e.g., 'train', 'validation', 'test').")
    hf_group.add_argument("--hf_input_column", type=str, default=None,
                        help="For structured HF datasets: name of the column for input/prompt text.")
    hf_group.add_argument("--hf_target_column", type=str, default=None,
                        help="For structured HF datasets: name of the column for target/response text.")
    hf_group.add_argument("--hf_formatting_template", type=str, default=None,
                        help="For structured HF datasets: a string template to combine input and target columns. "
                             "Example: 'USER: {input}\\nASSISTANT: {target}'. Use {input} and {target} as placeholders.")
    hf_group.add_argument("--hf_single_content_column", type=str, default="text",
                        help="For HF datasets with a single main text column: the name of that column. "
                             "Used if --hf_input_column and --hf_target_column are not provided. Common examples: 'text', 'content', 'code'.")
    hf_group.add_argument("--hf_dataset_trust_remote_code", action="store_true",
                        help="Pass trust_remote_code=True when loading Hugging Face dataset. Use with caution.")

    tokenizer_group = parser.add_argument_group('Tokenizer Configuration')
    tokenizer_group.add_argument("--tokenizer_name_or_path", type=str, required=True,
                        help="Name of the tokenizer on Hugging Face Hub (e.g., 'gpt2') or a local path to tokenizer files.")
    tokenizer_group.add_argument("--tokenizer_trust_remote_code", action="store_true",
                        help="Pass trust_remote_code=True when loading tokenizer. Use with caution for unverified tokenizers.")

    proc_group = parser.add_argument_group('Processing Configuration')
    proc_group.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length (in tokens) for each example. Longer sequences are truncated, shorter ones padded.")
    proc_group.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of valid examples to process and save. If None, processes all available examples.")
    proc_group.add_argument("--add_special_tokens", action="store_true",
                        help="If set, instruct the tokenizer to add special tokens (e.g., BOS/EOS) during the final encoding step for each sequence, if configured for it.")
    proc_group.add_argument("--output_dtype", type=str, default="int32", choices=["int16", "int32"],
                        help="NumPy data type for storing token IDs in the output .memmap file ('int16' or 'int32').")

    line_buffered_group = parser.add_argument_group('Line-Buffered Text File Chunking Specific Arguments (used if data_source_type is text_file_chunks_line_buffered)')
    line_buffered_group.add_argument("--line_buffer_char_target", type=int, default=500000,
                        help="Approximate number of characters to buffer before tokenizing when using text_file_chunks_line_buffered.")
    line_buffered_group.add_argument("--add_special_tokens_for_chunking", action="store_true",
                        help="Whether to add special tokens when tokenizing intermediate text blocks for text_file_chunks_line_buffered. Typically False for pre-training.")

    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument("--output_path", type=str, required=True,
                        help="Path to save the processed memory-mapped dataset file (e.g., ./processed_data/dataset.memmap).")
    output_group.add_argument("--overwrite_output", action="store_true",
                        help="If set, overwrite the output file if it already exists. Otherwise, an error is raised.")

    args = parser.parse_args()

    if args.data_source_type == "hf_dataset":
        if args.hf_input_column and not (args.hf_target_column and args.hf_formatting_template):
            parser.error("--hf_input_column requires both --hf_target_column and --hf_formatting_template to be set for structured datasets.")
        if not args.hf_input_column and not args.hf_target_column and not args.hf_single_content_column:
             logger.warning("For 'hf_dataset', neither structured input/target columns nor '--hf_single_content_column' were specified. "
                           f"Attempting to use default '{args.hf_single_content_column}', but this may fail if the column does not exist. "
                            "It's recommended to explicitly specify the content column(s).")
    elif args.data_source_type == "text_file_chunks" or args.data_source_type == "text_file_chunks_line_buffered":
        if not os.path.isfile(args.dataset_name_or_path):
            parser.error(f"For data_source_type '{args.data_source_type}', --dataset_name_or_path must be a path to a single existing file. Provided: '{args.dataset_name_or_path}'")

    try:
        logger.info(f"--- Lunaris Codex: Data Preparation (v{SCRIPT_VERSION}) ---")
        logger.info(f"Full run arguments: {vars(args)}")

        tokenizer = get_tokenizer(args.tokenizer_name_or_path, trust_remote_code_flag=args.tokenizer_trust_remote_code)

        example_iterator = None
        logger.info(f"Preparing to load data from source type: '{args.data_source_type}' using dataset identifier: '{args.dataset_name_or_path}'")

        if args.data_source_type == "hf_dataset":
            example_iterator = yield_examples_from_hf_dataset(
                args.dataset_name_or_path,
                dataset_config_name=args.hf_dataset_config_name,
                dataset_data_dir=args.hf_dataset_data_dir,
                split=args.hf_dataset_split,
                input_column=args.hf_input_column,
                target_column=args.hf_target_column,
                formatting_template=args.hf_formatting_template,
                single_content_column=args.hf_single_content_column,
                trust_remote_code_hf_dataset=args.hf_dataset_trust_remote_code
            )
        elif args.data_source_type == "text_file_lines":
            example_iterator = yield_examples_from_text_files_lines(args.dataset_name_or_path)
        elif args.data_source_type == "text_file_chunks":
            example_iterator = yield_examples_from_text_file_chunks(
                args.dataset_name_or_path,
                tokenizer,
                args.max_length
            )
        elif args.data_source_type == "text_file_chunks_line_buffered":
            example_iterator = yield_examples_from_text_file_chunks_line_buffered(
                args.dataset_name_or_path,
                tokenizer,
                args.max_length,
                line_buffer_char_target=args.line_buffer_char_target,
                add_special_tokens_for_chunking=args.add_special_tokens_for_chunking
            )

        if example_iterator:
            num_sequences_saved = process_and_save_dataset(
                example_iterator, tokenizer, args.output_path,
                args.max_length, args.max_examples, args.add_special_tokens,
                dtype_str=args.output_dtype,
                overwrite_output=args.overwrite_output
            )
            if num_sequences_saved > 0:
                logger.info(f"Data preparation finished successfully. {num_sequences_saved} sequences were saved to {args.output_path}.")
            else:
                logger.warning(f"Data preparation completed, but no sequences were saved. This might be due to "
                               "empty input, all examples being filtered out, or max_examples=0. "
                               f"Output file: {args.output_path}")
        else:
            logger.error("Failed to initialize example iterator. This typically occurs if no input files are found "
                         "matching the pattern for 'text_file_lines', an invalid file path for chunking modes, "
                         "or if there's an issue loading the Hugging Face dataset "
                         "(e.g., dataset not found, network issues, or permission problems for private datasets). "
                         "Please verify data source parameters, file paths, and network connectivity.")

    except FileNotFoundError as fnf_error:
        logger.error(f"A required file was not found: {fnf_error}")
    except FileExistsError as fee_error:
        logger.error(f"Output file exists and overwrite is not permitted: {fee_error}")
    except ValueError as val_error:
        logger.error(f"A value-related error occurred: {val_error}", exc_info=True) # Added exc_info for more details on ValueErrors
    except ImportError as imp_error:
        logger.error(f"An import error occurred: {imp_error}. Ensure all dependencies (e.g., 'datasets', 'transformers', 'numpy') are correctly installed.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the data preparation process: {e}", exc_info=True)
