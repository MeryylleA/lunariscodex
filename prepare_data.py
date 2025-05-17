# prepare_data.py
import torch # Maintained for consistency, though not directly used here
import numpy as np
from datasets import load_dataset, IterableDataset # Dataset class also imported if needed for type hints elsewhere
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os
import logging
import glob # For expanding glob patterns

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_tokenizer(tokenizer_name_or_path: str) -> AutoTokenizer:
    """
    Loads a tokenizer using AutoTokenizer from Hugging Face Transformers.
    Sets a pad_token_id if it's not already defined, preferring eos_token_id,
    then bos_token_id, or adding a new '<|PAD|>' token as a last resort.

    Args:
        tokenizer_name_or_path (str): The name or path of the tokenizer.

    Returns:
        AutoTokenizer: The loaded tokenizer instance.
    """
    logger.info(f"Loading tokenizer from: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            logger.info(f"Tokenizer does not have a pad_token_id. Using eos_token_id ({tokenizer.eos_token_id}) as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.bos_token_id is not None:
            logger.info(f"Tokenizer does not have a pad_token_id or eos_token_id. Using bos_token_id ({tokenizer.bos_token_id}) as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.bos_token_id
        else:
            # Add a new pad token if no other suitable token is found
            new_pad_token = '<|PAD|>'
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
            logger.info(f"Tokenizer did not have pad, eos, or bos token. Added new pad_token: '{new_pad_token}' (ID: {tokenizer.pad_token_id}). Vocab size is now: {len(tokenizer)}")
            # Note: If a new token is added, the model's embedding layer might need resizing if vocab_size was fixed.

    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, Pad token ID: {tokenizer.pad_token_id}")
    return tokenizer

def yield_examples_from_hf_dataset(dataset_name_or_path: str, config_name: str, split: str, content_column: str = "content", streaming: bool = True):
    """
    Yields text examples from a Hugging Face Hub dataset.

    Args:
        dataset_name_or_path (str): Name or path of the Hugging Face dataset.
        config_name (str): Specific configuration/subset of the dataset (e.g., 'data/python').
        split (str): Dataset split to use (e.g., 'train', 'validation').
        content_column (str): The name of the column containing the text content.
        streaming (bool): Whether to load the dataset in streaming mode.

    Yields:
        str: Text content of each example.
    """
    logger.info(f"Loading Hugging Face dataset: {dataset_name_or_path}, Config: {config_name}, Split: {split}, Column: {content_column}")
    dataset = load_dataset(dataset_name_or_path, name=config_name, split=split, streaming=streaming, trust_remote_code=True) # Added trust_remote_code for some datasets

    count = 0
    skipped_missing_content = 0
    for example in dataset:
        if content_column not in example or not example[content_column]:
            # logger.debug(f"Example {count} missing content in column '{content_column}' or content is empty. Skipping.")
            skipped_missing_content +=1
            continue
        yield example[content_column]
        count +=1
    logger.info(f"Finished yielding {count} examples from Hugging Face dataset. Skipped {skipped_missing_content} due to missing content.")

def yield_examples_from_text_files_lines(file_pattern: str):
    """
    Yields text examples from local text files, where each non-empty line is an example.
    Supports glob patterns for file matching.

    Args:
        file_pattern (str): Glob pattern શાંતmatching input text files (e.g., './data/*.txt').

    Yields:
        str: Text content of each non-empty line.
    """
    logger.info(f"Reading examples from local text files (one line per example) matching pattern: {file_pattern}")
    # Expand glob pattern to get a list of file paths
    # recursive=True allows for patterns like 'data/**/*.txt'
    filepaths = glob.glob(file_pattern, recursive=True)

    if not filepaths:
        logger.error(f"No files found matching pattern: {file_pattern}")
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    logger.info(f"Found {len(filepaths)} files to process: {filepaths}")
    total_lines_yielded = 0
    for file_path in filepaths:
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    stripped_line = line.strip()
                    if not stripped_line: # Skip empty or whitespace-only lines
                        # logger.debug(f"Skipping empty line {line_num+1} in {file_path}")
                        continue
                    yield stripped_line
                    total_lines_yielded += 1
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}. Skipping this file.")
            continue
    logger.info(f"Finished yielding {total_lines_yielded} non-empty lines from all matched files.")

def yield_examples_from_text_file_chunks(file_path: str, tokenizer: AutoTokenizer, max_length: int):
    """
    Yields text examples from a single large local text file, by splitting it into chunks
    based on tokenized length.

    Args:
        file_path (str): Path to the input text file.
        tokenizer (AutoTokenizer): The tokenizer instance.
        max_length (int): The maximum number of tokens per chunk.

    Yields:
        str: Text content of each chunk.
    """
    logger.info(f"Reading examples from local text file (chunking mode): {file_path}")
    if not os.path.isfile(file_path): # Ensure the file exists
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}"); raise

    logger.info("Tokenizing the full text for chunking...")
    # Tokenize without adding special tokens here, as `process_and_save_dataset` will handle it per chunk
    all_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    logger.info(f"Full text tokenized into {len(all_tokens)} tokens.")

    num_chunks = 0
    for i in range(0, len(all_tokens), max_length):
        chunk_tokens = all_tokens[i : i + max_length]
        # Decode back to text to maintain a consistent interface for process_and_save_dataset.
        # Future optimization: allow process_and_save_dataset to accept pre-tokenized input.
        yield tokenizer.decode(chunk_tokens)
        num_chunks +=1
    logger.info(f"Finished yielding {num_chunks} chunks from {file_path}.")

def process_and_save_dataset(example_iterator, tokenizer: AutoTokenizer, output_path: str,
                             max_length: int, max_examples: int = None, add_special_tokens: bool = False,
                             dtype_str: str = "int32"):
    """
    Processes text examples from an iterator, tokenizes them, and saves them
    to a memory-mapped NumPy file.

    Args:
        example_iterator: An iterator yielding text strings.
        tokenizer (AutoTokenizer): The tokenizer instance.
        output_path (str): Path to save the output .memmap file.
        max_length (int): Maximum sequence length for tokenization and padding.
        max_examples (int, optional): Maximum number of examples to process. Defaults to None (all).
        add_special_tokens (bool): Whether the tokenizer should add special tokens (BOS/EOS).
        dtype_str (str): Data type for storing tokens ('int16' or 'int32').

    Returns:
        int: The number of sequences successfully saved.
    """
    logger.info(f"Starting processing of up to {max_examples if max_examples else 'all available'} examples.")
    logger.info(f"Tokenization settings: max_length={max_length}, add_special_tokens={add_special_tokens}")

    tokenized_sequences = []
    total_tokens_from_examples = 0
    examples_attempted_read = 0
    examples_processed_valid = 0

    # Determine dtype for memmap based on vocab size and user preference
    dtype = np.int16 if dtype_str == "int16" and tokenizer.vocab_size <= np.iinfo(np.int16).max else np.int32
    if dtype_str == "int16" and tokenizer.vocab_size > np.iinfo(np.int16).max:
        logger.warning(f"Requested int16 dtype, but vocab size ({tokenizer.vocab_size}) exceeds limit for int16. Using int32 instead.")
        dtype = np.int32
    logger.info(f"Using dtype '{np.dtype(dtype).name}' for tokens in memmap.")

    for i, text_content in tqdm(enumerate(example_iterator), total=max_examples if max_examples else None, desc="Processing examples"):
        examples_attempted_read += 1
        if max_examples and examples_attempted_read > max_examples:
            logger.info(f"Limit of {max_examples} examples to process from iterator reached.")
            break # Stop iterating if max_examples limit is hit

        if not text_content or not isinstance(text_content, str) or not text_content.strip():
            # logger.debug(f"Example {i} is empty, not a string, or whitespace-only. Skipping.")
            continue # Skip this example

        tokens = tokenizer.encode(
            text_content,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )

        tokenized_sequences.append(tokens)
        total_tokens_from_examples += len(tokens)
        examples_processed_valid += 1 # Count only valid, non-empty examples that were tokenized

    num_sequences_to_save = len(tokenized_sequences)
    if num_sequences_to_save == 0:
        logger.error("No valid examples were processed to be saved! Please check your data source and parameters.")
        raise ValueError("No examples were processed successfully.")

    logger.info(f"Attempted to read {examples_attempted_read-1 if max_examples and examples_attempted_read > max_examples else examples_attempted_read} examples from source.") # examples_attempted_read includes the one that breaks the loop
    logger.info(f"Successfully processed and tokenized {examples_processed_valid} valid examples.")
    logger.info(f"Creating memmap file for {num_sequences_to_save} sequences.")

    # Estimate output file size
    bytes_per_token = np.dtype(dtype).itemsize
    estimated_size_bytes = num_sequences_to_save * max_length * bytes_per_token
    estimated_size_mb = estimated_size_bytes / (1024 * 1024)
    logger.info(f"Estimated memmap file size: {estimated_size_mb:.2f} MB")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Output directory created: {output_dir}")

    # Create and populate the memory-mapped file
    memmap_data = np.memmap(
        output_path, dtype=dtype, mode='w+', shape=(num_sequences_to_save, max_length)
    )

    logger.info("Populating memmap file with tokenized sequences and padding...")
    for i, tokens in tqdm(enumerate(tokenized_sequences), total=num_sequences_to_save, desc="Saving sequences to memmap"):
        current_len = len(tokens)
        if current_len <= max_length: # Should always be true due to truncation
            # Pad sequence if it's shorter than max_length
            padded_tokens = tokens + [tokenizer.pad_token_id] * (max_length - current_len)
        else:
            # This case should ideally not be hit if truncation=True in tokenizer.encode
            logger.warning(f"Sequence {i} has length {current_len} > max_length {max_length} despite truncation. Truncating again.")
            padded_tokens = tokens[:max_length]

        memmap_data[i, :] = padded_tokens

    memmap_data.flush() # Ensure data is written to disk
    logger.info(f"Successfully saved {num_sequences_to_save} sequences to {output_path} (shape: {memmap_data.shape})")
    logger.info(f"Total tokens from valid examples (before final padding per sequence): {total_tokens_from_examples:,}")
    logger.info(f"Pad token ID used for padding: {tokenizer.pad_token_id}")
    return num_sequences_to_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares and tokenizes text/code datasets for Lunaris Codex, saving to a memory-mapped file.")

    # --- Data Source Arguments ---
    parser.add_argument("--data_source_type", type=str, default="hf_dataset",
                        choices=["hf_dataset", "text_file_lines", "text_file_chunks"],
                        help="The type of data source to use.")
    parser.add_argument("--dataset_name_or_path", type=str, required=True,
                        help="Name/path for the Hugging Face dataset (e.g., 'bigcode/the-stack-dedup') or path/glob pattern for local files (e.g., './my_texts/*.txt').")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="Specific configuration or 'data_dir' for Hugging Face datasets (e.g., 'data/python'). Only applicable if data_source_type is 'hf_dataset'.")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="The dataset split to use (e.g., 'train', 'validation'). Only for 'hf_dataset'.")
    parser.add_argument("--content_column", type=str, default="content",
                        help="The column name in the Hugging Face dataset that contains the text/code. Only for 'hf_dataset'.")

    # --- Tokenizer Arguments ---
    parser.add_argument("--tokenizer_name_or_path", type=str, default="bigcode/starcoder",
                        help="Name or path of the Hugging Face tokenizer (e.g., 'gpt2', './my_tokenizer/').")

    # --- Processing Arguments ---
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length. Sequences will be truncated or padded to this length.")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process from the source. Processes all if None. (Default: None)")
    parser.add_argument("--add_special_tokens", action="store_true",
                        help="Whether the tokenizer should add special tokens (e.g., BOS/EOS). (Default: False)")
    parser.add_argument("--output_dtype", type=str, default="int32", choices=["int16", "int32"],
                        help="Data type for storing tokens in the memmap file ('int16' or 'int32'). 'int16' saves space if vocab_size <= 65535. (Default: int32)")

    # --- Output Argument ---
    parser.add_argument("--output_path", type=str, default="processed_data/lunaris_dataset.memmap",
                        help="Path to save the processed memory-mapped dataset file. (Default: processed_data/lunaris_dataset.memmap)")

    args = parser.parse_args()

    try:
        logger.info("Initializing tokenizer...")
        tokenizer = get_tokenizer(args.tokenizer_name_or_path)

        example_iterator = None
        logger.info(f"Preparing to load data from source type: {args.data_source_type}")

        if args.data_source_type == "hf_dataset":
            example_iterator = yield_examples_from_hf_dataset(
                args.dataset_name_or_path,
                args.dataset_config_name,
                args.dataset_split,
                args.content_column
            )
        elif args.data_source_type == "text_file_lines":
            example_iterator = yield_examples_from_text_files_lines(args.dataset_name_or_path)
        elif args.data_source_type == "text_file_chunks":
            example_iterator = yield_examples_from_text_file_chunks(args.dataset_name_or_path, tokenizer, args.max_length)
        else:
            # This case should be caught by argparse choices, but as a safeguard:
            raise ValueError(f"Unsupported data_source_type: {args.data_source_type}")

        if example_iterator:
            num_sequences_saved = process_and_save_dataset(
                example_iterator,
                tokenizer,
                args.output_path,
                args.max_length,
                args.max_examples,
                args.add_special_tokens,
                dtype_str=args.output_dtype
            )
            logger.info(f"Data preparation finished successfully. {num_sequences_saved} sequences were saved to {args.output_path}.")
        else:
            # This should ideally not be reached if an exception is raised in the iterators for invalid paths etc.
            logger.error("Failed to create example iterator. Please check data source parameters and file paths.")

    except FileNotFoundError as fnf_error:
        logger.error(f"File Not Found Error: {fnf_error}")
    except ValueError as val_error: # Catch ValueErrors from our script or libraries
        logger.error(f"Value Error: {val_error}")
    except Exception as e: # Catch-all for any other unexpected errors
        logger.error(f"An unexpected error occurred during the data preparation process: {e}", exc_info=True)
