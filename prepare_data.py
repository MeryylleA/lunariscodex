# prepare_data.py
import numpy as np
from datasets import load_dataset # IterableDataset is not currently used, can be removed if not planned.
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SCRIPT VERSION ---
SCRIPT_VERSION = "0.2.1" # Updated version reflecting these changes

def get_tokenizer(tokenizer_name_or_path: str, trust_remote_code_flag: bool = True) -> AutoTokenizer: # Added flag for trust_remote_code
    """
    Loads a tokenizer using AutoTokenizer from Hugging Face.
    Handles common cases for setting pad_token_id if not already defined.
    """
    logger.info(f"Loading tokenizer from: {tokenizer_name_or_path}")
    # trust_remote_code=True is kept as per your original, but now configurable via a flag in main
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=trust_remote_code_flag)

    if tokenizer.pad_token_id is None:
        original_vocab_size = len(tokenizer)

        if tokenizer.eos_token_id is not None:
            logger.info(f"Tokenizer lacks a pad_token_id. Using eos_token_id ({tokenizer.eos_token_id}) as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            pad_token_source_info = f"eos_token ('{tokenizer.eos_token}')"
        elif tokenizer.bos_token_id is not None:
            logger.info(f"Tokenizer lacks a pad_token_id and eos_token_id. Using bos_token_id ({tokenizer.bos_token_id}) as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.bos_token_id
            if tokenizer.pad_token is None and tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
            pad_token_source_info = f"bos_token ('{tokenizer.bos_token}')"
        else:
            new_pad_token_str = '<|PAD|>'
            logger.warning(f"Tokenizer lacked pad, eos, or bos token. Adding new special pad_token: '{new_pad_token_str}'.")
            # Using add_special_tokens to ensure embedding layer is resized if necessary.
            tokenizer.add_special_tokens({'pad_token': new_pad_token_str})
            pad_token_source_info = f"newly added token '{new_pad_token_str}'"
            logger.info(f"Pad token ID {tokenizer.pad_token_id} assigned. Vocab size changed from {original_vocab_size} to {len(tokenizer)}.")

        if tokenizer.pad_token is None and tokenizer.pad_token_id is not None: # Fallback if pad_token string is still None despite ID being set
            logger.warning(f"Pad token string could not be directly inferred from {pad_token_source_info} for pad_token_id {tokenizer.pad_token_id}. "
                           f"Attempting to decode pad_token_id or using a placeholder.")
            # Try to decode the pad_token_id to get its string representation
            decoded_pad_token = tokenizer.decode([tokenizer.pad_token_id], skip_special_tokens=False) # Don't skip, we want to see the special token
            if decoded_pad_token:
                tokenizer.pad_token = decoded_pad_token
                logger.info(f"Inferred pad_token string as '{decoded_pad_token}' by decoding pad_token_id {tokenizer.pad_token_id}.")
            else:
                # As a last resort, create a placeholder string. This is unlikely if add_special_tokens was used.
                placeholder_pad_token_str = f"[PAD_ID_{tokenizer.pad_token_id}]"
                tokenizer.pad_token = placeholder_pad_token_str
                logger.warning(f"Could not decode pad_token_id {tokenizer.pad_token_id}. Using placeholder string: '{placeholder_pad_token_str}'.")


    if tokenizer.pad_token_id is None:
        # This case should ideally not be reached with the logic above.
        logger.error("CRITICAL: Could not set a pad_token_id for the tokenizer. Padding will fail. Please check the tokenizer's configuration.")
        raise ValueError("pad_token_id could not be determined or set. This is required for padding sequences.")

    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, Pad token: '{tokenizer.pad_token}', Pad token ID: {tokenizer.pad_token_id}")
    return tokenizer

def yield_examples_from_hf_dataset(
    dataset_name_or_path: str,
    dataset_config_name: str = None,
    dataset_data_dir: str = None,
    split: str = "train",
    input_column: str = None,
    target_column: str = None,
    formatting_template: str = None,
    single_content_column: str = "text", # Defaulted to 'text' in your latest version
    streaming: bool = True,
    trust_remote_code_hf_dataset: bool = True # Added flag
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
        trust_remote_code=trust_remote_code_hf_dataset # Use the passed flag
    )

    processed_count = 0
    skipped_count = 0

    # tqdm can be used with iterable datasets if the total number of examples is known or estimated.
    # For pure streaming without a total, tqdm won't show a progress bar but will count iterations.
    desc = f"Iterating HF Dataset '{dataset_name_or_path}' ({split})"
    # dataset_iterator = tqdm(dataset, desc=desc) if not streaming else dataset # tqdm is more useful for non-streaming
    dataset_iterator = dataset # Keep it simple for now

    for example_idx, example in enumerate(dataset_iterator):
        text_to_yield = None
        try:
            if input_column and target_column and formatting_template:
                # Process structured data with input, target, and a formatting template
                if input_column not in example or target_column not in example or \
                   not example[input_column] or not example[target_column]: # Check for presence and non-empty
                    # logger.debug(f"Example {example_idx} missing '{input_column}' or '{target_column}', or content is empty. Skipping.")
                    skipped_count +=1
                    continue
                text_to_yield = formatting_template.format(input=example[input_column], target=example[target_column])
            elif single_content_column:
                # Process data with a single content column
                if single_content_column not in example or not example[single_content_column]:
                    # logger.debug(f"Example {example_idx} missing content in column '{single_content_column}' or content is empty. Skipping.")
                    skipped_count +=1
                    continue
                text_to_yield = example[single_content_column]
            else:
                # This case should ideally be caught by argument validation in main
                logger.warning(f"Example {example_idx}: Misconfiguration for HF dataset. Provide (input_column, target_column, formatting_template) or single_content_column. Skipping.")
                skipped_count +=1
                continue
        except KeyError as e:
            logger.warning(f"KeyError while formatting example {example_idx} with template. Error: {e}. Example keys: {list(example.keys())}. Ensure template placeholders match column names. Skipping.")
            skipped_count +=1
            continue
        except Exception as e: # Catch any other unexpected error during example processing
            logger.error(f"Unexpected error processing example {example_idx} from HF dataset: {e}. Skipping.")
            skipped_count +=1
            continue

        # Ensure the result is a non-empty string before yielding
        if text_to_yield and isinstance(text_to_yield, str) and text_to_yield.strip():
            yield text_to_yield
            processed_count +=1
        elif text_to_yield is not None: # If text_to_yield was produced but became empty or non-string
            # logger.debug(f"Example {example_idx} resulted in empty or non-string content after formatting/retrieval. Skipping.")
            skipped_count += 1

    logger.info(f"Finished yielding {processed_count} examples from Hugging Face dataset '{dataset_name_or_path}' (split: {split}).")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} examples due to missing/empty content, formatting errors, or configuration issues.")


def yield_examples_from_text_files_lines(file_pattern: str):
    """Yields examples from local text files, where each non-empty line is an example."""
    logger.info(f"Attempting to read examples from local text files (one line per example) matching pattern: {file_pattern}")

    # glob.glob with recursive=True correctly handles '**' for recursive search.
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
                    if not stripped_line: # Check if line is empty after stripping
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
    # add_special_tokens=False is important here to avoid adding BOS/EOS to every chunk.
    all_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    logger.info(f"Full text tokenized into {len(all_tokens)} tokens.")

    num_chunks = 0
    if not all_tokens:
        logger.warning(f"Tokenization of {file_path} resulted in zero tokens. No chunks to yield.")
        return iter([])

    for i in range(0, len(all_tokens), max_length):
        chunk_tokens = all_tokens[i : i + max_length]
        # Decode the chunk back to string. skip_special_tokens=True is generally good here
        # to avoid re-introducing special tokens that were not part of the original chunk's intent.
        yield tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        num_chunks +=1
    logger.info(f"Finished yielding {num_chunks} chunks of max_length {max_length} from {file_path}.")


def process_and_save_dataset(example_iterator, tokenizer: AutoTokenizer, output_path: str,
                             max_length: int, max_examples: int = None, add_special_tokens: bool = False,
                             dtype_str: str = "int32", overwrite_output: bool = False):
    """Processes text examples from an iterator, tokenizes them, and saves to a memmap file."""

    if os.path.exists(output_path) and not overwrite_output:
        logger.error(f"Output file {output_path} already exists and --overwrite_output is False. Halting to prevent data loss.")
        raise FileExistsError(f"Output file {output_path} already exists. Use --overwrite_output to replace it.")
    elif os.path.exists(output_path) and overwrite_output:
        logger.info(f"Output file {output_path} exists and will be overwritten as per --overwrite_output flag.")

    # Determine max_examples_str for logging
    max_examples_str = str(max_examples) if max_examples is not None else "all available"
    logger.info(f"Starting processing of up to {max_examples_str} examples.")
    logger.info(f"Tokenization settings: max_length={max_length}, add_special_tokens={add_special_tokens}")

    tokenized_sequences = []
    total_tokens_in_valid_examples = 0 # Renamed for clarity
    examples_from_iterator_count = 0
    valid_examples_tokenized_count = 0 # Renamed for clarity

    target_dtype = np.int32
    if dtype_str == "int16":
        # Standard int16 range is -32768 to 32767. Token IDs are non-negative.
        # Max vocab size for int16 should be <= 32767 to be safe, though some might use up to 65535 if thinking unsigned.
        # Transformers tokenizers typically don't exceed 50-100k, but good to be explicit.
        if tokenizer.vocab_size <= 32767:
            target_dtype = np.int16
        else:
            logger.warning(
                f"Requested int16 dtype, but tokenizer vocab size ({tokenizer.vocab_size}) "
                f"exceeds the safe limit for int16 (32767). Using int32 instead to ensure all token IDs fit."
            )
    logger.info(f"Using NumPy dtype '{np.dtype(target_dtype).name}' for storing tokens in the memmap file.")

    # Using tqdm for progress indication
    for text_content in tqdm(example_iterator,
                             total=max_examples if max_examples is not None else None,
                             desc="Processing examples",
                             unit="example"):
        examples_from_iterator_count += 1

        if not text_content or not isinstance(text_content, str) or not text_content.strip():
            # logger.debug(f"Skipping example from iterator: content is empty or not a string.")
            continue

        # Tokenize the current example
        tokens = tokenizer.encode(
            text_content,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )

        if not tokens: # If tokenization results in an empty list (e.g., all unknown tokens and no special tokens added)
            # logger.debug(f"Skipping example: tokenization resulted in an empty token list.")
            continue

        tokenized_sequences.append(tokens)
        total_tokens_in_valid_examples += len(tokens)
        valid_examples_tokenized_count += 1

        if max_examples is not None and valid_examples_tokenized_count >= max_examples:
            logger.info(f"Limit of {max_examples} valid examples to process and tokenize has been reached.")
            break

        # Heuristic to prevent excessively long runs if iterator is bad and max_examples is high
        if max_examples is not None and examples_from_iterator_count > max_examples * 2 and valid_examples_tokenized_count < max_examples:
             logger.warning(f"Attempted to read {examples_from_iterator_count} items from iterator but only found {valid_examples_tokenized_count} valid examples. "
                            f"Stopping early to prevent potential infinite loop for max_examples={max_examples}.")
             break

    num_sequences_to_save = len(tokenized_sequences)
    if num_sequences_to_save == 0:
        logger.warning("No valid examples were processed or tokenized. The output memmap file will not be created or will be empty if it previously existed and was overwritten.")
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

    # Create or overwrite the memmap file
    memmap_data = np.memmap(output_path, dtype=target_dtype, mode='w+', shape=(num_sequences_to_save, max_length))

    logger.info("Populating memmap file with tokenized sequences (padding/truncating as needed)...")
    pad_token_id_to_use = tokenizer.pad_token_id
    if pad_token_id_to_use is None: # Should not happen due to get_tokenizer logic, but as a safeguard
        logger.error("CRITICAL: tokenizer.pad_token_id is None at the saving stage. Cannot pad sequences.")
        raise ValueError("pad_token_id is None, cannot perform padding.")

    for i, tokens in tqdm(enumerate(tokenized_sequences), total=num_sequences_to_save, desc="Saving sequences to memmap", unit="sequence"):
        current_len = len(tokens)
        if current_len > max_length:
            padded_tokens = tokens[:max_length] # Explicit truncation, though tokenizer.encode should handle it
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
    del memmap_data # Close the memmap file by deleting the object

    logger.info(f"Successfully saved {num_sequences_to_save} sequences to {output_path} (final shape: ({num_sequences_to_save}, {max_length})).")
    logger.info(f"Total tokens from valid examples (sum of lengths before padding to max_length): {total_tokens_in_valid_examples:,}")
    pad_token_str = tokenizer.decode([pad_token_id_to_use], skip_special_tokens=False) if pad_token_id_to_use is not None else "[[PAD ID NOT SET]]"
    logger.info(f"Pad token ID used for padding: {pad_token_id_to_use} (decoded as: '{pad_token_str}')")
    return num_sequences_to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Prepares and tokenizes text/code datasets for Lunaris Codex (v{SCRIPT_VERSION}), saving to a memory-mapped NumPy file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Grouping arguments for better help message readability
    source_group = parser.add_argument_group('Data Source Configuration')
    source_group.add_argument("--data_source_type", type=str, default="hf_dataset",
                        choices=["hf_dataset", "text_file_lines", "text_file_chunks"],
                        help="The type of data source to use.")
    source_group.add_argument("--dataset_name_or_path", type=str, required=True,
                        help="Identifier for Hugging Face dataset (e.g., 'username/dataset_name') or a path/glob pattern for local files (e.g., './data/*.txt').")

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
    tokenizer_group.add_argument("--tokenizer_name_or_path", type=str, required=True, # Made required
                        help="Name of the tokenizer on Hugging Face Hub (e.g., 'gpt2') or a local path to tokenizer files.")
    tokenizer_group.add_argument("--tokenizer_trust_remote_code", action="store_true",
                        help="Pass trust_remote_code=True when loading tokenizer. Use with caution for unverified tokenizers.")


    proc_group = parser.add_argument_group('Processing Configuration')
    proc_group.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length (in tokens) for each example. Longer sequences are truncated, shorter ones padded.")
    proc_group.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of valid examples to process and save. If None, processes all available examples.")
    proc_group.add_argument("--add_special_tokens", action="store_true",
                        help="If set, instruct the tokenizer to add special tokens (e.g., BOS/EOS) during encoding, if configured for it.")
    proc_group.add_argument("--output_dtype", type=str, default="int32", choices=["int16", "int32"],
                        help="NumPy data type for storing token IDs in the output .memmap file ('int16' or 'int32').")

    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument("--output_path", type=str, required=True, # Made required
                        help="Path to save the processed memory-mapped dataset file (e.g., ./processed_data/dataset.memmap).")
    output_group.add_argument("--overwrite_output", action="store_true",
                        help="If set, overwrite the output file if it already exists. Otherwise, an error is raised.")

    args = parser.parse_args()

    # Argument validation
    if args.data_source_type == "hf_dataset":
        if args.hf_input_column and not (args.hf_target_column and args.hf_formatting_template):
            parser.error("--hf_input_column requires both --hf_target_column and --hf_formatting_template to be set for structured datasets.")
        if not args.hf_input_column and not args.hf_target_column and not args.hf_single_content_column:
             logger.warning("For 'hf_dataset', neither structured input/target columns nor '--hf_single_content_column' were specified. "
                           f"Attempting to use default '{args.hf_single_content_column}', but this may fail if the column does not exist. "
                            "It's recommended to explicitly specify the content column(s).")
    elif args.data_source_type == "text_file_chunks":
        if not os.path.isfile(args.dataset_name_or_path): # Check specifically for single file for chunking
            parser.error(f"For 'text_file_chunks', --dataset_name_or_path must be a path to a single existing file. Provided: '{args.dataset_name_or_path}'")


    try:
        logger.info(f"--- Lunaris Codex: Data Preparation (v{SCRIPT_VERSION}) ---")
        logger.info(f"Full run arguments: {vars(args)}")

        logger.info("Initializing tokenizer...")
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
                trust_remote_code_hf_dataset=args.hf_dataset_trust_remote_code # Pass flag
                # Consider making streaming an option: streaming=args.hf_streaming
            )
        elif args.data_source_type == "text_file_lines":
            example_iterator = yield_examples_from_text_files_lines(args.dataset_name_or_path)
        elif args.data_source_type == "text_file_chunks":
            # File existence already checked by argparse logic or above
            example_iterator = yield_examples_from_text_file_chunks(args.dataset_name_or_path, tokenizer, args.max_length)

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
                         "matching the pattern, or if there's an issue loading the Hugging Face dataset "
                         "(e.g., dataset not found, network issues, or permission problems for private datasets). "
                         "Please verify data source parameters, file paths, and network connectivity.")

    except FileNotFoundError as fnf_error:
        logger.error(f"A required file was not found: {fnf_error}")
    except FileExistsError as fee_error:
        logger.error(f"Output file exists and overwrite is not permitted: {fee_error}")
    except ValueError as val_error: # Includes issues from tokenizer or data processing
        logger.error(f"A value-related error occurred: {val_error}")
    except ImportError as imp_error:
        logger.error(f"An import error occurred: {imp_error}. Ensure all dependencies (e.g., 'datasets', 'transformers', 'numpy') are correctly installed.")
    except Exception as e: # Catch-all for any other unexpected errors
        logger.error(f"An unexpected error occurred during the data preparation process: {e}", exc_info=True) # exc_info=True will log the full traceback
