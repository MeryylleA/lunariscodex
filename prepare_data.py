# prepare_data.py
import numpy as np
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_tokenizer(tokenizer_name_or_path: str) -> AutoTokenizer:
    logger.info(f"Loading tokenizer from: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True) # Added trust_remote_code

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            logger.info(f"Tokenizer lacks a pad_token_id. Using eos_token_id ({tokenizer.eos_token_id}) as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.bos_token_id is not None:
            logger.info(f"Tokenizer lacks a pad_token_id or eos_token_id. Using bos_token_id ({tokenizer.bos_token_id}) as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.bos_token_id
        else:
            new_pad_token = '<|PAD|>'
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
            logger.info(f"Tokenizer lacked pad, eos, or bos token. Added new pad_token: '{new_pad_token}' (ID: {tokenizer.pad_token_id}). Vocab size now: {len(tokenizer)}")

    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, Pad token ID: {tokenizer.pad_token_id}")
    return tokenizer

def yield_examples_from_hf_dataset(
    dataset_name_or_path: str,
    dataset_config_name: str = None, # Renamed from config_name for clarity
    dataset_data_dir: str = None, # New argument for subdirectories like 'data/'
    split: str = "train",
    input_column: str = None,
    target_column: str = None,
    formatting_template: str = None, # Example: "Q: {input}\nA: {target}"
    single_content_column: str = "content", # Fallback if not using input/target structure
    streaming: bool = True
):
    logger.info(f"Loading Hugging Face dataset: '{dataset_name_or_path}', Config: '{dataset_config_name}', Data dir: '{dataset_data_dir}', Split: '{split}'")

    # Pass data_dir to load_dataset if provided
    dataset = load_dataset(dataset_name_or_path, name=dataset_config_name, data_dir=dataset_data_dir, split=split, streaming=streaming, trust_remote_code=True)

    count = 0
    skipped_missing_content = 0

    for example_idx, example in enumerate(dataset):
        text_to_yield = None
        if input_column and target_column and formatting_template:
            # Structured input/target processing
            if input_column not in example or target_column not in example or \
               not example[input_column] or not example[target_column]:
                # logger.debug(f"Example {example_idx} missing '{input_column}' or '{target_column}', or content is empty. Skipping.")
                skipped_missing_content +=1
                continue
            try:
                text_to_yield = formatting_template.format(input=example[input_column], target=example[target_column])
            except KeyError as e:
                logger.error(f"KeyError formatting example {example_idx} with template. Error: {e}. Check column names. Example keys: {list(example.keys())}. Skipping.")
                skipped_missing_content +=1
                continue
        elif single_content_column:
            # Fallback to single content column processing
            if single_content_column not in example or not example[single_content_column]:
                # logger.debug(f"Example {example_idx} missing content in column '{single_content_column}' or content is empty. Skipping.")
                skipped_missing_content +=1
                continue
            text_to_yield = example[single_content_column]
        else:
            logger.error(f"Example {example_idx}: Misconfiguration for HF dataset. Provide either (input_column, target_column, formatting_template) or single_content_column. Skipping.")
            skipped_missing_content +=1
            continue

        if text_to_yield:
            yield text_to_yield
            count +=1

    logger.info(f"Finished yielding {count} examples from Hugging Face dataset. Skipped {skipped_missing_content} due to missing/invalid content or configuration issues.")


# ... (yield_examples_from_text_files_lines e yield_examples_from_text_file_chunks permanecem os mesmos) ...
# (Vou omiti-los aqui para encurtar, mas eles estão no código que você já tem)

def yield_examples_from_text_files_lines(file_pattern: str):
    logger.info(f"Reading examples from local text files (one line per example) matching pattern: {file_pattern}")
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
                    if not stripped_line: continue
                    yield stripped_line
                    total_lines_yielded += 1
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}. Skipping this file.")
            continue
    logger.info(f"Finished yielding {total_lines_yielded} non-empty lines from all matched files.")

def yield_examples_from_text_file_chunks(file_path: str, tokenizer: AutoTokenizer, max_length: int):
    logger.info(f"Reading examples from local text file (chunking mode): {file_path}")
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f: full_text = f.read()
    except Exception as e: logger.error(f"Error reading file {file_path}: {e}"); raise
    logger.info("Tokenizing the full text for chunking...")
    all_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    logger.info(f"Full text tokenized into {len(all_tokens)} tokens.")
    num_chunks = 0
    for i in range(0, len(all_tokens), max_length):
        chunk_tokens = all_tokens[i : i + max_length]; yield tokenizer.decode(chunk_tokens); num_chunks +=1
    logger.info(f"Finished yielding {num_chunks} chunks from {file_path}.")


def process_and_save_dataset(example_iterator, tokenizer: AutoTokenizer, output_path: str,
                             max_length: int, max_examples: int = None, add_special_tokens: bool = False,
                             dtype_str: str = "int32"):
    logger.info(f"Starting processing of up to {max_examples if max_examples else 'all available'} examples.")
    logger.info(f"Tokenization settings: max_length={max_length}, add_special_tokens={add_special_tokens}")
    tokenized_sequences, total_tokens_from_examples, examples_attempted_read, examples_processed_valid = [], 0, 0, 0
    dtype = np.int16 if dtype_str == "int16" and tokenizer.vocab_size <= np.iinfo(np.int16).max else np.int32
    if dtype_str == "int16" and tokenizer.vocab_size > np.iinfo(np.int16).max:
        logger.warning(f"Requested int16 dtype, but vocab size ({tokenizer.vocab_size}) exceeds limit. Using int32."); dtype = np.int32
    logger.info(f"Using dtype '{np.dtype(dtype).name}' for tokens in memmap.")

    for i, text_content in tqdm(enumerate(example_iterator), total=max_examples if max_examples else None, desc="Processing examples"):
        examples_attempted_read += 1
        if max_examples and examples_attempted_read > max_examples:
            logger.info(f"Limit of {max_examples} examples to process from iterator reached."); break
        if not text_content or not isinstance(text_content, str) or not text_content.strip(): continue
        tokens = tokenizer.encode(text_content, truncation=True, max_length=max_length, add_special_tokens=add_special_tokens)
        tokenized_sequences.append(tokens); total_tokens_from_examples += len(tokens); examples_processed_valid += 1

    num_sequences_to_save = len(tokenized_sequences)
    if num_sequences_to_save == 0:
        logger.error("No valid examples were processed to be saved! Check data source and parameters."); raise ValueError("No examples processed.")

    logger.info(f"Attempted to read {examples_attempted_read - (1 if max_examples and examples_attempted_read > max_examples else 0)} examples from source.")
    logger.info(f"Successfully processed and tokenized {examples_processed_valid} valid examples.")
    logger.info(f"Creating memmap file for {num_sequences_to_save} sequences.")
    bytes_per_token = np.dtype(dtype).itemsize; estimated_size_bytes = num_sequences_to_save * max_length * bytes_per_token
    logger.info(f"Estimated memmap file size: {estimated_size_bytes / (1024 * 1024):.2f} MB")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir); logger.info(f"Output directory created: {output_dir}")
    memmap_data = np.memmap(output_path, dtype=dtype, mode='w+', shape=(num_sequences_to_save, max_length))

    logger.info("Populating memmap file with tokenized sequences and padding...")
    for i, tokens in tqdm(enumerate(tokenized_sequences), total=num_sequences_to_save, desc="Saving sequences to memmap"):
        current_len = len(tokens)
        padded_tokens = tokens + [tokenizer.pad_token_id] * (max_length - current_len) if current_len <= max_length else tokens[:max_length]
        if not current_len <= max_length: logger.warning(f"Seq {i} len {current_len} > max_length {max_length}, truncated.")
        memmap_data[i, :] = padded_tokens

    memmap_data.flush()
    logger.info(f"Successfully saved {num_sequences_to_save} sequences to {output_path} (shape: {memmap_data.shape})")
    logger.info(f"Total tokens from valid examples (before final padding per sequence): {total_tokens_from_examples:,}")
    logger.info(f"Pad token ID used for padding: {tokenizer.pad_token_id}")
    return num_sequences_to_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares and tokenizes text/code datasets for Lunaris Codex, saving to a memory-mapped file.")

    # Data Source Args
    parser.add_argument("--data_source_type", type=str, default="hf_dataset",
                        choices=["hf_dataset", "text_file_lines", "text_file_chunks"],
                        help="The type of data source to use.")
    parser.add_argument("--dataset_name_or_path", type=str, required=True,
                        help="Name/path for the Hugging Face dataset or path/glob pattern for local files.")
    # HF specific
    parser.add_argument("--hf_dataset_config_name", type=str, default=None,
                        help="Specific configuration or 'data_dir' for Hugging Face datasets (e.g., 'data/python'). Only for 'hf_dataset'.")
    parser.add_argument("--hf_dataset_data_dir", type=str, default=None,
                        help="Subdirectory within the Hugging Face dataset repository containing data files (e.g., 'data'). Only for 'hf_dataset'.")
    parser.add_argument("--hf_dataset_split", type=str, default="train",
                        help="The dataset split to use (e.g., 'train', 'validation'). Only for 'hf_dataset'.")
    parser.add_argument("--hf_input_column", type=str, default=None,
                        help="For structured HF datasets: name of the column for input/prompt text.")
    parser.add_argument("--hf_target_column", type=str, default=None,
                        help="For structured HF datasets: name of the column for target/response text.")
    parser.add_argument("--hf_formatting_template", type=str, default=None,
                        help="For structured HF datasets: a string template to combine input and target columns, e.g., 'PROMPT: {input}\\nRESPONSE: {target}'. Use {input} and {target} as placeholders.")
    parser.add_argument("--hf_single_content_column", type=str, default="content",
                        help="For HF datasets with a single text column: the name of that column. Used if hf_input_column is not set.")

    # Tokenizer Arguments
    parser.add_argument("--tokenizer_name_or_path", type=str, default="bigcode/starcoder",
                        help="Name or path of the Hugging Face tokenizer.")

    # Processing Arguments
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length for tokenization and memmap.")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process. Processes all if None.")
    parser.add_argument("--add_special_tokens", action="store_true",
                        help="Tokenizer should add special tokens (e.g., BOS/EOS). Default: False.")
    parser.add_argument("--output_dtype", type=str, default="int32", choices=["int16", "int32"],
                        help="Data type for saving tokens ('int16' or 'int32').")

    # Output Argument
    parser.add_argument("--output_path", type=str, default="processed_data/lunaris_dataset.memmap",
                        help="Path to save the processed memory-mapped dataset file.")

    args = parser.parse_args()

    try:
        logger.info("Initializing tokenizer...")
        tokenizer = get_tokenizer(args.tokenizer_name_or_path)

        example_iterator = None
        logger.info(f"Preparing to load data from source type: {args.data_source_type}")

        if args.data_source_type == "hf_dataset":
            example_iterator = yield_examples_from_hf_dataset(
                args.dataset_name_or_path,
                dataset_config_name=args.hf_dataset_config_name,
                dataset_data_dir=args.hf_dataset_data_dir,
                split=args.hf_dataset_split,
                input_column=args.hf_input_column,
                target_column=args.hf_target_column,
                formatting_template=args.hf_formatting_template,
                single_content_column=args.hf_single_content_column
            )
        elif args.data_source_type == "text_file_lines":
            example_iterator = yield_examples_from_text_files_lines(args.dataset_name_or_path)
        elif args.data_source_type == "text_file_chunks":
            example_iterator = yield_examples_from_text_file_chunks(args.dataset_name_or_path, tokenizer, args.max_length)
        else:
            raise ValueError(f"Unsupported data_source_type: {args.data_source_type}")

        if example_iterator:
            num_sequences_saved = process_and_save_dataset(
                example_iterator, tokenizer, args.output_path,
                args.max_length, args.max_examples, args.add_special_tokens,
                dtype_str=args.output_dtype
            )
            logger.info(f"Data preparation finished successfully. {num_sequences_saved} sequences were saved to {args.output_path}.")
        else:
            logger.error("Could not create example iterator. Please check data source parameters and file paths.")

    except FileNotFoundError as fnf_error:
        logger.error(f"File Not Found Error: {fnf_error}")
    except ValueError as val_error:
        logger.error(f"Value Error: {val_error}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the data preparation process: {e}", exc_info=True)
