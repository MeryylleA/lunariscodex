<!-- README for BPE Processor - Lunaris Codex -->
<!-- Last Updated: May 21, 2025 (Reflecting tokenization implementation) -->

# Lunaris BPE Processor

**Version: 0.2.0** 
<!-- Bump version due to significant new functionality (tokenization) -->

The `Lunaris BPE Processor` (executable name typically `bpe_processor`, formerly `bpe_trainer`) is a command-line utility written in C++. It is designed to:
1.  **Train** Byte Pair Encoding (BPE) models by learning merges from a given text corpus. This generates a vocabulary map (token to ID) and an ordered list of merge operations.
2.  **Tokenize** new input text into a sequence of token IDs using a previously trained BPE model.

This tool is a core component for users of the [Lunaris Codex](https://github.com/MeryylleA/lunariscodex) project who wish to create and utilize custom subword tokenizers tailored to their specific datasets, especially for tasks like code generation or specialized language modeling.

## Purpose and Functionality

The `Lunaris BPE Processor` provides end-to-end BPE capabilities, from model training to text tokenization.

**1. Training (`--action train`):**
    *   Analyzes a raw text corpus.
    *   Initializes a vocabulary based on the chosen mode (byte-level or word-level).
    *   Iteratively identifies the most frequent pair of adjacent tokens in the corpus and merges them to form new tokens.
    *   Saves the learned **vocabulary map** (mapping token strings to integer IDs) and the **ordered list of merge operations**. These are stored in a primary JSON file, with human-readable plain text versions also generated.

**2. Tokenization (`--action tokenize`):**
    *   Loads a pre-trained BPE model (its vocabulary map and merge rules) from the JSON file.
    *   Pre-tokenizes the input text into initial units (bytes or words, based on the loaded model's training mode).
    *   Applies the learned merge rules in their original order to the sequence of initial tokens.
    *   Converts the final sequence of merged tokens into their corresponding integer IDs.
    *   Outputs the sequence of token IDs.

*(Future functionality may include detokenization: converting token IDs back to text.)*

## Features

*   **Corpus-Driven BPE Training:** Learns merge rules directly from user-provided text data.
*   **Byte-Level and Word-Level Modes:**
    *   **Byte-Level (`--mode byte`):** (Recommended for general use) Starts with individual bytes. Non-printable/non-ASCII bytes are represented as hexadecimal escape sequences (e.g., `\x0A`). Robust for diverse text and code.
    *   **Word-Level (`--mode word`):** Starts with space-separated words.
*   **Configurable Target Vocabulary Size (`--vocab-size`):** For training.
*   **Efficient Tokenization:** Applies learned merges systematically to new text.
*   **Structured Model Output (`[prefix_]bpe_model_lunaris.json`):**
    *   `vocabulary_map`: A JSON object mapping token strings to their integer IDs (e.g., `{"hello": 273, " world": 451}`).
    *   `merges`: An ordered list of the learned merge pairs (original, non-escaped token strings).
    *   `bpe_config`: Contains training configuration, like the `mode`.
    *   `stats`: Key statistics from the training process.
*   **Plain Text Outputs (for inspection):**
    *   `[prefix_]vocabulary_lunaris.txt`: Lists each token and its ID.
    *   `[prefix_]merges_lunaris.txt`: Lists the ordered merge operations.
*   **Verbose Mode (`--verbose`):** For detailed logging during training and tokenization.
*   **Model Loading & Usage:** Loads trained BPE models for tokenization.
*   **Efficient C++ Implementation.**

## Prerequisites

*   A C++17 compatible compiler (e.g., `g++`, `clang++`).
*   Standard C++ libraries, including `std::filesystem`.
*   [nlohmann/json.hpp](https://github.com/nlohmann/json): Header-only JSON library. Ensure `json.hpp` is accessible to your compiler (e.g., in the `bpe_trainer/` directory or an include path).

## Compilation

The primary method to compile the `bpe_processor` executable is using the main `Makefile` in the root of the Lunaris Codex project:
```bash
# From the root directory of the Lunaris Codex project
make bpe_processor 
# Or to build all C++ utilities:
# make all
```
This will typically place the compiled executable at `bpe_trainer/bpe_processor` (assuming the source file `bpe_processor.cpp` is in the `bpe_trainer/` directory).

For manual compilation (e.g., if `nlohmann/json.hpp` is in the current `bpe_trainer/` directory and your source is `bpe_processor.cpp`):
```bash
cd bpe_trainer/ 
g++ bpe_processor.cpp -o bpe_processor -std=c++17 -O2 -I. -lstdc++fs
```

## Usage

Run the compiled `bpe_processor` executable from your terminal. It's often convenient to run it from the root of the Lunaris Codex project, adjusting paths accordingly.

```bash
./bpe_trainer/bpe_processor --action <train|tokenize> [options...]
```

### Command-Line Arguments:

*   `--action ACTION`: **(Required)** Specifies the operation:
    *   `train`: Trains a new BPE model.
    *   `tokenize`: Loads a trained model and tokenizes input text.
*   `--corpus FILE_PATH`: **(Required for `--action train`)** Path to the input corpus text file.
*   `--output PATH_OR_PREFIX`: (Optional for `--action train`) Path for saving output model files.
    *   If `PATH_OR_PREFIX` ends with a directory separator (`/` or `\`), it's treated as a directory. Files (e.g., `bpe_model_lunaris.json`) are saved inside with default names.
    *   If it does not end with a separator (e.g., `./my_models/custom_bpe`), it's treated as a prefix. Files will be named like `custom_bpe_bpe_model_lunaris.json`.
    *   Default: Creates/uses a directory `./bpe_lunaris_model/`.
*   `--vocab-size N`: (Optional for `--action train`) Target vocabulary size. Default: `32000`.
*   `--mode LEVEL`: (Optional for `--action train`) Initial tokenization mode: `byte` or `word`. Default: `byte`.
*   `--model_path PATH`: **(Required for `--action tokenize`)** Path to the trained BPE model. This can be the path to the directory containing `bpe_model_lunaris.json`, the path to the `bpe_model_lunaris.json` file itself, or the prefix used during training if files were saved with a prefix.
*   `--input_text "TEXT"`: (Required for `--action tokenize` if `--input_file` is not used) Text string to tokenize.
*   `--input_file FILE_PATH`: (Required for `--action tokenize` if `--input_text` is not used) Path to a file whose content will be tokenized.
*   `--verbose`: (Optional) Enable verbose logging.
*   `-h`, `--help`: (Optional) Display the help message.

### Example 1: Training a Byte-Level BPE Model

```bash
# Create corpus.txt first
# echo "hello world of BPE tokenizers" > corpus.txt
# echo "another line for the BPE trainer" >> corpus.txt

./bpe_trainer/bpe_processor \
    --action train \
    --corpus ./corpus.txt \
    --output ./my_custom_bpe_model/ \
    --vocab-size 500 \
    --mode byte \
    --verbose
```
Output: Saves model files to `./my_custom_bpe_model/`.

### Example 2: Tokenizing Text with a Trained Model

```bash
./bpe_trainer/bpe_processor \
    --action tokenize \
    --model_path ./my_custom_bpe_model/ \
    --input_text "This is a test sentence to tokenize with BPE." \
    --verbose
```
Output: Prints a sequence of token IDs to standard output.

## Output Files (from Training - `--action train`)

1.  **`[prefix_]bpe_model_lunaris.json`**: Contains `vocabulary_map` (token string to ID), ordered `merges`, `bpe_config`, and training `stats`. **This is the primary file needed for tokenization.**
2.  **`[prefix_]vocabulary_lunaris.txt`**: Human-readable list of vocabulary tokens (escaped for display), where the line number (0-indexed) corresponds to the ID.
3.  **`[prefix_]merges_lunaris.txt`**: Human-readable list of ordered merge operations (escaped for display).

## How BPE Works (Briefly)

1.  **Initialization:** Corpus is split into initial tokens (bytes or words).
2.  **Iteration:** Repeatedly find the most frequent adjacent pair, merge it into a new token, add to vocab, and update the corpus.
3.  **Termination:** Stops when target vocab size is reached or no more frequent pairs exist.

## Roadmap & Future Possibilities

*   **Implement BPE Decoder (`--action detokenize`):** Convert token IDs back to text.
*   **Full Integration with Lunaris Codex `prepare_data.py`:** Allow `prepare_data.py` to use BPE models from this tool.
*   **Enhanced Special Token Handling:** More sophisticated management of `[UNK]`, `[PAD]`, etc.
*   **Advanced Pre-tokenization Rules.**
*   **Further Performance Optimizations.**

## Contributing
Contributions are welcome! Please see the main Lunaris Codex [CONTRIBUTING.md](https://github.com/MeryylleA/lunariscodex/blob/main/CONTRIBUTING.md).
