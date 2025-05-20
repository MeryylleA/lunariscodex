<!-- README for bpe_trainer - Reviewed and Updated by Lunaris Codex Assistant (Gemini 2.5 Pro) - May 20, 2025 -->
<!-- Based on PR #36, the following sections were updated: -->
<!-- - Purpose and Functionality: To reflect upcoming tokenization. -->
<!-- - Command-Line Arguments: To include --action and tokenize options. -->
<!-- - Output Files: To describe the new vocabulary_map format. -->
<!-- - Version bumped to 0.1.1. -->

# Lunaris BPE Processor (bpe_trainer)

**Version: 0.1.1**

The `Lunaris BPE Processor` (executable: `bpe_trainer`) is a command-line utility written in C++ designed to learn Byte Pair Encoding (BPE) merges from a given text corpus and, in upcoming versions, to tokenize text using a trained BPE model. It generates a vocabulary map (token to ID) and an ordered list of merge operations.

This tool is a core component for users of the [Lunaris Codex](https://github.com/MeryylleA/lunariscodex) project who wish to create and eventually use custom tokenizers tailored to their specific datasets, especially for tasks like code generation or specialized language modeling.

## Purpose and Functionality

The tool currently focuses on the **training phase** of BPE and is being actively extended to include **tokenization capabilities**.

**Current Functionality (`--action train`):**
1.  Analyzes a raw text corpus.
2.  Initializes a vocabulary based on the chosen mode (byte-level or word-level).
3.  Iteratively identifies the most frequent pair of adjacent tokens and merges them.
4.  Saves the learned **vocabulary map** (mapping token strings to integer IDs) and the **ordered list of merge operations** in a primary JSON file, alongside human-readable plain text versions.

**Upcoming Functionality (`--action tokenize` - In Development):**
*   Load a pre-trained BPE model (its vocabulary map and merge rules).
*   Tokenize new input text into a sequence of corresponding token IDs.
*   (Future) Detokenize a sequence of token IDs back into human-readable text.

The outputs from the training phase are the essential components required by any BPE encoder/decoder.

## Features

*   **Corpus-Driven BPE Training:** Learns merge rules directly from your provided text data.
*   **Two Pre-tokenization Modes for Training (`--mode byte` or `--mode word`):**
    *   **Byte-Level:** (Recommended for general use) Starts with individual bytes as initial tokens. Non-printable/non-ASCII bytes are represented as safe hexadecimal escape sequences (e.g., `\x0A` for newline). This mode is robust for diverse text and code.
    *   **Word-Level:** Starts with space-separated "words" as initial tokens. BPE then learns to merge parts of these words.
*   **Configurable Target Vocabulary Size (`--vocab-size`):** Allows specification of the desired final vocabulary size during training.
*   **Structured Model Output (`*_bpe_model_lunaris.json`):**
    *   `vocabulary_map`: A JSON object mapping token strings to their integer IDs (e.g., `{"hello": 256, " world": 257}`).
    *   `merges`: An ordered list of the learned merge pairs, stored with their original (non-escaped) token strings.
    *   `bpe_config`: Contains training configuration details, such as the `mode` used (e.g., "byte").
    *   `stats`: Key statistics from the training process.
*   **Plain Text Outputs:**
    *   `*_vocabulary_lunaris.txt`: A plain text file listing each token from the final vocabulary (escaped for display) on a new line. The line number (0-indexed) corresponds to the token's ID.
    *   `*_merges_lunaris.txt`: A plain text file listing the ordered merge operations (tokens escaped for display), one pair per line.
*   **Detailed Statistics:** Reports on raw corpus characters, token counts, vocabulary sizes, total merges, and training time.
*   **Verbose Mode (`--verbose`):** Optional flag for detailed logging of each merge operation and progress.
*   **Model Loading Capability:** The processor can now load a trained BPE model from its JSON file in preparation for tokenization.
*   **Efficient C++ Implementation:** Designed for performance.

## Prerequisites

*   A C++17 compatible compiler (e.g., `g++`, `clang++`).
*   Standard C++ libraries, including support for `std::filesystem`.
*   [nlohmann/json.hpp](https://github.com/nlohmann/json): A header-only JSON library for C++. This needs to be accessible by your compiler (e.g., placed in the `bpe_trainer/` directory, or in a system include path).

## Compilation

The primary and recommended way to compile the `bpe_trainer` executable is by using the main `Makefile` located in the root directory of the Lunaris Codex project:
```bash
# From the root directory of the Lunaris Codex project
make bpe_trainer
```
This command will typically place the compiled executable at `bpe_trainer/bpe_trainer`. You can also use `make all` to build all C++ utilities.

For manual compilation (if, for instance, `nlohmann/json.hpp` is in the current `bpe_trainer/` directory):
```bash
cd bpe_trainer/
g++ bpe_trainer.cpp -o bpe_trainer -std=c++17 -O2 -I. -lstdc++fs
```
*   The `-O2` flag enables optimizations.
*   `-I.` tells the compiler to look for include files in the current directory.
*   `-lstdc++fs` links the filesystem library, which might be necessary on some Linux systems.

## Usage

Run the compiled `bpe_trainer` executable from your terminal, typically from the root of the Lunaris Codex project.

```bash
./bpe_trainer/bpe_trainer --action <train|tokenize> [options...]
```

### Command-Line Arguments:

*   `--action ACTION`: **(Required)** Specifies the operation:
    *   `train`: Trains a new BPE model.
    *   `tokenize`: (Development in progress) Loads a trained model to tokenize text.
*   `--corpus FILE_PATH`: **(Required for `--action train`)** Path to the input corpus text file.
*   `--output PATH_OR_PREFIX`: (Optional for `--action train`) Path for saving output model files.
    *   If `PATH_OR_PREFIX` ends with a directory separator (`/` or `\`), it's treated as a directory. Files (e.g., `bpe_model_lunaris.json`) are saved inside with default names.
    *   If `PATH_OR_PREFIX` does not end with a separator (e.g., `./my_models/custom_bpe`), it's treated as a prefix. Files will be named like `custom_bpe_bpe_model_lunaris.json`, etc., in the `./my_models/` directory.
    *   Default: Creates a directory `./bpe_lunaris_model/` and saves files with default names.
*   `--vocab-size N`: (Optional for `--action train`) Target vocabulary size. Default: `32000`.
*   `--mode LEVEL`: (Optional for `--action train`) Initial tokenization mode: `byte` or `word`. Default: `byte`.
*   `--model_path PATH_OR_PREFIX`: **(Required for `--action tokenize`)** Path to the trained BPE model directory or file prefix (location of the `*_bpe_model_lunaris.json` file).
*   `--input_text "TEXT"`: (Optional for `--action tokenize`) Text string to tokenize directly.
*   `--input_file FILE_PATH`: (Optional for `--action tokenize`) Path to a file whose content will be tokenized. Use either `--input_text` or `--input_file` for tokenization.
*   `--verbose`: (Optional) Enable verbose logging output.
*   `-h`, `--help`: (Optional) Display the help message and exit.

### Example 1: Training a Byte-Level BPE Model

```bash
# Ensure corpus.txt exists
# Example: echo "hello world of BPE tokenizers" > corpus.txt
#          echo "another line for the BPE trainer" >> corpus.txt

./bpe_trainer/bpe_trainer \
    --action train \
    --corpus ./corpus.txt \
    --output ./my_custom_bpe_model/ \
    --vocab-size 500 \
    --mode byte \
    --verbose
```
This command will:
1. Read `corpus.txt`.
2. Perform BPE training in byte-level mode.
3. Aim for a vocabulary size up to 500.
4. Save `bpe_model_lunaris.json`, `vocabulary_lunaris.txt`, and `merges_lunaris.txt` into the `./my_custom_bpe_model/` directory.

### Example 2: Loading a Trained Model (Tokenization - Placeholder Output)

```bash
./bpe_trainer/bpe_trainer \
    --action tokenize \
    --model_path ./my_custom_bpe_model/ \
    --input_text "This is a test sentence to tokenize." \
    --verbose
```
This will load the BPE model from `./my_custom_bpe_model/` and (currently) print an informational message. Future versions will output token IDs.

## Output Files (from Training - `--action train`)

The training process generates the following files in the location specified by `--output`:

1.  **`[prefix_]bpe_model_lunaris.json`**: A JSON file containing:
    *   `vocabulary_map`: A JSON object mapping token strings to their integer IDs (e.g., `{"hello": 273, " world": 451}`).
    *   `merges`: An ordered list of pairs (original, non-escaped token strings) that were merged during training. The order is crucial for tokenization.
    *   `bpe_config`: Contains training configuration, like `{"mode": "byte"}`.
    *   `stats`: Key statistics from the training process.
2.  **`[prefix_]vocabulary_lunaris.txt`**: A plain text file listing each token from the final vocabulary (escaped for display) on a new line. The line number (0-indexed) corresponds to the token's ID. This file is useful for human inspection.
3.  **`[prefix_]merges_lunaris.txt`**: A plain text file listing the ordered merge operations (tokens escaped for display), one pair per line (e.g., `h e`). This file is useful for human inspection.

## How BPE Works (Briefly)
<!-- This section can remain largely the same as your current README -->
<!-- ... (brief explanation of BPE) ... -->

1.  **Initialization:** The corpus is split into an initial sequence of tokens.
    *   In **byte-level** mode, each byte becomes a token.
    *   In **word-level** mode, text is split by spaces, and each resulting "word" is an initial token. The vocabulary starts with all unique initial tokens.
2.  **Iteration:** The algorithm repeatedly performs the following steps:
    a.  Count the frequency of all adjacent pairs of tokens in the current representation of the corpus.
    b.  Find the pair that occurs most frequently.
    c.  Merge this pair into a new, single token (e.g., if `('t', 'h')` is most frequent, it becomes `'th'`).
    d.  Add this new token to the vocabulary.
    e.  Record this merge operation.
    f.  Update all occurrences of the pair in the corpus with the new merged token.
3.  **Termination:** The process stops when the desired vocabulary size is reached or when no more pairs can be merged (e.g., all remaining pairs occur only once).

## Roadmap & Future Possibilities
<!-- Reflecting current status -->
*   **Implement BPE Tokenizer/Encoder (`--action tokenize`):** (Actively In Progress) Fully implement the tokenization logic within this tool to convert new text into a sequence of token IDs using a loaded, trained model.
*   **Implement BPE Decoder (`--action detokenize`):** Add functionality to convert a sequence of token IDs back into human-readable text.
*   **Integration with Lunaris Codex `prepare_data.py`:** Allow `prepare_data.py` to natively use custom BPE models trained and applied by this tool.
*   **Support for Special Tokens:** Define strategies for handling pre-defined special tokens (like `[UNK]`, `[PAD]`, `[BOS]`, `[EOS]`) during training and tokenization.
*   **Advanced Pre-tokenization Rules:** For `word-level` mode, explore more sophisticated pre-tokenization (e.g., regex-based splitting, better punctuation handling).
*   **Performance Optimizations:** For extremely large corpora, investigate further optimizations.

## Contributing
Contributions are welcome! If you have ideas for improvements, new features, or find any bugs, please open an issue or a pull request on the [Lunaris Codex GitHub repository](https://github.com/MeryylleA/lunariscodex).
