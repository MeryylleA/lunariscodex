# Lunaris BPE Tokenizer Trainer (bpe_trainer)

**Version: 0.1.0**

The `Lunaris BPE Tokenizer Trainer` is a command-line utility written in C++ designed to learn Byte Pair Encoding (BPE) merges from a given text corpus. It generates a vocabulary and a list of merge operations that can be used to build a BPE-based subword tokenizer.

This tool is a core component for users of the [Lunaris Codex](https://github.com/MeryylleA/lunariscodex) project who wish to create custom tokenizers tailored to their specific datasets, especially for tasks like code generation or specialized language modeling.

## Purpose

The primary goal of this tool is **not** to be a full-fledged tokenizer itself (i.e., it doesn't take arbitrary text and output token IDs directly in this version). Instead, it focuses on the **training phase** of BPE:
1.  Analyzing a raw text corpus.
2.  Starting with a base vocabulary (either individual bytes or pre-segmented words).
3.  Iteratively finding the most frequent pair of adjacent tokens and merging them into a new, single token.
4.  Repeating this process until a target vocabulary size is reached or no more frequent pairs can be merged.
5.  Saving the learned **vocabulary** and the **ordered list of merge operations**.

These outputs (vocabulary and merges) are the essential components required by a BPE *encoder/decoder* to tokenize new text and detokenize token IDs back into text.

## Features

*   **Corpus-Driven BPE Training:** Learns merge rules directly from your provided text data.
*   **Two Pre-tokenization Modes:**
    *   **Byte-Level (`--mode byte`):** Starts with individual bytes as initial tokens. Bytes are represented as their ASCII characters if printable, or as safe hexadecimal escape sequences (e.g., `\x0A` for newline, `\xC3` for non-ASCII bytes). This mode is generally recommended for building robust subword tokenizers for diverse text and code.
    *   **Word-Level (`--mode word`):** Starts with space-separated "words" as initial tokens. BPE then learns to merge parts of these words or entire words. Simpler for understanding BPE concepts on smaller, well-behaved text.
*   **Configurable Target Vocabulary Size:** You can specify the desired final size of your vocabulary via `--vocab-size`.
*   **Detailed Statistics:** Reports on raw corpus characters, initial token count, initial/final vocabulary sizes, total merges, and training time.
*   **Verbose Mode:** Optional `--verbose` flag for detailed logging of each merge operation and progress.
*   **Structured Output:**
    *   Saves a `bpe_model_lunaris.json` file containing the final vocabulary, the learned merge operations, training mode, and key statistics.
    *   Also saves `vocabulary_lunaris.txt` and `merges_lunaris.txt` in human-readable plain text formats.
*   **Efficient C++ Implementation:** Designed for performance when processing text corpora.

## Prerequisites

*   A C++17 compatible compiler (e.g., `g++`, `clang++`).
*   Standard C++ libraries.
*   [nlohmann/json.hpp](https://github.com/nlohmann/json): A header-only JSON library for C++. You'll need to place `json.hpp` (usually found in `single_include/nlohmann/`) in a location your compiler can find it (e.g., in the same directory as `bpe_trainer.cpp`, or in an include path).
*   `std::filesystem` support (C++17).

## Compilation

Navigate to the `bpe_trainer/` directory (or wherever `bpe_trainer.cpp` and `json.hpp` are located) and compile using a C++17 compiler:

```bash
# Ensure nlohmann/json.hpp is accessible (e.g., in the current dir or via -I)
# If json.hpp is in the same directory:
g++ bpe_trainer.cpp -o bpe_trainer -std=c++17 -O2 -I.

# If json.hpp is in ./include/nlohmann/json.hpp:
# g++ bpe_trainer.cpp -o bpe_trainer -std=c++17 -O2 -I./include
```
*The `-O2` flag enables optimizations. `-I.` tells the compiler to look for includes in the current directory.*

On some older Linux systems, you might need to link the filesystem library if you encounter linker errors related to `std::filesystem`:
```bash
g++ bpe_trainer.cpp -o bpe_trainer -std=c++17 -O2 -I. -lstdc++fs
```

## Usage

Run the compiled `bpe_trainer` executable from your terminal.

```bash
./bpe_trainer --corpus <path_to_corpus.txt> [options...]
```

### Command-Line Arguments:

*   `--corpus FILE`: **(Required)** Path to the input corpus text file. Each line in the file is processed.
*   `--output DIR`: (Optional) Output directory for model files (vocabulary, merges, JSON).
    *   Default: `./bpe_lunaris_model`
*   `--vocab-size N`: (Optional) Target vocabulary size. The training will perform merges until this size is approached or no more beneficial merges can befound.
    *   Default: `32000`
*   `--mode LEVEL`: (Optional) Initial tokenization mode.
    *   `byte`: Initial tokens are individual bytes. Non-printable/non-ASCII bytes are represented as hex escapes (e.g., `\x0A`).
    *   `word`: Initial tokens are space-separated words from the corpus.
    *   Default: `byte`
*   `--verbose`: (Optional) Enable verbose logging output during training, showing each merge and progress.
*   `-h`, `--help`: (Optional) Display the help message and exit.

### Example: Training a Byte-Level BPE Model

```bash
# Create a sample corpus file (e.g., corpus.txt)
# echo "hello world" > corpus.txt
# echo "hello new world" >> corpus.txt

./bpe_trainer \
    --corpus ./corpus.txt \
    --output ./my_custom_bpe_model \
    --vocab-size 500 \
    --mode byte \
    --verbose
```
This will:
1.  Read `corpus.txt`.
2.  Initialize vocabulary with unique bytes (represented safely) found in `corpus.txt` and standard ASCII.
3.  Perform BPE merges until the vocabulary size approaches 500 or no more merges are beneficial.
4.  Save `bpe_model_lunaris.json`, `vocabulary_lunaris.txt`, and `merges_lunaris.txt` into the `./my_custom_bpe_model/` directory.

## Output Files

The training process generates the following files in the specified output directory:

1.  **`bpe_model_lunaris.json`**: A JSON file containing:
    *   `vocabulary`: A sorted list of all tokens in the final vocabulary.
    *   `merges`: An ordered list of pairs that were merged during training. Each sub-array is `["token1", "token2"]`. The order is important for correctly applying BPE during tokenization.
    *   `bpe_config`: Contains training configuration like `{"mode": "byte"}`.
    *   `stats`: Key statistics from the training process (corpus size, vocab sizes, number of merges, training time).
2.  **`vocabulary_lunaris.txt`**: A plain text file listing each token in the final vocabulary on a new line, sorted alphabetically. Non-printable/non-ASCII bytes are represented as hex escapes.
3.  **`merges_lunaris.txt`**: A plain text file listing the ordered merge operations, one pair per line, formatted as `token1 token2`.

## How BPE Works (Briefly)

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

This `bpe_trainer` is the foundational step. Future development could include:

*   **BPE Tokenizer/Encoder:** A companion tool or library function that takes the `vocabulary.txt` and `merges.txt` (or `bpe_model_lunaris.json`) generated by this trainer and uses them to tokenize new, unseen text into a sequence of token strings or IDs.
*   **BPE Decoder:** A function to convert a sequence of token strings/IDs back into human-readable text.
*   **Integration with Lunaris Codex `prepare_data.py`:** Allow `prepare_data.py` to use a custom BPE model trained by this tool.
*   **Support for Special Tokens:** Handling pre-defined special tokens (like `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[BOS]`, `[EOS]`) during training or allowing them to be added to the vocabulary.
*   **Pre-tokenization Rules:** More sophisticated pre-tokenization rules for `word-level` mode (e.g., handling punctuation better than simple space splitting, regex-based splitting).
*   **Performance Optimizations:** For extremely large corpora, explore more advanced data structures or parallelization for pair counting and merging.
*   **Caching of Pair Counts:** To speed up re-training or a_paramount_ (paramount) iterative training.

## Contributing
Contributions are welcome! If you have ideas for improvements, new features, or find any bugs, please open an issue or a pull request on the [Lunaris Codex GitHub repository](https://github.com/MeryylleA/lunariscodex).
