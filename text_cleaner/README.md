**Version: 0.3.1**

`lunaris_text_cleaner` is a C++ command-line utility designed to perform various cleaning and normalization operations on text files. It can process individual files or entire directories (recursively), preparing them for further processing steps, such as tokenization for training language models like Lunaris Codex.

It aims to provide a fast and efficient way to improve dataset quality by handling common text issues directly in C++, and it reports the total processing time.

## Features

-   **Flexible Input:** Processes a single text file or all matching files within a directory (with optional recursive search).
-   **Whitespace Normalization:**
    -   Trims leading and trailing whitespace from each line.
    -   Reduces multiple internal spaces/tabs to a single space.
-   **Empty Line Removal:** Optionally removes lines that become entirely empty *after* whitespace normalization.
-   **Case Conversion:** Optionally converts all text to lowercase.
-   **Non-Printable Character Removal:** Removes common non-printable ASCII characters while preserving essential whitespace like tabs (`\t`), newlines (`\n`), and carriage returns (`\r`).
-   **URL Processing:**
    -   Detects and removes/replaces common URL patterns.
    -   Uses a user-defined placeholder string (e.g., `"[URL]"`) if provided.
-   **Email Processing:**
    -   Detects and removes/replaces common email address patterns.
    -   Uses a user-defined placeholder string (e.g., `"[EMAIL]"`) if provided.
-   **Exact Duplicate Line Removal:** After all other selected processing steps are applied to a line, this option can remove lines that are exact duplicates of previously processed and written lines (on a per-file basis if processing multiple files).
-   **Directory Structure Replication:** When processing directories, the input's subdirectory structure is replicated in the output directory.
-   **File Handling:** Reads from input and writes cleaned content to new output files, creating parent directories for output files if they don't exist.
-   **Performance Indication:** Reports total execution time.
-   **Configurable:** All operations are controlled via command-line arguments.

## Prerequisites

-   A C++17 compatible compiler (e.g., `g++`).
-   Standard C++ libraries (iostream, fstream, string, vector, algorithm, cctype, filesystem, regex, set, iomanip, sstream, chrono).
-   `std::filesystem` support is part of C++17.

## Compilation

Navigate to the directory containing `lunaris_text_cleaner.cpp` and compile using g++ (or your preferred C++17 compiler):

```bash
g++ lunaris_text_cleaner.cpp -o lunaris_text_cleaner -std=c++17
```
On some older Linux systems or specific toolchains, you *might* need to link the filesystem library explicitly if you encounter linker errors related to `std::filesystem`:
```bash
g++ lunaris_text_cleaner.cpp -o lunaris_text_cleaner -std=c++17 -lstdc++fs
```

## Usage

Run the compiled executable from your terminal.

**For a single file:**
```bash
./lunaris_text_cleaner --input <input_file.txt> --output <cleaned_file.txt> [options...]
```

**For all files in a directory (matching a pattern):**
```bash
./lunaris_text_cleaner --input <input_directory/> --output <output_directory/> [--input-pattern "*.log"] [--recursive] [options...]
```

### Command-Line Arguments:

*   `--input <path>`: **(Required)** Path to the input text file or source directory.
*   `--output <path>`: **(Required)** Path to the output text file or base output directory. If input is a directory, output must also be a directory (will be created if it doesn't exist).
*   `--input-pattern <glob>`: (Optional) Glob-like pattern for files if `--input` is a directory (e.g., `"*.txt"`, `"data_*"`, `"*.*"`). Default: `"*.txt"`.
    *Current simple pattern matching supports: `*`, `*.*`, `*.ext`, `prefix_*`, `*_suffix`, `prefix_*_suffix` and exact names.*
*   `--recursive`: (Optional) If `--input` is a directory, search for files recursively in subdirectories.
*   `--normalize-whitespace`: (Optional) Enable trimming and reduction of multiple whitespaces.
*   `--remove-empty-lines`: (Optional) Remove lines that become empty *after* whitespace normalization (requires `--normalize-whitespace`).
*   `--to-lowercase`: (Optional) Convert all text to lowercase.
*   `--remove-non-printable`: (Optional) Remove non-printable ASCII characters (preserves tab, newline, CR).
*   `--process-urls`: (Optional) Enable processing of URLs. If `--url-placeholder` is empty, URLs are removed.
*   `--url-placeholder <str>`: (Optional) String to replace detected URLs with. Effective if `--process-urls` is set.
*   `--process-emails`: (Optional) Enable processing of email addresses. If `--email-placeholder` is empty, emails are removed.
*   `--email-placeholder <str>`: (Optional) String to replace detected email addresses with. Effective if `--process-emails` is set.
*   `--remove-exact-duplicates`: (Optional) Remove lines that are exact duplicates (after all other enabled per-line cleaning).
*   `-h`, `--help`: Display the help message and exit.

### Example: Cleaning all `.txt` files in a directory recursively

```bash
./lunaris_text_cleaner \
    --input ./raw_corpus_dir \
    --output ./cleaned_corpus_dir \
    --input-pattern "*.txt" \
    --recursive \
    --normalize-whitespace \
    --remove-empty-lines \
    --to-lowercase \
    --process-urls --url-placeholder "[URL]" \
    --process-emails --email-placeholder "[EMAIL]" \
    --remove-exact-duplicates
```
This command will:
1. Search for all `.txt` files in `./raw_corpus_dir` and its subdirectories.
2. For each file found:
    a. Apply URL replacement (with `[URL]`).
    b. Apply email replacement (with `[EMAIL]`).
    c. Normalize whitespace.
    d. Convert to lowercase.
    e. Remove lines that became empty.
    f. Remove exact duplicate lines within that file.
3. Save the cleaned files to `./cleaned_corpus_dir`, preserving the subdirectory structure from `raw_corpus_dir`.
4. Report overall statistics and total processing time.

## Order of Operations (per line)

The cleaning steps are applied in the following order for each line:
1.  Non-printable character removal.
2.  URL processing.
3.  Email processing.
4.  Whitespace normalization.
5.  Conversion to lowercase.
6.  Check/removal of (now possibly) empty lines.
7.  Check/removal of exact duplicates (based on the fully processed line content for that file).

## Future Considerations / Potential Improvements

-   More sophisticated/robust glob pattern matching for `--input-pattern`.
-   More advanced URL and email detection regex.
-   Options for Unicode normalization (e.g., NFC, NFKC) or handling specific Unicode character blocks.
-   Removal of near-duplicates (e.g., using MinHash or SimHash).
-   Language-specific cleaning rules (e.g., for code comments, though this is complex).
-   Performance optimizations for extremely large files when not using directory iteration (e.g., chunked reading within `process_single_file` if it were to handle truly massive single files).
-   Option to specify padding token ID for `lunaris_data_analyzer` (this is for the other tool, just a note).

---
