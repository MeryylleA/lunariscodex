# Lunaris Text Cleaner

**Version: 0.3.7**

`lunaris_text_cleaner` is a C++ command-line utility designed to perform various cleaning and normalization operations on text files. It can process individual files or entire directories (recursively), preparing raw text data for further processing steps, such as tokenization for training language models like Lunaris Codex.

This tool aims to provide a fast and efficient C++ solution for common text dataset quality improvements, reporting detailed statistics and total processing time.

## Features

*   **Flexible Input:** Processes a single text file or all matching files within a directory (with optional recursive search).
*   **Robust File Handling:**
    *   Reads entire file content into memory for processing, with checks for very large files and memory allocation issues.
    *   Creates output directories if they don't exist.
    *   When processing directories, the input's subdirectory structure is replicated in the output directory.
*   **HTML Cleaning (`--remove-html`):**
    *   Removes `<!DOCTYPE ...>` declarations.
    *   Removes HTML comments (`<!-- ... -->`).
    *   Removes entire `<script>...</script>` blocks (tags and content).
    *   Removes entire `<style>...</style>` blocks (tags and content).
    *   Removes general HTML/XML tags (e.g., `<p>`, `<div>`, `<br/>`).
        *   *See "Limitations" section for details on HTML parsing.*
*   **Whitespace Normalization (`--normalize-whitespace`):**
    *   Trims leading and trailing whitespace from each line.
    *   Reduces multiple internal spaces/tabs to a single space.
*   **Empty Line Removal (`--remove-empty-lines`):** Optionally removes lines that become entirely empty *after* HTML cleaning and whitespace normalization.
*   **Case Conversion (`--to-lowercase`):** Optionally converts all text to lowercase.
*   **Non-Printable Character Removal (`--remove-non-printable`):** Removes non-printable ASCII control characters (bytes 0-31, excluding Tab, LF, CR, and byte 127 DEL). Preserves essential whitespace like tabs (`\t`), newlines (`\n`), and carriage returns (`\r`), and standard printable ASCII characters (32-126).
    *   *Note: This removes actual control character bytes, not their textual escape sequences like `"\x01"`.*
*   **URL Processing (`--process-urls`):**
    *   Detects and removes/replaces common URL patterns.
    *   Uses a user-defined placeholder string (e.g., `"[URL]"`) via `--url-placeholder`.
*   **Email Processing (`--process-emails`):**
    *   Detects and removes/replaces common email address patterns.
    *   Uses a user-defined placeholder string (e.g., `"[EMAIL]"`) via `--email-placeholder`.
*   **Exact Duplicate Line Removal (`--remove-exact-duplicates`):** After all other selected processing steps are applied to a line, this option can remove lines that are exact duplicates of previously processed and written lines (on a per-file basis).
*   **Glob-like Pattern Matching (`--input-pattern`):**
    *   Supports basic wildcards (`*`, `?`) for filename matching.
    *   Handles common cases like `*.txt`, `file.*`, `file?.log`.
    *   Escapes common regex metacharacters within the pattern for more intuitive glob behavior (e.g., `file.name*` matches a literal dot).
*   **Performance Indication:** Reports total execution time and various processing statistics.
*   **Configurable:** All operations are controlled via command-line arguments.

## Prerequisites

*   A C++17 compatible compiler (e.g., `g++`, `clang++`).
*   Standard C++ libraries (iostream, fstream, string, vector, algorithm, cctype, filesystem, regex, set, iomanip, sstream, chrono, limits, cstdint).
*   `std::filesystem` support is part of C++17.

## Compilation

Navigate to the directory containing `lunaris_text_cleaner.cpp` and compile using a C++17 compiler:

```bash
g++ lunaris_text_cleaner.cpp -o lunaris_text_cleaner -std=c++17 -O2
```
*The `-O2` flag enables optimizations, which is recommended for performance.*

On some older Linux systems or specific toolchains, you *might* need to link the filesystem library explicitly if you encounter linker errors related to `std::filesystem`:
```bash
g++ lunaris_text_cleaner.cpp -o lunaris_text_cleaner -std=c++17 -O2 -lstdc++fs
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
*   `--input-pattern <glob>`: (Optional) Glob-like pattern for files if `--input` is a directory (e.g., `"*.txt"`, `"data_?"`, `"file.specific.log"`). Default: `"*.txt"`.
*   `--recursive`: (Optional) If `--input` is a directory, search for files recursively in subdirectories.
*   `--normalize-whitespace`: (Optional) Enable trimming and reduction of multiple whitespaces.
*   `--remove-empty-lines`: (Optional) Remove lines that become empty *after* HTML cleaning and whitespace normalization (requires `--normalize-whitespace`).
*   `--to-lowercase`: (Optional) Convert all text to lowercase.
*   `--remove-non-printable`: (Optional) Remove non-printable ASCII characters (preserves tab, newline, CR, and standard printable ASCII).
*   `--remove-html`: (Optional) Remove DOCTYPE, HTML/XML comments, script/style blocks (tags and content), and other general HTML/XML tags.
*   `--process-urls`: (Optional) Enable processing of URLs. If `--url-placeholder` is empty, URLs are removed.
*   `--url-placeholder <str>`: (Optional) String to replace detected URLs with. Effective if `--process-urls` is set.
*   `--process-emails`: (Optional) Enable processing of email addresses. If `--email-placeholder` is empty, emails are removed.
*   `--email-placeholder <str>`: (Optional) String to replace detected email addresses with. Effective if `--process-emails` is set.
*   `--remove-exact-duplicates`: (Optional) Remove lines that are exact duplicates (after all other enabled per-line cleaning).
*   `-h`, `--help`: Display the help message and exit.

### Example: Comprehensive cleaning of `.txt` files

```bash
./lunaris_text_cleaner \
    --input ./raw_data_dir \
    --output ./cleaned_data_dir \
    --input-pattern "*.txt" \
    --recursive \
    --remove-html \
    --normalize-whitespace \
    --remove-empty-lines \
    --to-lowercase \
    --remove-non-printable \
    --process-urls --url-placeholder "[URL]" \
    --process-emails --email-placeholder "[EMAIL]" \
    --remove-exact-duplicates
```
This command will:
1.  Search for all files (recursively) in `./raw_data_dir` that end with `.txt`.
2.  For each file found, apply the full suite of cleaning operations.
3.  Save the cleaned files to `./cleaned_data_dir`, preserving the subdirectory structure.
4.  Report overall statistics and total processing time.

## Order of Operations

The cleaning steps are applied in the following order for each processed file:

1.  **File Reading:** The entire file is read into memory.
2.  **Global HTML Cleaning (if `--remove-html` is enabled, applied to the entire file content):**
    1.  DOCTYPE removal (`<!DOCTYPE ...>`)
    2.  HTML comment removal (`<!-- ... -->`)
    3.  Script block removal (`<script>...</script>`, including content)
    4.  Style block removal (`<style>...</style>`, including content)
    5.  General HTML/XML tag removal (e.g., `<p>`, `<div>`)
3.  **Per-line operations (applied sequentially to each line from the content processed above):**
    1.  Non-printable character removal (`--remove-non-printable`)
    2.  URL processing (`--process-urls`)
    3.  Email processing (`--process-emails`)
    4.  Whitespace normalization (`--normalize-whitespace`)
    5.  Conversion to lowercase (`--to-lowercase`)
    6.  Check/removal of (now possibly) empty lines (`--remove-empty-lines`, if whitespace normalization is also on)
    7.  Check/removal of exact duplicates (`--remove-exact-duplicates`)
4.  **Writing to Output:** Processed lines are written to the corresponding output file.

## Limitations and Known Issues

*   **HTML Parsing with Regex:** The HTML cleaning functionality (`--remove-html`) is based on regular expressions. While effective for many common cases, regex-based HTML parsing is inherently limited and may not perfectly handle all complex, malformed, or edge-case HTML structures. For truly robust HTML parsing, dedicated HTML parser libraries would be required.
*   **Glob Pattern Matching:** The `--input-pattern` implements basic glob-like functionality (`*`, `?`) by converting the pattern to a regular expression. While it handles common cases and escapes some regex metacharacters for intuitive use (e.g., a literal `.` in the pattern), it may not support all advanced features of POSIX glob syntax or complex regex-like constructs within the glob pattern itself. For very complex file matching, pre-filtering files using shell commands might be more appropriate.
*   **Memory Usage for Large Files:** The current approach reads the entire file content into memory before processing. This is generally efficient for typical text files, but for extremely large single files (multiple gigabytes), it could lead to high memory consumption.

## Future Considerations / Potential Improvements

*   **More Sophisticated Glob Pattern Matching:** Implementing or integrating a more complete globbing library or refining the current regex conversion.
*   **Advanced Regex Options:**
    *   Allowing users to provide custom regex patterns for removal/replacement.
*   **Unicode Support:**
    *   More explicit handling of Unicode characters in non-printable removal (current version focuses on ASCII).
    *   Options for Unicode normalization (NFC, NFKC), likely requiring a library like ICU.
*   **Near-Duplicate Line Removal:** Techniques like MinHash or SimHash for semantic duplicate detection.
*   **Performance Optimizations:** Investigating memory mapping or chunked processing for extremely large files, especially for operations that can be performed line-by-line or in smaller segments.
*   **Configuration Files:** Allowing cleaning configurations to be specified in a file.

## Contributing

Contributions to Lunaris Text Cleaner are welcome! Whether it's reporting bugs, suggesting new features, improving documentation, or submitting pull requests for code changes, your help is appreciated.

If you're looking to contribute, some interesting areas include:
*   Addressing a known limitation.
*   Implementing one of the "Future Considerations".
*   Adding more comprehensive unit tests for the cleaning functions.
*   Refactoring for even better performance or code clarity.

Please feel free to open an issue on the GitHub repository to discuss your ideas or report problems.

---
