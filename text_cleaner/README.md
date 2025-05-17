```markdown
# Lunaris Text Cleaner (`lunaris_text_cleaner`)

**Version: 0.2.1**

`lunaris_text_cleaner` is a C++ command-line utility designed to perform various cleaning and normalization operations on large text files, preparing them for further processing étapes, such as tokenization for training language models like Lunaris Codex.

It aims to provide a fast and efficient way to improve dataset quality by handling common text issues directly in C++.

## Features

-   **Whitespace Normalization:**
    -   Trims leading and trailing whitespace from each line.
    -   Reduces multiple internal spaces/tabs to a single space.
-   **Empty Line Removal:** Optionally removes lines that become entirely empty *after* whitespace normalization.
-   **Case Conversion:** Optionally converts all text to lowercase.
-   **Non-Printable Character Removal:** Removes common non-printable ASCII characters while preserving essential whitespace like tabs (`\t`), newlines (`\n`), and carriage returns (`\r`).
-   **URL Processing:**
    -   Detects and removes common URL patterns.
    -   Optionally replaces URLs with a user-defined placeholder string (e.g., `"[URL]"`).
-   **Email Processing:**
    -   Detects and removes common email address patterns.
    -   Optionally replaces email addresses with a user-defined placeholder string (e.g., `"[EMAIL]"`).
-   **Exact Duplicate Line Removal:** After all other selected processing steps are applied to a line, this option can remove lines that are exact duplicates of previously processed and written lines.
-   **File Handling:** Reads from an input file and writes the cleaned content to a new output file, creating parent directories for the output file if they don't exist.
-   **Configurable:** All cleaning operations are controlled via command-line arguments.

## Prerequisites

-   A C++17 compatible compiler (e.g., `g++`).
-   Standard C++ libraries (iostream, fstream, string, vector, algorithm, cctype, filesystem, regex, set, iomanip, sstream).
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

Run the compiled executable from your terminal, providing the input file, output file, and desired cleaning options:

```bash
./lunaris_text_cleaner --input <input_file.txt> --output <cleaned_file.txt> [options...]
```

### Command-Line Arguments:

*   `--input <path>`: **(Required)** Path to the input text file.
*   `--output <path>`: **(Required)** Path to save the cleaned output text file.
*   `--normalize-whitespace`: (Optional) Enable trimming and reduction of multiple whitespaces to a single space.
*   `--remove-empty-lines`: (Optional) Remove lines that become empty *after* whitespace normalization. This option is only effective if `--normalize-whitespace` is also enabled.
*   `--to-lowercase`: (Optional) Convert all text to lowercase.
*   `--remove-non-printable`: (Optional) Remove non-printable ASCII characters (preserves tab, newline, carriage return).
*   `--process-urls`: (Optional) Enable processing of URLs. If `--url-placeholder` is not provided or is empty, URLs will be removed.
*   `--url-placeholder <str>`: (Optional) String to replace detected URLs with (e.g., `"<URL_TOKEN>"`). Effective only if `--process-urls` is set.
*   `--process-emails`: (Optional) Enable processing of email addresses. If `--email-placeholder` is not provided or is empty, emails will be removed.
*   `--email-placeholder <str>`: (Optional) String to replace detected email addresses with (e.g., `"<EMAIL_TOKEN>"`). Effective only if `--process-emails` is set.
*   `--remove-exact-duplicates`: (Optional) Remove lines that are exact duplicates of previously processed and written lines. This check is performed *after* all other enabled cleaning operations for a given line.
*   `-h`, `--help`: Display the help message and exit.

### Example: Applying Multiple Cleaning Operations

```bash
./lunaris_text_cleaner \
    --input ./raw_dataset/corpus_part1.txt \
    --output ./cleaned_dataset/corpus_part1_cleaned.txt \
    --normalize-whitespace \
    --remove-empty-lines \
    --to-lowercase \
    --process-urls --url-placeholder "[URL]" \
    --process-emails --email-placeholder "[EMAIL]" \
    --remove-exact-duplicates
```
This command will:
1. Read from `raw_dataset/corpus_part1.txt`.
2. Apply URL replacement (with `[URL]`).
3. Apply email replacement (with `[EMAIL]`).
4. Normalize whitespace on each line.
5. Convert each line to lowercase.
6. Remove any line that becomes empty after the above steps.
7. Remove any line that, after all aformentioned processing, is an exact duplicate of a line already written to the output.
8. Save the result to `cleaned_dataset/corpus_part1_cleaned.txt`.

## Order of Operations

The cleaning steps are applied in the following order for each line read from the input file:
1.  Non-printable character removal (if `--remove-non-printable` is set).
2.  URL processing (removal or replacement, if `--process-urls` is set).
3.  Email processing (removal or replacement, if `--process-emails` is set).
4.  Whitespace normalization (if `--normalize-whitespace` is set).
5.  Conversion to lowercase (if `--to-lowercase` is set).
6.  Check for (and optionally remove) lines that became empty after the above.
7.  Check for (and optionally remove) exact duplicate lines based on the fully processed content of the line.

## Future Considerations / Potential Improvements

-   Support for processing all files in an input directory (glob patterns).
-   More sophisticated URL and email detection regex.
-   Options for handling or normalizing specific Unicode characters or blocks.
-   Removal of near-duplicates (e.g., using MinHash or SimHash).
-   Language-specific cleaning rules (e.g., for code comments).
-   Performance optimizations for extremely large files (e.g., chunked reading and parallel processing if feasible).

---
