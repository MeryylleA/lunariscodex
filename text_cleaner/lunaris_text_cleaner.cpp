#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>    // std::transform, std::remove_if, std::sort
#include <cctype>       // std::isspace, std::isprint, std::tolower
#include <filesystem>   // std::filesystem (C++17)
#include <regex>        // std::regex, std::regex_replace (C++11)
#include <set>          // std::set for exact duplicate removal
#include <sstream>      // For std::ostringstream to build the options string
#include <iomanip>      // For std::fixed, std::setprecision
#include <chrono>       // For timing execution
#include <limits>       // For std::numeric_limits
#include <cstdint>      // For std::uintmax_t

// Structure to hold parsed command-line arguments
struct CleanArgs {
    std::string input_path;
    std::string output_path;
    std::string input_pattern = "*.txt"; // Default input file pattern
    bool recursive_search = false;
    bool normalize_whitespace = false;
    bool remove_empty_lines_after_ws_norm = false;
    bool to_lowercase = false;
    bool remove_non_printable = false;
    bool remove_html = false;
    bool process_urls = false;
    std::string url_placeholder = "";
    bool process_emails = false;
    std::string email_placeholder = "";
    bool remove_exact_duplicate_lines = false;
};

// Forward declaration for the main file processing function
bool process_single_file(const std::filesystem::path& input_file,
                         const std::filesystem::path& output_file,
                         const CleanArgs& args,
                         long long& lines_read_count, long long& lines_written_count,
                         long long& lines_became_empty_count, long long& lines_removed_duplicate_count,
                         long long& html_modified_lines_count,
                         long long& urls_processed_lines_count, long long& emails_processed_lines_count);

// Argument Parser for command-line options
bool parse_clean_arguments(int argc, char* argv[], CleanArgs& args) {
    if (argc == 1) { // If no arguments, show help by default
        argv[argc++] = (char*)"--help";
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--help" || arg == "-h")) {
            // Display help message and usage instructions
            std::cout << "Lunaris Text Cleaner (C++ v0.3.7) Usage:" << std::endl; // Updated version
            std::cout << "  --input <path>          (Required) Path to the input file or directory." << std::endl;
            std::cout << "  --output <path>         (Required) Path to the output file or base directory." << std::endl;
            std::cout << "  --input-pattern <glob>  (Optional) Glob-like pattern for files if --input is a directory. Default: \"*.txt\"." << std::endl;
            std::cout << "  --recursive             (Optional) Search recursively if --input is a directory." << std::endl;
            std::cout << "  --normalize-whitespace  (Optional) Trim and reduce multiple whitespaces to one." << std::endl;
            std::cout << "  --remove-empty-lines    (Optional) Remove lines that become empty after normalization (requires --normalize-whitespace)." << std::endl;
            std::cout << "  --to-lowercase          (Optional) Convert all text to lowercase." << std::endl;
            std::cout << "  --remove-non-printable  (Optional) Remove non-printable ASCII characters (keeps tab, newline, CR)." << std::endl;
            std::cout << "  --remove-html           (Optional) Remove DOCTYPE, HTML/XML comments, script/style blocks, and tags." << std::endl;
            std::cout << "  --process-urls          (Optional) Process URLs. If --url-placeholder is empty, URLs are removed." << std::endl;
            std::cout << "  --url-placeholder <str> (Optional) Replace URLs with this string. Effective if --process-urls is set." << std::endl;
            std::cout << "  --process-emails        (Optional) Process email addresses. If --email-placeholder is empty, emails are removed." << std::endl;
            std::cout << "  --email-placeholder <str>(Optional) Replace emails with this string. Effective if --process-emails is set." << std::endl;
            std::cout << "  --remove-exact-duplicates (Optional) Remove exact duplicate lines (after other processing)." << std::endl;
            return false; // Indicate that help was shown, and normal execution should not proceed
        } else if (arg == "--input" && i + 1 < argc) args.input_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.output_path = argv[++i];
        else if (arg == "--input-pattern" && i + 1 < argc) args.input_pattern = argv[++i];
        else if (arg == "--recursive") args.recursive_search = true;
        else if (arg == "--normalize-whitespace") args.normalize_whitespace = true;
        else if (arg == "--remove-empty-lines") args.remove_empty_lines_after_ws_norm = true;
        else if (arg == "--to-lowercase") args.to_lowercase = true;
        else if (arg == "--remove-non-printable") args.remove_non_printable = true;
        else if (arg == "--remove-html") args.remove_html = true;
        else if (arg == "--process-urls") args.process_urls = true;
        else if (arg == "--url-placeholder" && i + 1 < argc) args.url_placeholder = argv[++i];
        else if (arg == "--process-emails") args.process_emails = true;
        else if (arg == "--email-placeholder" && i + 1 < argc) args.email_placeholder = argv[++i];
        else if (arg == "--remove-exact-duplicates") args.remove_exact_duplicate_lines = true;
        else {
            std::cerr << "Error: Unknown argument or missing value for argument: " << arg << std::endl;
            return false; // Indicate parsing failure
        }
    }

    // Validate required arguments
    if (args.input_path.empty() || args.output_path.empty()) {
        std::cerr << "Error: --input and --output paths are required arguments." << std::endl;
        if (!(argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) {
            // Only show "Use --help" if help wasn't the primary action
            std::cerr << "Use -h or --help for usage information." << std::endl;
        }
        return false; // Indicate validation failure
    }

    // Validate argument dependencies
    if (args.remove_empty_lines_after_ws_norm && !args.normalize_whitespace) {
        std::cerr << "Warning: --remove-empty-lines is only effective if --normalize-whitespace is also enabled. Option will be ignored." << std::endl;
        args.remove_empty_lines_after_ws_norm = false; // Disable the option as it's ineffective
    }
    return true; // Arguments parsed and validated successfully
}

// --- String Cleaning Helper Functions ---

// Removes leading whitespace from a string in-place.
static inline void ltrim_inplace(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch){
        return !std::isspace(ch);
    }));
}

// Removes trailing whitespace from a string in-place.
static inline void rtrim_inplace(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch){
        return !std::isspace(ch);
    }).base(), s.end());
}

// Removes leading and trailing whitespace from a string in-place.
static inline void trim_inplace(std::string &s) {
    ltrim_inplace(s);
    rtrim_inplace(s);
}

// Reduces multiple internal whitespace characters in a string to a single space.
// Does not trim leading/trailing whitespace; use trim_inplace for that.
std::string reduce_internal_whitespaces(const std::string& input_str) {
    if (input_str.empty()) return "";
    std::string result;
    result.reserve(input_str.length()); // Pre-allocate for efficiency
    bool last_was_space = false;
    for (char c : input_str) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!last_was_space) {
                result += ' '; // Add a single space for a sequence of whitespaces
                last_was_space = true;
            }
        } else {
            result += c;
            last_was_space = false;
        }
    }
    return result;
}

// Original function to remove non-printable characters using std::isprint.
// Kept for reference or if a locale-dependent behavior is ever desired.
// std::string apply_remove_non_printable(const std::string& s) {
//     std::string result; result.reserve(s.length());
//     for (char c : s) { if (std::isprint(static_cast<unsigned char>(c)) || c == '\t' || c == '\n' || c == '\r') { result += c; }}
//     return result;
// }

// Removes non-printable ASCII control characters, keeping Tab, LF, CR, and standard printable ASCII.
// This function targets specific ASCII byte values and is not locale-dependent for the basic ASCII range.
// It does NOT remove textual escape sequences like the string "\x01".
std::string apply_remove_non_printable_explicit(const std::string& s) {
    std::string result;
    result.reserve(s.length());
    for (unsigned char uc : s) { // Iterate as unsigned char to correctly handle byte values
        // Keep Tab (ASCII 9), Line Feed (ASCII 10), Carriage Return (ASCII 13)
        // Keep standard printable ASCII characters (space (32) to tilde (126))
        if (uc == 9 || uc == 10 || uc == 13 || (uc >= 32 && uc <= 126)) {
            result += static_cast<char>(uc);
        }
        // All other byte values (especially ASCII control characters < 32, excluding 9,10,13, and bytes > 126) are omitted.
    }
    return result;
}


// Pre-compiled regex objects for performance.
// std::regex_constants::optimize hint can help, actual effect varies by library implementation.
// std::regex_constants::icase for case-insensitive matching.
const std::regex doctype_regex(R"(<!DOCTYPE[^>]*>)", std::regex_constants::icase | std::regex_constants::optimize);
const std::regex html_comment_regex(R"(<!--[\s\S]*?-->)", std::regex_constants::optimize); // [\s\S] matches any char including newlines
const std::regex script_tag_regex(R"(<script[^>]*>[\s\S]*?<\/script>)", std::regex_constants::icase | std::regex_constants::optimize);
const std::regex style_tag_regex(R"(<style[^>]*>[\s\S]*?<\/style>)", std::regex_constants::icase | std::regex_constants::optimize);
// General HTML/XML tag regex: matches opening, closing, and self-closing tags.
const std::regex general_html_tag_regex(R"(<\/?\s*([a-zA-Z_:][-a-zA-Z0-9_:.]*)([^>]*)?>)", std::regex_constants::optimize);

// Regex for URLs (common protocols and www start)
const std::regex url_regex(R"((?:https?://|ftp://|www\.)[^\s/$.?#][^\s]*)", std::regex_constants::icase | std::regex_constants::optimize);
// Regex for email addresses
const std::regex email_regex(R"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b)", std::regex_constants::optimize);


// --- Core File Processing Logic ---
// Processes a single input file according to the provided arguments and updates statistics.
bool process_single_file(
    const std::filesystem::path& input_file_path,
    const std::filesystem::path& output_file_path,
    const CleanArgs& args,
    long long& file_lines_read_stat, long long& file_lines_written_stat,
    long long& file_lines_became_empty_stat, long long& file_lines_removed_duplicate_stat,
    long long& file_html_modified_flag, // Flag: 1 if HTML ops changed content, 0 otherwise
    long long& file_urls_processed_lines_stat, long long& file_emails_processed_lines_stat
) {
    std::cout << "  Processing: " << input_file_path.string() << "\n    Output to : " << output_file_path.string() << std::endl;

    // Robust file reading into a string buffer
    std::ifstream infile(input_file_path, std::ios::binary); // Open in binary for consistent size calculation
    if (!infile.is_open()) {
        std::cerr << "    Error: Could not open input file '" << input_file_path.string() << "'." << std::endl;
        return false;
    }

    std::string full_file_content;
    infile.seekg(0, std::ios::end);
    std::streampos end_pos = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::streampos beg_pos = infile.tellg();

    if (end_pos == static_cast<std::streampos>(-1) || beg_pos == static_cast<std::streampos>(-1) || end_pos <= beg_pos) {
        // File is empty, unreadable, or tellg failed. full_file_content will remain empty.
    } else {
        std::uintmax_t file_size_uint = static_cast<std::uintmax_t>(end_pos - beg_pos);

        // Check if file size exceeds what std::string::resize can handle
        if (file_size_uint > std::numeric_limits<size_t>::max()) {
            std::cerr << "    Error: File '" << input_file_path.string() << "' is too large for std::string buffer. Size: " << file_size_uint << std::endl;
            infile.close();
            return false;
        }
        size_t file_size_for_buffer = static_cast<size_t>(file_size_uint);

        try {
            full_file_content.resize(file_size_for_buffer);
        } catch (const std::length_error& e) { // Catch memory allocation failure for the buffer
            std::cerr << "    Error: Not enough memory to allocate buffer for file '" << input_file_path.string() << "'. Requested size: " << file_size_for_buffer << ". Error: " << e.what() << std::endl;
            infile.close();
            return false;
        }

        infile.read(&full_file_content[0], file_size_for_buffer);

        // Verify if the number of bytes read matches the expected size
        if (static_cast<size_t>(infile.gcount()) != file_size_for_buffer) {
            std::cerr << "    Warning: Read " << infile.gcount() << " bytes, but expected " << file_size_for_buffer << " for file '" << input_file_path.string() << "'. Processing read content." << std::endl;
            full_file_content.resize(static_cast<size_t>(infile.gcount())); // Adjust buffer to actual bytes read
        }
    }
    infile.close(); // Close input file as soon as content is read into memory

    std::string processed_content = full_file_content; // Start with the full file content for processing
    bool html_content_was_changed_by_html_ops = false;

    // --- Apply global (full-content) operations first ---
    if (args.remove_html) {
        std::string temp_content_holder; // Temporary string for regex_replace results

        temp_content_holder = std::regex_replace(processed_content, doctype_regex, "");
        if (temp_content_holder != processed_content) html_content_was_changed_by_html_ops = true;
        processed_content = temp_content_holder;

        temp_content_holder = std::regex_replace(processed_content, html_comment_regex, "");
        if (temp_content_holder != processed_content) html_content_was_changed_by_html_ops = true;
        processed_content = temp_content_holder;

        temp_content_holder = std::regex_replace(processed_content, script_tag_regex, "");
        if (temp_content_holder != processed_content) html_content_was_changed_by_html_ops = true;
        processed_content = temp_content_holder;

        temp_content_holder = std::regex_replace(processed_content, style_tag_regex, "");
        if (temp_content_holder != processed_content) html_content_was_changed_by_html_ops = true;
        processed_content = temp_content_holder;

        temp_content_holder = std::regex_replace(processed_content, general_html_tag_regex, "");
        if (temp_content_holder != processed_content) html_content_was_changed_by_html_ops = true;
        processed_content = temp_content_holder;
    }

    if (html_content_was_changed_by_html_ops) {
        file_html_modified_flag = 1; // Set flag if any HTML operation altered the content
    }

    // Ensure output directory exists before opening the output file
    if (output_file_path.has_parent_path()) {
        try {
            if (!std::filesystem::exists(output_file_path.parent_path())) {
                std::filesystem::create_directories(output_file_path.parent_path());
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "    Error creating output directory for '" << output_file_path.string() << "': " << e.what() << std::endl;
            return false;
        }
    }
    std::ofstream outfile(output_file_path);
    if (!outfile.is_open()) {
        std::cerr << "    Error: Could not open output file '" << output_file_path.string() << "' for writing." << std::endl;
        return false;
    }

    std::set<std::string> seen_lines_for_deduplication; // Used for --remove-exact-duplicates
    std::istringstream content_stream(processed_content); // Stream for line-by-line processing
    std::string line; // Current line being processed

    // Count original lines read from the raw input file content (before any processing)
    // This stat should reflect the actual lines in the input file.
    std::istringstream original_content_stream_for_stats(full_file_content);
    std::string original_line_for_stats_count;
    while(std::getline(original_content_stream_for_stats, original_line_for_stats_count)) {
        file_lines_read_stat++;
    }

    // --- Apply line-by-line operations ---
    while (std::getline(content_stream, line)) {
        std::string original_line_state_for_empty_check = line; // State before per-line ops
        bool line_was_modified_by_per_line_ops = false;

        if (args.remove_non_printable) {
            std::string temp_line = apply_remove_non_printable_explicit(line);
            if (temp_line != line) line_was_modified_by_per_line_ops = true;
            line = temp_line;
        }

        bool url_found_in_line = false;
        if (args.process_urls) {
            std::string temp_line = std::regex_replace(line, url_regex, args.url_placeholder);
            if (temp_line != line) {
                url_found_in_line = true;
                line_was_modified_by_per_line_ops = true;
            }
            line = temp_line;
        }
        if(url_found_in_line) file_urls_processed_lines_stat++;

        bool email_found_in_line = false;
        if (args.process_emails) {
            std::string temp_line = std::regex_replace(line, email_regex, args.email_placeholder);
            if (temp_line != line) {
                email_found_in_line = true;
                line_was_modified_by_per_line_ops = true;
            }
            line = temp_line;
        }
        if(email_found_in_line) file_emails_processed_lines_stat++;

        if (args.normalize_whitespace) {
            std::string original_before_ws_norm = line;
            trim_inplace(line); // Trim leading/trailing first
            line = reduce_internal_whitespaces(line); // Reduce internal spaces
            trim_inplace(line); // Trim again in case reduce_internal left a trailing space from an all-space line
            if (line != original_before_ws_norm) line_was_modified_by_per_line_ops = true;
        }

        if (args.to_lowercase) {
            std::string original_before_lc = line;
            std::transform(line.begin(), line.end(), line.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            if (line != original_before_lc) line_was_modified_by_per_line_ops = true;
        }

        // Check if the line became empty due to the per-line operations above
        if (line.empty() && !original_line_state_for_empty_check.empty()) {
            file_lines_became_empty_stat++;
        }

        // Remove lines that are now empty, if the option is set (and normalization was done)
        if (line.empty() && args.remove_empty_lines_after_ws_norm && args.normalize_whitespace) {
            continue; // Skip writing this empty line
        }

        // Remove exact duplicate lines (after all other processing on the line)
        if (args.remove_exact_duplicate_lines) {
            auto insert_result = seen_lines_for_deduplication.emplace(line);
            if (!insert_result.second) { // If emplace returns false, the line was already in the set
                file_lines_removed_duplicate_stat++;
                continue; // Skip writing this duplicate line
            }
        }

        outfile << line << std::endl;
        file_lines_written_stat++;
    }

    outfile.close();
    std::cout << "    Finished processing: " << input_file_path.filename().string() << std::endl;
    return true;
}

// Helper function to handle processing for a single directory entry (file).
// It checks against the input pattern and calls process_single_file.
void handle_file_processing(
    const std::filesystem::directory_entry& dir_entry,
    const CleanArgs& args,
    const std::filesystem::path& input_base_path,  // Base input directory for relative path calculation
    const std::filesystem::path& output_base_path, // Base output directory
    long long& total_files_processed,
    long long& total_lines_read_overall, long long& total_lines_written_overall,
    long long& total_lines_became_empty_overall, long long& total_lines_removed_duplicate_overall,
    long long& total_html_modified_overall,
    long long& total_urls_processed_lines_overall, long long& total_emails_processed_lines_overall
) {
    bool matches_pattern = false;
    std::string filename_str = dir_entry.path().filename().string();
    const std::string& pattern_to_match = args.input_pattern;

    // Simple glob-like pattern matching
    if (pattern_to_match == "*.*" || pattern_to_match == "*") { // Match all files
        matches_pattern = true;
    } else if (pattern_to_match.rfind("*.", 0) == 0) { // Match by extension, e.g., "*.txt"
        std::string ext_to_match = pattern_to_match.substr(1); // e.g., ".txt"
        if (dir_entry.path().extension().string() == ext_to_match) {
            matches_pattern = true;
        }
    } else if (pattern_to_match.find("*") == std::string::npos && pattern_to_match.find("?") == std::string::npos) { // Exact match if no wildcards
        if (filename_str == pattern_to_match) matches_pattern = true;
    } else { // Generic glob to regex conversion for patterns with * or ?
        std::string regex_str = pattern_to_match;

        // Escape regex metacharacters present in the glob pattern (except * and ? which are wildcards)
        std::string temp_escaped_regex_str;
        temp_escaped_regex_str.reserve(regex_str.length() * 2);
        for (char c : regex_str) {
            // Characters to escape: . ^ $ | ( ) [ ] { } + \ (and * ? if they were not wildcards)
            // Here, we only escape non-glob-wildcard regex metacharacters.
            if (c == '.' || c == '^' || c == '$' || c == '|' ||
                c == '(' || c == ')' || c == '[' || c == ']' ||
                c == '{' || c == '}' || c == '+' || c == '\\') {
                temp_escaped_regex_str += '\\';
                }
                temp_escaped_regex_str += c;
        }
        regex_str = temp_escaped_regex_str;

        // Convert glob wildcards '*' and '?' to their regex equivalents '.*' and '.'
        std::string final_regex_str;
        final_regex_str.reserve(regex_str.length() * 2);
        size_t last_pos = 0;
        for (size_t i = 0; i < regex_str.length(); ++i) {
            if (regex_str[i] == '*') {
                // Check if this '*' was an escaped literal '*' (i.e., preceded by '\')
                if (i > 0 && regex_str[i-1] == '\\') {
                    // It's an escaped literal '*', so keep it as is (already has backslash)
                    // No, this logic is flawed. The previous loop already added a \.
                    // We need to replace unescaped * and ?.
                    // Simpler: first escape everything, then replace \* with .* and \? with .
                    // For now, this simpler replacement handles basic cases.
                    // The escaping above handles things like "file.name*.txt" correctly if '.' is escaped.
                    // This part converts the glob '*' to regex '.*'
                    // Let's refine the replacement part only for '*' and '?' that weren't part of an escape sequence from the previous step
                    // For simplicity, assume previous step correctly escaped literal . etc. and this step converts glob * and ?
                }
                // This logic needs careful review for complex patterns like "\*.txt" (literal star)
                // The current loop below is a simplified version.
            }
        }
        // Simpler approach for glob to regex after escaping non-wildcard metachars:
        std::string temp_glob_to_regex_build;
        bool prev_char_was_escape = false;
        for(char c_glob : regex_str) { // regex_str here is after initial escaping of .,^,$, etc.
            if (c_glob == '*' && !prev_char_was_escape) {
                temp_glob_to_regex_build += ".*";
            } else if (c_glob == '?' && !prev_char_was_escape) {
                temp_glob_to_regex_build += ".";
            } else if (c_glob == '\\') {
                if (prev_char_was_escape) { // Handle double backslash \\ -> \
                    temp_glob_to_regex_build += '\\';
                    prev_char_was_escape = false;
                } else {
                    temp_glob_to_regex_build += '\\';
                    prev_char_was_escape = true;
                }
            }
            else {
                temp_glob_to_regex_build += c_glob;
                prev_char_was_escape = false;
            }
        }
        regex_str = temp_glob_to_regex_build;


        try {
            // Compile the generated regex string
            std::regex pattern_regex_obj(regex_str, std::regex_constants::optimize | std::regex_constants::ECMAScript);
            if (std::regex_match(filename_str, pattern_regex_obj)) {
                matches_pattern = true;
            }
        } catch (const std::regex_error& e) {
            std::cerr << "Warning: Invalid regex generated from input pattern '" << pattern_to_match
            << "' (became '" << regex_str << "'). Error: " << e.what() << " Code: " << e.code() << std::endl;
            // matches_pattern remains false if regex is invalid
        }
    }

    if (matches_pattern) {
        std::filesystem::path current_input_file = dir_entry.path();
        // Construct output path preserving relative directory structure
        std::filesystem::path relative_path = std::filesystem::relative(current_input_file, input_base_path);
        std::filesystem::path current_output_file = output_base_path / relative_path;

        // Per-file statistics
        long long current_file_lines_read = 0, current_file_lines_written = 0;
        long long current_file_became_empty = 0, current_file_removed_duplicate = 0;
        long long current_file_html_modified_flag = 0;
        long long current_file_urls_processed = 0, current_file_emails_processed = 0;

        if(process_single_file(current_input_file, current_output_file, args,
            current_file_lines_read, current_file_lines_written,
            current_file_became_empty, current_file_removed_duplicate,
            current_file_html_modified_flag,
            current_file_urls_processed, current_file_emails_processed)) {
            total_files_processed++;
        // Aggregate statistics
        total_lines_read_overall += current_file_lines_read;
        total_lines_written_overall += current_file_lines_written;
        total_lines_became_empty_overall += current_file_became_empty;
        total_lines_removed_duplicate_overall += current_file_removed_duplicate;
        total_html_modified_overall += current_file_html_modified_flag; // Sum of flags (0 or 1 per file)
        total_urls_processed_lines_overall += current_file_urls_processed;
        total_emails_processed_lines_overall += current_file_emails_processed;
            }
    }
}

// Main program entry point
int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now(); // For timing execution

    CleanArgs args;
    if (!parse_clean_arguments(argc, argv, args)) {
        // Exit if parsing failed or if help was displayed (parse_clean_arguments returns false for --help)
        return (argc <= 1 || (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) ? 0 : 1;
    }

    // Print program header and selected options
    std::cout << "--- Lunaris Text Cleaner (C++ v0.3.7) ---" << std::endl; // Updated version
    std::cout << "Input path: " << args.input_path << std::endl;
    std::cout << "Output path: " << args.output_path << std::endl;
    if (std::filesystem::is_directory(args.input_path)) {
        std::cout << "Input file pattern: \"" << args.input_pattern << "\"" << (args.recursive_search ? " (recursive)" : "") << std::endl;
    }
    std::ostringstream options_ss; // Build a string of active options
    if(args.normalize_whitespace) options_ss << "NormalizeWhitespace ";
    if(args.remove_empty_lines_after_ws_norm) options_ss << "RemoveEmptyLines ";
    if(args.to_lowercase) options_ss << "ToLowercase ";
    if(args.remove_non_printable) options_ss << "RemoveNonPrintable ";
    if(args.remove_html) options_ss << "RemoveHTML ";
    if(args.process_urls) options_ss << "ProcessURLs" << (args.url_placeholder.empty() ? "[remove] " : "[replace_with:\"" + args.url_placeholder + "\"] ");
    if(args.process_emails) options_ss << "ProcessEmails" << (args.email_placeholder.empty() ? "[remove] " : "[replace_with:\"" + args.email_placeholder + "\"] ");
    if(args.remove_exact_duplicate_lines) options_ss << "RemoveExactDuplicates ";
    std::string options_str = options_ss.str();
    std::cout << "Options: " << (options_str.empty() ? "None" : options_str) << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;

    // Overall statistics counters
    long long total_files_processed = 0;
    long long total_lines_read_overall = 0;
    long long total_lines_written_overall = 0;
    long long total_lines_became_empty_overall = 0;
    long long total_lines_removed_duplicate_overall = 0;
    long long total_html_modified_overall = 0;
    long long total_urls_processed_lines_overall = 0;
    long long total_emails_processed_lines_overall = 0;

    std::filesystem::path input_fs_path(args.input_path);
    std::filesystem::path output_fs_path(args.output_path);

    try {
        if (std::filesystem::is_regular_file(input_fs_path)) {
            // --- Single File Mode ---
            std::cout << "Mode: Processing a single file." << std::endl;
            if (std::filesystem::is_directory(output_fs_path)) {
                // If output is a directory, append input filename to it
                output_fs_path /= input_fs_path.filename();
            } else if (output_fs_path.has_parent_path()) {
                // Ensure parent directory of the output file exists
                if (!std::filesystem::exists(output_fs_path.parent_path())) {
                    std::filesystem::create_directories(output_fs_path.parent_path());
                }
            }
            if(process_single_file(input_fs_path, output_fs_path, args,
                total_lines_read_overall, total_lines_written_overall,
                total_lines_became_empty_overall, total_lines_removed_duplicate_overall,
                total_html_modified_overall,
                total_urls_processed_lines_overall, total_emails_processed_lines_overall)) {
                total_files_processed = 1;
                }
        } else if (std::filesystem::is_directory(input_fs_path)) {
            // --- Directory Mode ---
            std::cout << "Mode: Processing directory..." << std::endl;
            if (std::filesystem::exists(output_fs_path) && !std::filesystem::is_directory(output_fs_path)) {
                std::cerr << "Error: Input is a directory, but output path '" << args.output_path << "' is an existing file. Output must be a directory." << std::endl;
                return 1;
            }
            // Ensure output base directory exists
            if (!std::filesystem::exists(output_fs_path)) {
                std::filesystem::create_directories(output_fs_path);
            }

            std::filesystem::directory_options dir_iter_options = std::filesystem::directory_options::follow_directory_symlink;

            if (args.recursive_search) {
                for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(input_fs_path, dir_iter_options)) {
                    if (dir_entry.is_regular_file()) {
                        handle_file_processing(dir_entry, args, input_fs_path, output_fs_path,
                                               total_files_processed, total_lines_read_overall, total_lines_written_overall,
                                               total_lines_became_empty_overall, total_lines_removed_duplicate_overall,
                                               total_html_modified_overall,
                                               total_urls_processed_lines_overall, total_emails_processed_lines_overall);
                    }
                }
            } else { // Non-recursive search
                for (const auto& dir_entry : std::filesystem::directory_iterator(input_fs_path, dir_iter_options)) {
                    if (dir_entry.is_regular_file()) {
                        handle_file_processing(dir_entry, args, input_fs_path, output_fs_path,
                                               total_files_processed, total_lines_read_overall, total_lines_written_overall,
                                               total_lines_became_empty_overall, total_lines_removed_duplicate_overall,
                                               total_html_modified_overall,
                                               total_urls_processed_lines_overall, total_emails_processed_lines_overall);
                    }
                }
            }
        } else {
            std::cerr << "Error: Input path '" << args.input_path << "' is not a valid file or directory." << std::endl;
            return 1;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem Error during processing: " << e.what() << std::endl;
        return 1;
    } catch (const std::regex_error& e) {
        std::cerr << "Regex Error during processing: " << e.what() << " (Code: " << e.code() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) { // Catch any other standard exceptions
        std::cerr << "Standard Exception during processing: " << e.what() << std::endl;
        return 1;
    }


    // Calculate and display processing time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Display overall processing summary
    std::cout << "\n--- Overall Processing Summary ---" << std::endl;
    if (total_files_processed > 0) {
        std::cout << "Total files processed: " << total_files_processed << std::endl;
        std::cout << "Total lines originally read from input files: " << total_lines_read_overall << std::endl;
        if (args.remove_non_printable) std::cout << "Non-printable character removal was applied." << std::endl;
        if (args.remove_html) std::cout << "Total files where HTML content was modified/removed: " << total_html_modified_overall << std::endl;
        if (args.process_urls) std::cout << "Total lines with URLs processed/replaced: " << total_urls_processed_lines_overall << std::endl;
        if (args.process_emails) std::cout << "Total lines with Emails processed/replaced: " << total_emails_processed_lines_overall << std::endl;
        if (args.normalize_whitespace) std::cout << "Whitespace normalization was applied." << std::endl;
        if (args.to_lowercase) std::cout << "Text was converted to lowercase." << std::endl;
        std::cout << "Total lines that became empty after processing: " << total_lines_became_empty_overall << std::endl;
        if (args.remove_empty_lines_after_ws_norm && args.normalize_whitespace) {
            std::cout << "  (These processed empty lines were removed from output)" << std::endl;
        }
        if (args.remove_exact_duplicate_lines) std::cout << "Total exact duplicate lines removed: " << total_lines_removed_duplicate_overall << std::endl;
        std::cout << "Total lines written to output: " << total_lines_written_overall << std::endl;
    } else {
        std::cout << "No files were processed." << std::endl;
    }
    std::cout << "Processing finished in " << std::fixed << std::setprecision(3) << elapsed_seconds.count() << " seconds." << std::endl;
    std::cout << "------------------------------------" << std::endl;

    return 0; // Successful execution
}
