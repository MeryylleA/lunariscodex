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

// Structure to hold parsed command-line arguments
struct CleanArgs {
    std::string input_path;
    std::string output_path;
    std::string input_pattern = "*.txt";
    bool recursive_search = false;
    bool normalize_whitespace = false;
    bool remove_empty_lines_after_ws_norm = false;
    bool to_lowercase = false;
    bool remove_non_printable = false;
    bool process_urls = false;
    std::string url_placeholder = "";
    bool process_emails = false;
    std::string email_placeholder = "";
    bool remove_exact_duplicate_lines = false;
};

// Forward declaration for the core file processing logic
bool process_single_file(const std::filesystem::path& input_file,
                         const std::filesystem::path& output_file,
                         const CleanArgs& args,
                         long long& lines_read_count, long long& lines_written_count,
                         long long& lines_became_empty_count, long long& lines_removed_duplicate_count,
                         long long& urls_processed_lines_count, long long& emails_processed_lines_count);

// Simple command-line argument parser
bool parse_clean_arguments(int argc, char* argv[], CleanArgs& args) {
    if (argc == 1) { // If no arguments, simulate --help
        argv[argc++] = (char*)"--help";
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--help" || arg == "-h")) {
            std::cout << "Lunaris Text Cleaner (C++ v0.3.1) Usage:" << std::endl;
            std::cout << "  --input <path>          (Required) Path to the input file or directory." << std::endl;
            std::cout << "  --output <path>         (Required) Path to the output file or base directory." << std::endl;
            std::cout << "  --input-pattern <glob>  (Optional) Glob-like pattern for files if --input is a directory (e.g., \"*.txt\", \"file_prefix_*\"). Default: \"*.txt\"." << std::endl;
            std::cout << "  --recursive             (Optional) Search recursively if --input is a directory." << std::endl;
            std::cout << "  --normalize-whitespace  (Optional) Trim and reduce multiple whitespaces to one." << std::endl;
            std::cout << "  --remove-empty-lines    (Optional) Remove lines that become empty after normalization (requires --normalize-whitespace)." << std::endl;
            std::cout << "  --to-lowercase          (Optional) Convert all text to lowercase." << std::endl;
            std::cout << "  --remove-non-printable  (Optional) Remove non-printable ASCII characters (keeps tab, newline, carriage return)." << std::endl;
            std::cout << "  --process-urls          (Optional) Process URLs. If --url-placeholder is empty, URLs are removed." << std::endl;
            std::cout << "  --url-placeholder <str> (Optional) Replace URLs with this string. Effective if --process-urls is set." << std::endl;
            std::cout << "  --process-emails        (Optional) Process email addresses. If --email-placeholder is empty, emails are removed." << std::endl;
            std::cout << "  --email-placeholder <str>(Optional) Replace emails with this string. Effective if --process-emails is set." << std::endl;
            std::cout << "  --remove-exact-duplicates (Optional) Remove exact duplicate lines (after other processing)." << std::endl;
            return false; // Exit after showing help
        } else if (arg == "--input" && i + 1 < argc) args.input_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.output_path = argv[++i];
        else if (arg == "--input-pattern" && i + 1 < argc) args.input_pattern = argv[++i];
        else if (arg == "--recursive") args.recursive_search = true;
        else if (arg == "--normalize-whitespace") args.normalize_whitespace = true;
        else if (arg == "--remove-empty-lines") args.remove_empty_lines_after_ws_norm = true;
        else if (arg == "--to-lowercase") args.to_lowercase = true;
        else if (arg == "--remove-non-printable") args.remove_non_printable = true;
        else if (arg == "--process-urls") args.process_urls = true;
        else if (arg == "--url-placeholder" && i + 1 < argc) args.url_placeholder = argv[++i];
        else if (arg == "--process-emails") args.process_emails = true;
        else if (arg == "--email-placeholder" && i + 1 < argc) args.email_placeholder = argv[++i];
        else if (arg == "--remove-exact-duplicates") args.remove_exact_duplicate_lines = true;
        else { std::cerr << "Error: Unknown argument or missing value for argument: " << arg << std::endl; return false; }
    }

    if (args.input_path.empty() || args.output_path.empty()) {
        std::cerr << "Error: --input and --output paths are required arguments." << std::endl;
        if (!(argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) {
            std::cerr << "Use -h or --help for usage information." << std::endl;
        }
        return false;
    }
    if (args.remove_empty_lines_after_ws_norm && !args.normalize_whitespace) {
        std::cerr << "Warning: --remove-empty-lines is only effective if --normalize-whitespace is also enabled. Option --remove-empty-lines will be ignored." << std::endl;
        args.remove_empty_lines_after_ws_norm = false;
    }
    if (!args.url_placeholder.empty() && !args.process_urls) {
        std::cerr << "Warning: --url-placeholder is specified, but --process-urls is not. URLs will not be processed." << std::endl;
    }
    if (!args.email_placeholder.empty() && !args.process_emails) {
        std::cerr << "Warning: --email-placeholder is specified, but --process-emails is not. Emails will not be processed." << std::endl;
    }
    return true;
}

// --- String Cleaning Helper Functions ---
static inline void ltrim_inplace(std::string &s) { s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch){ return !std::isspace(ch); })); }
static inline void rtrim_inplace(std::string &s) { s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), s.end()); }
static inline void trim_inplace(std::string &s) { ltrim_inplace(s); rtrim_inplace(s); }

std::string reduce_internal_whitespaces(const std::string& input_str) {
    std::string result; result.reserve(input_str.length()); bool last_was_space = false;
    for (char c : input_str) { if (std::isspace(static_cast<unsigned char>(c))) { if (!last_was_space) { result += ' '; last_was_space = true; }} else { result += c; last_was_space = false; }}
    if (result == " ") return ""; return result;
}
std::string apply_remove_non_printable(const std::string& s) {
    std::string result; result.reserve(s.length());
    for (char c : s) { if (std::isprint(static_cast<unsigned char>(c)) || c == '\t' || c == '\n' || c == '\r') { result += c; }}
    return result;
}

// Pre-compiled regex for performance
const std::regex url_regex(R"((?:https?://|ftp://|www\.)[^\s/$.?#].[^\s]*)", std::regex_constants::icase | std::regex_constants::optimize);
const std::regex email_regex(R"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b)", std::regex_constants::optimize);

// --- Core File Processing Logic ---
bool process_single_file(
    const std::filesystem::path& input_file_path,
    const std::filesystem::path& output_file_path,
    const CleanArgs& args,
    long long& file_lines_read_stat, long long& file_lines_written_stat,
    long long& file_lines_became_empty_stat, long long& file_lines_removed_duplicate_stat,
    long long& file_urls_processed_lines_stat, long long& file_emails_processed_lines_stat
) {
    std::cout << "  Processing: " << input_file_path.string() << "\n    Output to : " << output_file_path.string() << std::endl;

    std::ifstream infile(input_file_path);
    if (!infile.is_open()) {
        std::cerr << "    Error: Could not open input file '" << input_file_path.string() << "'." << std::endl;
        return false;
    }

    if (output_file_path.has_parent_path()) {
        try {
            if (!std::filesystem::exists(output_file_path.parent_path())) {
                std::filesystem::create_directories(output_file_path.parent_path());
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "    Error creating output directory for '" << output_file_path.string() << "': " << e.what() << std::endl;
            infile.close();
            return false;
        }
    }
    std::ofstream outfile(output_file_path);
    if (!outfile.is_open()) {
        std::cerr << "    Error: Could not open output file '" << output_file_path.string() << "' for writing." << std::endl;
        infile.close();
        return false;
    }

    std::string line;
    std::set<std::string> seen_lines_for_deduplication; // Per-file deduplication

    while (std::getline(infile, line)) {
        file_lines_read_stat++;
        std::string current_line = line;

        if (args.remove_non_printable) current_line = apply_remove_non_printable(current_line);

        bool url_found_in_line = false;
        if (args.process_urls) {
            std::string temp_line = std::regex_replace(current_line, url_regex, args.url_placeholder);
            if (temp_line != current_line) url_found_in_line = true;
            current_line = temp_line;
        }
        if(url_found_in_line) file_urls_processed_lines_stat++;

        bool email_found_in_line = false;
        if (args.process_emails) {
            std::string temp_line = std::regex_replace(current_line, email_regex, args.email_placeholder);
            if (temp_line != current_line) email_found_in_line = true;
            current_line = temp_line;
        }
        if(email_found_in_line) file_emails_processed_lines_stat++;

        if (args.normalize_whitespace) {
            trim_inplace(current_line);
            current_line = reduce_internal_whitespaces(current_line);
            trim_inplace(current_line);
        }
        if (args.to_lowercase) {
            std::transform(current_line.begin(), current_line.end(), current_line.begin(),
                           [](unsigned char c){ return std::tolower(c); });
        }
        if (current_line.empty()) {
            file_lines_became_empty_stat++;
            if (args.remove_empty_lines_after_ws_norm && args.normalize_whitespace) continue;
        }
        if (args.remove_exact_duplicate_lines) {
            if (seen_lines_for_deduplication.count(current_line)) {
                file_lines_removed_duplicate_stat++; continue;
            }
            seen_lines_for_deduplication.insert(current_line);
        }
        outfile << current_line << std::endl;
        file_lines_written_stat++;
    }
    infile.close(); outfile.close();
    std::cout << "    Finished processing: " << input_file_path.filename().string() << std::endl;
    return true;
}

// Helper function to handle file matching and processing logic for directory iteration
void handle_file_processing(
    const std::filesystem::directory_entry& dir_entry,
    const CleanArgs& args,
    const std::filesystem::path& input_base_path,
    const std::filesystem::path& output_base_path,
    long long& total_files_processed,
    long long& total_lines_read_overall, long long& total_lines_written_overall,
    long long& total_lines_became_empty_overall, long long& total_lines_removed_duplicate_overall,
    long long& total_urls_processed_lines_overall, long long& total_emails_processed_lines_overall
) {
    bool matches_pattern = false;
    std::string filename_str = dir_entry.path().filename().string();
    const std::string& pattern_to_match = args.input_pattern;

    // Simple pattern matching logic (can be expanded with regex for full glob)
    if (pattern_to_match == "*.*" || pattern_to_match == "*") {
        matches_pattern = true;
    } else if (pattern_to_match.rfind("*.", 0) == 0) { // Starts with *. e.g. *.txt
        std::string ext_to_match = pattern_to_match.substr(1); // Gets .txt
        if (dir_entry.path().extension().string() == ext_to_match) {
            matches_pattern = true;
        }
    } else if (pattern_to_match.find("*") == std::string::npos) { // No wildcards, exact match
        if (filename_str == pattern_to_match) matches_pattern = true;
    } else {
        // Basic wildcard support: * at start, end, or both (e.g. prefix_*, *_suffix, prefix_*_suffix)
        // This is still limited compared to full glob. For more complex needs, a glob library is better.
        std::regex pattern_regex(std::regex_replace(pattern_to_match, std::regex("\\*"), ".*"));
        if (std::regex_match(filename_str, pattern_regex)) {
            matches_pattern = true;
        }
    }

    if (matches_pattern) {
        std::filesystem::path current_input_file = dir_entry.path();
        std::filesystem::path relative_path = std::filesystem::relative(current_input_file, input_base_path);
        std::filesystem::path current_output_file = output_base_path / relative_path;

        long long current_file_lines_read = 0, current_file_lines_written = 0;
        long long current_file_became_empty = 0, current_file_removed_duplicate = 0;
        long long current_file_urls_processed = 0, current_file_emails_processed = 0;

        if(process_single_file(current_input_file, current_output_file, args,
            current_file_lines_read, current_file_lines_written,
            current_file_became_empty, current_file_removed_duplicate,
            current_file_urls_processed, current_file_emails_processed)) {
            total_files_processed++;
        total_lines_read_overall += current_file_lines_read;
        total_lines_written_overall += current_file_lines_written;
        total_lines_became_empty_overall += current_file_became_empty;
        total_lines_removed_duplicate_overall += current_file_removed_duplicate;
        total_urls_processed_lines_overall += current_file_urls_processed;
        total_emails_processed_lines_overall += current_file_emails_processed;
            }
    }
}


int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();

    CleanArgs args;
    if (!parse_clean_arguments(argc, argv, args)) {
        return (argc <= 1 || (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) ? 0 : 1;
    }

    std::cout << "--- Lunaris Text Cleaner (C++ v0.3.1) ---" << std::endl;
    std::cout << "Input path: " << args.input_path << std::endl;
    std::cout << "Output path: " << args.output_path << std::endl;
    if (std::filesystem::is_directory(args.input_path)) {
        std::cout << "Input file pattern: \"" << args.input_pattern << "\"" << (args.recursive_search ? " (recursive)" : "") << std::endl;
    }
    std::ostringstream options_ss;
    if(args.normalize_whitespace) options_ss << "NormalizeWhitespace ";
    if(args.remove_empty_lines_after_ws_norm) options_ss << "RemoveEmptyLines ";
    if(args.to_lowercase) options_ss << "ToLowercase ";
    if(args.remove_non_printable) options_ss << "RemoveNonPrintable ";
    if(args.process_urls) options_ss << "ProcessURLs" << (args.url_placeholder.empty() ? "[remove] " : "[replace_with:\"" + args.url_placeholder + "\"] ");
    if(args.process_emails) options_ss << "ProcessEmails" << (args.email_placeholder.empty() ? "[remove] " : "[replace_with:\"" + args.email_placeholder + "\"] ");
    if(args.remove_exact_duplicate_lines) options_ss << "RemoveExactDuplicates ";
    std::string options_str = options_ss.str();
    std::cout << "Options: " << (options_str.empty() ? "None" : options_str) << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;

    long long total_files_processed = 0;
    long long total_lines_read_overall = 0;
    long long total_lines_written_overall = 0;
    long long total_lines_became_empty_overall = 0;
    long long total_lines_removed_duplicate_overall = 0;
    long long total_urls_processed_lines_overall = 0;
    long long total_emails_processed_lines_overall = 0;

    std::filesystem::path input_fs_path(args.input_path);
    std::filesystem::path output_fs_path(args.output_path);

    if (std::filesystem::is_regular_file(input_fs_path)) {
        std::cout << "Mode: Processing a single file." << std::endl;
        if (std::filesystem::is_directory(output_fs_path)) {
            output_fs_path /= input_fs_path.filename();
        } else if (output_fs_path.has_parent_path()) { // Ensure parent of output file exists
            if (!std::filesystem::exists(output_fs_path.parent_path())) {
                std::filesystem::create_directories(output_fs_path.parent_path());
            }
        }
        if(process_single_file(input_fs_path, output_fs_path, args,
            total_lines_read_overall, total_lines_written_overall,
            total_lines_became_empty_overall, total_lines_removed_duplicate_overall,
            total_urls_processed_lines_overall, total_emails_processed_lines_overall)) {
            total_files_processed = 1;
            }
    } else if (std::filesystem::is_directory(input_fs_path)) {
        std::cout << "Mode: Processing directory..." << std::endl;
        if (std::filesystem::exists(output_fs_path) && !std::filesystem::is_directory(output_fs_path)) {
            std::cerr << "Error: Input is a directory, but output path '" << args.output_path << "' is an existing file. Output must be a directory for directory input." << std::endl;
            return 1;
        }
        // No need to create_directories for output_fs_path here,
        // process_single_file and handle_file_processing will create subdirs as needed.

        std::filesystem::directory_options dir_iter_options = std::filesystem::directory_options::follow_directory_symlink;
        // Potentially add std::filesystem::directory_options::skip_permission_denied if needed

        if (args.recursive_search) {
            for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(input_fs_path, dir_iter_options)) {
                if (dir_entry.is_regular_file()) {
                    handle_file_processing(dir_entry, args, input_fs_path, output_fs_path,
                                           total_files_processed, total_lines_read_overall, total_lines_written_overall,
                                           total_lines_became_empty_overall, total_lines_removed_duplicate_overall,
                                           total_urls_processed_lines_overall, total_emails_processed_lines_overall);
                }
            }
        } else {
            for (const auto& dir_entry : std::filesystem::directory_iterator(input_fs_path, dir_iter_options)) {
                if (dir_entry.is_regular_file()) {
                    handle_file_processing(dir_entry, args, input_fs_path, output_fs_path,
                                           total_files_processed, total_lines_read_overall, total_lines_written_overall,
                                           total_lines_became_empty_overall, total_lines_removed_duplicate_overall,
                                           total_urls_processed_lines_overall, total_emails_processed_lines_overall);
                }
            }
        }
    } else {
        std::cerr << "Error: Input path '" << args.input_path << "' is not a valid file or directory." << std::endl;
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "\n--- Overall Processing Summary ---" << std::endl;
    if (total_files_processed > 0) {
        std::cout << "Total files processed: " << total_files_processed << std::endl;
        std::cout << "Total lines read from input: " << total_lines_read_overall << std::endl;
        if (args.remove_non_printable) std::cout << "Non-printable character removal was applied." << std::endl;
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

    return 0;
}
