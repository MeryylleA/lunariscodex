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

// Structure to hold parsed command-line arguments
struct CleanArgs {
    std::string input_filepath;
    std::string output_filepath;
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

// Simple command-line argument parser
bool parse_clean_arguments(int argc, char* argv[], CleanArgs& args) {
    if (argc == 1) argv[argc++] = (char*)"--help";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--help" || arg == "-h")) {
            std::cout << "Lunaris Text Cleaner (C++ v0.2.1) Usage:" << std::endl; // Corrected version if needed
            std::cout << "  --input <path>          (Required) Path to the input text file." << std::endl;
            std::cout << "  --output <path>         (Required) Path to save the cleaned output text file." << std::endl;
            std::cout << "  --normalize-whitespace  (Optional) Trim and reduce multiple whitespaces to one." << std::endl;
            std::cout << "  --remove-empty-lines    (Optional) Remove lines that become empty after normalization (requires --normalize-whitespace)." << std::endl;
            std::cout << "  --to-lowercase          (Optional) Convert all text to lowercase." << std::endl;
            std::cout << "  --remove-non-printable  (Optional) Remove non-printable ASCII characters (keeps tab, newline, carriage return)." << std::endl;
            std::cout << "  --process-urls          (Optional) Process URLs. If --url-placeholder is empty, URLs are removed." << std::endl;
            std::cout << "  --url-placeholder <str> (Optional) Replace URLs with this string (e.g., \"<URL>\"). Effective if --process-urls is set." << std::endl;
            std::cout << "  --process-emails        (Optional) Process email addresses. If --email-placeholder is empty, emails are removed." << std::endl;
            std::cout << "  --email-placeholder <str>(Optional) Replace emails with this string (e.g., \"<EMAIL>\"). Effective if --process-emails is set." << std::endl;
            std::cout << "  --remove-exact-duplicates (Optional) Remove exact duplicate lines (after other processing)." << std::endl;
            std::cout << "  -h, --help              Show this help message." << std::endl;
            return false;
        } else if (arg == "--input" && i + 1 < argc) args.input_filepath = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.output_filepath = argv[++i];
        else if (arg == "--normalize-whitespace") args.normalize_whitespace = true;
        else if (arg == "--remove-empty-lines") args.remove_empty_lines_after_ws_norm = true;
        else if (arg == "--to-lowercase") args.to_lowercase = true;
        else if (arg == "--remove-non-printable") args.remove_non_printable = true;
        else if (arg == "--process-urls") args.process_urls = true;
        else if (arg == "--url-placeholder" && i + 1 < argc) args.url_placeholder = argv[++i];
        else if (arg == "--process-emails") args.process_emails = true;
        else if (arg == "--email-placeholder" && i + 1 < argc) args.email_placeholder = argv[++i];
        else if (arg == "--remove-exact-duplicates") args.remove_exact_duplicate_lines = true;
        else { std::cerr << "Error: Unknown argument or missing value: " << arg << std::endl; return false; }
    }
    if (args.input_filepath.empty() || args.output_filepath.empty()) {
        std::cerr << "Error: --input and --output file paths are required arguments." << std::endl;
        if (argc > 1 && std::string(argv[1]) != "--help" && std::string(argv[1]) != "-h") std::cerr << "Use -h or --help for usage information." << std::endl;
        return false;
    }
    if (args.remove_empty_lines_after_ws_norm && !args.normalize_whitespace) {
        std::cerr << "Warning: --remove-empty-lines is effective only with --normalize-whitespace. --remove-empty-lines will be ignored." << std::endl;
        args.remove_empty_lines_after_ws_norm = false;
    }
    if (!args.url_placeholder.empty() && !args.process_urls) {
        // Corrected: Use std::cerr for warnings in command-line tools
        std::cerr << "Warning: --url-placeholder is set, but --process-urls is not. URLs will not be processed or replaced." << std::endl;
    }
    if (!args.email_placeholder.empty() && !args.process_emails) {
        // Corrected: Use std::cerr for warnings
        std::cerr << "Warning: --email-placeholder is set, but --process-emails is not. Emails will not be processed or replaced." << std::endl;
    }
    return true;
}

// Helper: Trim from start (in place)
static inline void ltrim_inplace(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch){ return !std::isspace(ch); }));
}
// Helper: Trim from end (in place)
static inline void rtrim_inplace(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), s.end());
}
// Helper: Trim from both ends (in place)
static inline void trim_inplace(std::string &s) {
    ltrim_inplace(s); rtrim_inplace(s);
}
// Helper: Reduce multiple internal whitespaces to one, returns new string
std::string reduce_internal_whitespaces(const std::string& input_str) {
    std::string result;
    result.reserve(input_str.length());
    bool last_was_space = false;
    for (char c : input_str) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!last_was_space) { result += ' '; last_was_space = true; }
        } else { result += c; last_was_space = false; }
    }
    if (result == " ") return ""; // Handle case where original string was all spaces
    return result;
}

// Remove non-printable ASCII characters, keeping tab, newline, and carriage return
std::string apply_remove_non_printable(const std::string& s) {
    std::string result;
    result.reserve(s.length());
    for (char c : s) {
        if (std::isprint(static_cast<unsigned char>(c)) || c == '\t' || c == '\n' || c == '\r') {
            result += c;
        }
    }
    return result;
}

// Regex for matching URLs (simplified common cases, protocol is optional for http/https)
const std::regex url_regex(R"((?:https?://|ftp://|www\.)[^\s/$.?#].[^\s]*)", std::regex_constants::icase | std::regex_constants::optimize);
// Regex for matching emails (simplified common cases)
const std::regex email_regex(R"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b)", std::regex_constants::optimize);


int main(int argc, char* argv[]) {
    CleanArgs args;
    if (!parse_clean_arguments(argc, argv, args)) {
        return (argc <= 1 || (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) ? 0 : 1;
    }

    std::cout << "--- Lunaris Text Cleaner (C++ v0.2.1) ---" << std::endl;
    std::cout << "Input file: " << args.input_filepath << std::endl;
    std::cout << "Output file: " << args.output_filepath << std::endl;

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

    std::ifstream infile(args.input_filepath);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open input file '" << args.input_filepath << "'." << std::endl;
        return 1;
    }

    std::filesystem::path output_p(args.output_filepath);
    if (output_p.has_parent_path()) {
        try { // create_directories can throw if path is invalid or permissions issue
            if (!std::filesystem::exists(output_p.parent_path())) {
                std::filesystem::create_directories(output_p.parent_path());
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating output directory: " << e.what() << std::endl;
            infile.close();
            return 1;
        }
    }
    std::ofstream outfile(args.output_filepath);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file '" << args.output_filepath << "' for writing." << std::endl;
        infile.close();
        return 1;
    }

    std::string line;
    long long lines_read = 0;
    long long lines_written = 0;
    long long lines_became_empty_after_processing = 0;
    long long lines_removed_as_duplicate = 0;
    long long lines_affected_by_url_processing = 0;
    long long lines_affected_by_email_processing = 0;

    std::set<std::string> seen_lines_for_deduplication;

    std::cout << "Processing..." << std::endl;

    while (std::getline(infile, line)) {
        lines_read++;
        std::string current_line = line; // Work on a copy

        if (args.remove_non_printable) {
            current_line = apply_remove_non_printable(current_line);
        }

        if (args.process_urls) {
            std::string temp_line = std::regex_replace(current_line, url_regex, args.url_placeholder);
            if (temp_line != current_line) lines_affected_by_url_processing++;
            current_line = temp_line;
        }

        if (args.process_emails) {
            std::string temp_line = std::regex_replace(current_line, email_regex, args.email_placeholder);
            if (temp_line != current_line) lines_affected_by_email_processing++;
            current_line = temp_line;
        }

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
            lines_became_empty_after_processing++;
            // Only remove if both flags are set and the line is truly empty after all prior processing
            if (args.remove_empty_lines_after_ws_norm && args.normalize_whitespace) {
                continue;
            }
        }

        if (args.remove_exact_duplicate_lines) {
            // Deduplication happens after all other per-line transformations
            if (seen_lines_for_deduplication.count(current_line)) {
                lines_removed_as_duplicate++;
                continue;
            }
            seen_lines_for_deduplication.insert(current_line);
        }

        outfile << current_line << std::endl;
        lines_written++;
    }

    infile.close();
    outfile.close();

    std::cout << "\n--- Processing Summary ---" << std::endl;
    std::cout << "Lines read from input: " << lines_read << std::endl;
    if (args.remove_non_printable) std::cout << "Non-printable character removal applied." << std::endl;
    if (args.process_urls) std::cout << "Lines with URLs processed/replaced: " << lines_affected_by_url_processing << std::endl;
    if (args.process_emails) std::cout << "Lines with Emails processed/replaced: " << lines_affected_by_email_processing << std::endl;
    if (args.normalize_whitespace) std::cout << "Whitespace normalization applied." << std::endl;
    if (args.to_lowercase) std::cout << "Text converted to lowercase." << std::endl;
    std::cout << "Lines that became empty after processing: " << lines_became_empty_after_processing << std::endl;
    if (args.remove_empty_lines_after_ws_norm && args.normalize_whitespace) {
        std::cout << "  (These processed empty lines were removed from output)" << std::endl;
    }
    if (args.remove_exact_duplicate_lines) std::cout << "Exact duplicate lines removed (after all processing): " << lines_removed_as_duplicate << std::endl;
    std::cout << "Lines written to output: " << lines_written << std::endl;
    std::cout << "Cleaned file saved to: " << args.output_filepath << std::endl;
    std::cout << "--------------------------" << std::endl;

    return 0;
}
