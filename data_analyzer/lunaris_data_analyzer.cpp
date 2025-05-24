#include <iostream>          // For std::cout, std::cerr, std::endl
#include <fstream>           // For std::ifstream
#include <vector>            // For std::vector
#include <string>            // For std::string
#include <cstdint>           // For int32_t, int16_t
#include <stdexcept>         // For std::runtime_error
#include <algorithm>         // For std::min, std::sort
#include <iomanip>           // For std::fixed, std::setprecision
#include <map>               // For std::map (token frequency counting)
#include <filesystem>        // For std::filesystem::file_size (C++17)
#include <sstream>           // For std::stringstream (used in BPE vocab parsing if more complex)
#include "nlohmann/json.hpp" // For parsing HF JSON vocab files

// For mmap on Linux/POSIX systems
#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// Structure to hold parsed command-line arguments
struct Args {
    std::string filepath;
    long long num_sequences = 0;
    int max_length = 0;
    std::string dtype_str = "int32";
    int vocab_size = -1;
    int sequences_to_print = 3;
    int tokens_per_line_print = 16;
    int report_top_n_tokens = 0;
    bool use_mmap = true;
    long long pad_id = 0;       // NEW: Added pad_id argument with default 0
    std::string vocab_file_path;
    std::string tokenizer_type_str;
    std::string output_format_str = "text"; // Default to text output
};

// Simple command-line argument parser
bool parse_arguments(int argc, char* argv[], Args& args) {
    if (argc == 1) {
        argv[argc++] = (char*)"--help";
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--help" || arg == "-h")) {
            std::cout << "Lunaris Data Analyzer Usage (C++ Version):" << std::endl; // MODIFIED: Added version context
            std::cout << "  --file <path>                (Required) Path to the .memmap file." << std::endl;
            std::cout << "  --num_sequences <long>       (Required) Expected number of sequences." << std::endl;
            std::cout << "  --max_length <int>           (Required) Expected max length of sequences." << std::endl;
            std::cout << "  --dtype <str>                (Optional) Data type (int32, int16). Default: int32." << std::endl;
            std::cout << "  --pad_id <long>              (Optional) Token ID used for padding. Default: 0." << std::endl; // NEW: Help for pad_id
            std::cout << "  --vocab_size <int>           (Optional) Expected vocabulary size for token ID validation. Default: -1 (disabled)." << std::endl;
            std::cout << "  --print_seq <int>            (Optional) Number of sequences to print from start. Default: 3." << std::endl;
            std::cout << "  --top_n_tokens <int>         (Optional) Report frequency of top N tokens. Default: 0 (disabled)." << std::endl;
            std::cout << "  --vocab_file <path>          (Optional) Path to the vocabulary file (e.g., for token decoding)." << std::endl;
            std::cout << "  --tokenizer_type <type>        (Optional) Type of tokenizer vocabulary (e.g., 'bpe', 'hf_json'). Required if --vocab_file is used." << std::endl;
            std::cout << "  --output_format <format>     (Optional) Output format for statistics ('text', 'json'). Default: text." << std::endl;
            std::cout << "  --no_mmap                    (Optional) Disable mmap and use slower ifstream (e.g., for non-Linux or testing)." << std::endl;
            std::cout << "  -h, --help                   Show this help message." << std::endl;
            return false;
        } else if (arg == "--file" && i + 1 < argc) {
            args.filepath = argv[++i];
        } else if (arg == "--num_sequences" && i + 1 < argc) {
            try { args.num_sequences = std::stoll(argv[++i]); } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for --num_sequences: " << argv[i] << " (" << e.what() << ")" << std::endl; return false; }
        } else if (arg == "--max_length" && i + 1 < argc) {
            try { args.max_length = std::stoi(argv[++i]); } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for --max_length: " << argv[i] << " (" << e.what() << ")" << std::endl; return false; }
        } else if (arg == "--dtype" && i + 1 < argc) {
            args.dtype_str = argv[++i];
            if (args.dtype_str != "int32" && args.dtype_str != "int16") {
                std::cerr << "Error: Invalid value for --dtype. Must be 'int32' or 'int16'." << std::endl; return false;
            }
        } else if (arg == "--pad_id" && i + 1 < argc) { // NEW: Parsing for pad_id
            try { args.pad_id = std::stoll(argv[++i]); } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for --pad_id: " << argv[i] << " (" << e.what() << ")" << std::endl; return false; }
        } else if (arg == "--vocab_size" && i + 1 < argc) {
            try { args.vocab_size = std::stoi(argv[++i]); } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for --vocab_size: " << argv[i] << " (" << e.what() << ")" << std::endl; return false; }
        } else if (arg == "--print_seq" && i + 1 < argc) {
            try { args.sequences_to_print = std::stoi(argv[++i]); } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for --print_seq: " << argv[i] << " (" << e.what() << ")" << std::endl; return false; }
        } else if (arg == "--top_n_tokens" && i + 1 < argc) {
            try { args.report_top_n_tokens = std::stoi(argv[++i]); } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for --top_n_tokens: " << argv[i] << " (" << e.what() << ")" << std::endl; return false; }
        } else if (arg == "--vocab_file" && i + 1 < argc) {
            args.vocab_file_path = argv[++i];
        } else if (arg == "--tokenizer_type" && i + 1 < argc) {
            args.tokenizer_type_str = argv[++i];
            if (args.tokenizer_type_str != "bpe" && args.tokenizer_type_str != "hf_json") { // Basic validation
                std::cerr << "Warning: Invalid value for --tokenizer_type: " << args.tokenizer_type_str
                          << ". Expected 'bpe' or 'hf_json'. Continuing, but tokenizer features might not work as expected." << std::endl;
            }
        } else if (arg == "--output_format" && i + 1 < argc) {
            args.output_format_str = argv[++i];
            if (args.output_format_str != "text" && args.output_format_str != "json") {
                std::cerr << "Error: Invalid value for --output_format. Must be 'text' or 'json'." << std::endl; return false;
            }
        } else if (arg == "--no_mmap") {
            args.use_mmap = false;
        } else {
            std::cerr << "Error: Unknown argument or missing value for: " << arg << ". Use --help for usage." << std::endl;
            return false;
        }
    }
    if (args.filepath.empty() || args.num_sequences <= 0 || args.max_length <= 0) {
        std::cerr << "Error: --file, --num_sequences, and --max_length are required and must be positive." << std::endl;
        if (argc > 1 && std::string(argv[1]) != "--help" && std::string(argv[1]) != "-h") {
            std::cerr << "Use -h or --help for usage information." << std::endl;
        }
        return false;
    }

    // Validation for vocab_file and tokenizer_type dependency
    if (!args.vocab_file_path.empty() && args.tokenizer_type_str.empty()) {
        std::cerr << "Error: --tokenizer_type is required if --vocab_file is provided." << std::endl;
        return false;
    }
    // It's also valid to provide tokenizer_type without vocab_file in some scenarios, so no check for the reverse.

    return true;
}

size_t get_type_size_bytes(const std::string& dtype_str) {
    if (dtype_str == "int32") return sizeof(int32_t);
    if (dtype_str == "int16") return sizeof(int16_t);
    return 0;
}

std::string get_type_name(const std::string& dtype_str) {
    if (dtype_str == "int32") return "int32_t";
    if (dtype_str == "int16") return "int16_t";
    return "unknown_type";
}

// Type alias for vocabulary map
// Stores mapping from token ID (int) to token string (std::string)
using VocabMap = std::map<int, std::string>;

// Function to load vocabulary from file based on tokenizer type.
// Supported types:
// - "bpe": Assumes a file where each line is a token string. The token ID is its 0-indexed line number.
//          Example BPE line: "Ġhello"
// - "hf_json": Assumes a Hugging Face tokenizer.json file structure, specifically
//              looking for `json_data["model"]["vocab"]` which is expected to be an
//              object mapping token strings to integer IDs.
//              Example HF JSON: {"model": {"vocab": {"<unk>": 0, "<s>": 1, "Ġhello": 123}}}
bool load_vocabulary(const std::string& vocab_path, const std::string& tokenizer_type, VocabMap& vocab_map) {
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "Error: Could not open vocabulary file: " << vocab_path << std::endl;
        return false;
    }

    vocab_map.clear(); // Ensure the map is empty before loading

    if (tokenizer_type == "bpe") {
        std::string line;
        int token_id = 0;
        while (std::getline(vocab_file, line)) {
            // Assuming one token string per line, ID is the line number (0-indexed)
            // More complex BPE files might have "token score" or "token id", requiring std::stringstream
            // For now, simple line = token_string
            // Minimal processing: remove potential BOM or leading/trailing whitespace if necessary.
            // For this task, we'll assume clean lines.
            if (!line.empty()) { // Avoid adding empty strings if there are blank lines
                 vocab_map[token_id++] = line;
            }
        }
        if (vocab_map.empty() && token_id == 0) { // Check if file was empty or only contained empty lines
             std::cerr << "Warning: BPE vocabulary file '" << vocab_path << "' is empty or contains no valid token lines." << std::endl;
             // It's not necessarily a fatal error, could be an empty vocab, so return true.
        }
    } else if (tokenizer_type == "hf_json") {
        try {
            nlohmann::json json_data;
            vocab_file >> json_data; // Parse the JSON from the file stream

            if (!json_data.contains("model") || !json_data["model"].contains("vocab")) {
                std::cerr << "Error: Invalid HF JSON vocabulary format in '" << vocab_path
                          << "'. Missing 'model' or 'model.vocab' keys." << std::endl;
                vocab_file.close();
                return false;
            }

            const auto& vocab_obj = json_data["model"]["vocab"];
            if (!vocab_obj.is_object()) {
                 std::cerr << "Error: 'model.vocab' in HF JSON '" << vocab_path << "' is not a JSON object." << std::endl;
                 vocab_file.close();
                 return false;
            }

            for (const auto& item : vocab_obj.items()) {
                if (!item.value().is_number_integer()) {
                    std::cerr << "Warning: Token ID for token '" << item.key() << "' in '" << vocab_path << "' is not an integer. Skipping." << std::endl;
                    continue;
                }
                vocab_map[item.value().get<int>()] = item.key();
            }

            if (vocab_map.empty() && vocab_obj.empty()) {
                 std::cerr << "Warning: HF JSON vocabulary file '" << vocab_path << "' has an empty 'model.vocab' object." << std::endl;
                 // Not necessarily fatal, return true.
            }

        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "Error: Failed to parse HF JSON vocabulary file '" << vocab_path << "'. Detail: " << e.what() << std::endl;
            vocab_file.close();
            return false;
        } catch (const std::exception& e) { // Catch other potential exceptions
            std::cerr << "Error: An unexpected error occurred while processing HF JSON vocabulary '" << vocab_path << "': " << e.what() << std::endl;
            vocab_file.close();
            return false;
        }
    } else {
        std::cerr << "Error: Unknown or unsupported tokenizer_type: '" << tokenizer_type << "'." << std::endl;
        vocab_file.close();
        return false;
    }

    vocab_file.close(); // Ensure file is closed
    return true;
}


template<typename T>
void process_data_and_print_stats(const T* data_ptr, const Args& args, size_t total_tokens_in_file, const VocabMap& vocab_map) {
    // MODIFIED: Use args.pad_id instead of a hardcoded value
    long long padding_token_count = 0;
    long long out_of_vocab_count = 0;
    std::map<T, long long> token_frequencies;

    // First pass: Calculate basic statistics (padding, OOV, token frequencies)
    for (size_t i = 0; i < total_tokens_in_file; ++i) {
        T current_token = data_ptr[i];
        if (current_token == static_cast<T>(args.pad_id)) {
            padding_token_count++;
        }
        // Out-of-vocabulary check based on --vocab_size argument
        if (args.vocab_size > 0 && (current_token < 0 || current_token >= static_cast<T>(args.vocab_size))) {
            out_of_vocab_count++;
        }
        // Accumulate frequencies if top N report is requested
        if (args.report_top_n_tokens > 0) {
            token_frequencies[current_token]++;
        }
    }

    // Second part: Output statistics and sequences based on the chosen format
    if (args.output_format_str == "json") {
        nlohmann::json stats_json;
        // Populate JSON object with calculated statistics
        stats_json["total_tokens_in_file"] = total_tokens_in_file;
        stats_json["padding_token_id_used_for_stats"] = args.pad_id;
        stats_json["padding_token_count"] = padding_token_count;
        if (total_tokens_in_file > 0) {
            stats_json["padding_token_percentage"] = (static_cast<double>(padding_token_count) / total_tokens_in_file * 100.0);
        } else {
            stats_json["padding_token_percentage"] = 0.0;
        }

        if (args.vocab_size > 0) {
            stats_json["out_of_vocabulary_tokens"] = out_of_vocab_count;
            stats_json["vocabulary_size_for_oov"] = args.vocab_size;
        }

        if (args.report_top_n_tokens > 0 && !token_frequencies.empty()) {
            std::vector<std::pair<T, long long>> sorted_tokens(token_frequencies.begin(), token_frequencies.end());
            std::sort(sorted_tokens.begin(), sorted_tokens.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

            nlohmann::json top_tokens_json = nlohmann::json::array();
            for (int i = 0; i < std::min((size_t)args.report_top_n_tokens, sorted_tokens.size()); ++i) {
                T token_id = sorted_tokens[i].first;
                nlohmann::json token_info;
                token_info["token_id"] = token_id;
                // Decode token if vocabulary is available
                if (!vocab_map.empty()) {
                    auto it = vocab_map.find(static_cast<int>(token_id)); // Cast token_id to int for map lookup
                    if (it != vocab_map.end()) {
                        token_info["token_string"] = it->second; // Found in vocab
                    } else {
                        token_info["token_string"] = "UNK"; // Not found in vocab
                    }
                } else {
                    token_info["token_string"] = nullptr; // Vocabulary not provided/loaded
                }
                token_info["count"] = sorted_tokens[i].second;
                top_tokens_json.push_back(token_info);
            }
            stats_json["top_n_common_tokens"] = top_tokens_json;
        }
        std::cout << stats_json.dump(4) << std::endl;
        // Sequence printing is suppressed for JSON output of statistics
    } else { // Text output
        std::cout << "\nData Statistics:" << std::endl;
        std::cout << "  Total tokens in file     : " << total_tokens_in_file << std::endl;
        std::cout << "  Padding token ID used for stats : " << args.pad_id << std::endl;
        std::cout << "  Count of padding tokens  : " << padding_token_count;
        if (total_tokens_in_file > 0) {
            std::cout << " (" << std::fixed << std::setprecision(2)
            << (static_cast<double>(padding_token_count) / total_tokens_in_file * 100.0) << "%)";
        }
        std::cout << std::endl;

        if (args.vocab_size > 0) {
            std::cout << "  Out-of-vocabulary tokens : " << out_of_vocab_count
            << " (based on vocab_size " << args.vocab_size << ")" << std::endl;
        }

        if (args.report_top_n_tokens > 0 && !token_frequencies.empty()) {
            std::vector<std::pair<T, long long>> sorted_tokens(token_frequencies.begin(), token_frequencies.end());
            std::sort(sorted_tokens.begin(), sorted_tokens.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

            std::cout << "  Top " << std::min((size_t)args.report_top_n_tokens, sorted_tokens.size()) << " most common tokens:" << std::endl;
            for (int i = 0; i < std::min((size_t)args.report_top_n_tokens, sorted_tokens.size()); ++i) {
                T token_id = sorted_tokens[i].first;
                std::cout << "    Token ID " << token_id;
                if (!vocab_map.empty()) {
                    auto it = vocab_map.find(static_cast<int>(token_id));
                    if (it != vocab_map.end()) {
                        std::cout << " (\"" << it->second << "\")";
                    } else {
                        std::cout << " (UNK)";
                    }
                }
                std::cout << ": " << sorted_tokens[i].second << " occurrences" << std::endl;
            }
        }
        std::cout << std::endl;

        long long num_seq_to_display = std::min(args.num_sequences, (long long)args.sequences_to_print);
        if (args.sequences_to_print > 0 && args.num_sequences > 0 && num_seq_to_display == 0) {
            num_seq_to_display = 1;
        } else if (args.sequences_to_print < 0) {
            num_seq_to_display = 0;
        }

        if (num_seq_to_display > 0) {
            std::cout << "Displaying first " << num_seq_to_display << " sequence(s):" << std::endl;
            for (long long seq_idx = 0; seq_idx < num_seq_to_display; ++seq_idx) {
                std::cout << "Sequence " << seq_idx << " (Tokens): [";
                for (int token_idx = 0; token_idx < args.max_length; ++token_idx) {
                    size_t current_token_flat_index = seq_idx * args.max_length + token_idx;
                    T current_token_id = data_ptr[current_token_flat_index];
                    std::cout << current_token_id;
                    // Decode token if vocabulary is available
                    if (!vocab_map.empty()) {
                        auto it = vocab_map.find(static_cast<int>(current_token_id)); // Cast token_id to int for map lookup
                        if (it != vocab_map.end()) {
                            std::cout << " (\"" << it->second << "\")"; // Found in vocab
                        } else {
                            std::cout << " (UNK)"; // Not found in vocab
                        }
                    }
                    if (token_idx < args.max_length - 1) std::cout << ", ";
                    if ((token_idx + 1) % args.tokens_per_line_print == 0 && token_idx < args.max_length - 1) {
                        std::cout << "\n                   "; // Indent for multi-line sequence display
                    }
                }
                std::cout << "]" << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    Args args;
    if (!parse_arguments(argc, argv, args)) {
        return (argc <= 1 || (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) ? 0 : 1;
    }

    // Print initial configuration details if not outputting JSON statistics directly
    // (as JSON output should be self-contained and typically piped to other tools)
    if (args.output_format_str != "json") {
        std::cout << "--- Lunaris Data Analyzer (C++ Version) ---" << std::endl;
        std::cout << "Target file: " << args.filepath << std::endl;
        std::cout << "Expected shape: (" << args.num_sequences << ", " << args.max_length << ")" << std::endl;
        std::cout << "Expected data type: " << get_type_name(args.dtype_str) << std::endl;
        std::cout << "Padding ID for statistics: " << args.pad_id << std::endl;
        if (args.vocab_size > 0) std::cout << "Expected vocab size: " << args.vocab_size << std::endl;
        if (!args.vocab_file_path.empty()) {
            std::cout << "Vocabulary file: " << args.vocab_file_path << " (Type: " << args.tokenizer_type_str << ")" << std::endl;
        }
        std::cout << "Output format for statistics: " << args.output_format_str << std::endl;
        std::cout << "Attempting to use mmap: " << (args.use_mmap ? "Yes" : "No") << std::endl;
        std::cout << "-------------------------------------------\n" << std::endl;
    }

    size_t type_size = get_type_size_bytes(args.dtype_str);
    if (type_size == 0) {
        std::cerr << "Error: Unsupported dtype string '" << args.dtype_str << "' provided." << std::endl;
        return 1;
    }

    std::error_code fs_err_code;
    uintmax_t actual_file_size_bytes = 0;
    try { // NEW: Added try-catch for file_size
        actual_file_size_bytes = std::filesystem::file_size(args.filepath, fs_err_code);
        if (fs_err_code) { // Check error code immediately
            std::cerr << "Error: Could not get file size for '" << args.filepath << "'. Message: " << fs_err_code.message() << std::endl;
            return 1;
        }
    } catch (const std::filesystem::filesystem_error& e) { // Catch specific filesystem_error
        std::cerr << "Filesystem Error getting file size for '" << args.filepath << "': " << e.what() << std::endl;
        return 1;
    }


    long long expected_file_size_bytes = args.num_sequences * args.max_length * type_size;

    std::cout << "File Size Verification:" << std::endl;
    std::cout << "  Actual file size  : " << actual_file_size_bytes << " bytes" << std::endl;
    std::cout << "  Expected file size: " << expected_file_size_bytes << " bytes" << std::endl;

    if (actual_file_size_bytes != expected_file_size_bytes) {
        std::cerr << "Error: File size mismatch! Actual (" << actual_file_size_bytes
        << " bytes) vs Expected (" << expected_file_size_bytes << " bytes)." << std::endl;
        std::cerr << "Please verify --num_sequences, --max_length, and --dtype arguments or check file integrity." << std::endl;
        return 1;
    }
    std::cout << "  File size matches expected dimensions. OK." << std::endl << std::endl;

    if (actual_file_size_bytes == 0) {
        std::cout << "File is empty. No data to analyze." << std::endl;
        return 0;
    }

    size_t total_tokens_in_file = actual_file_size_bytes / type_size;
    bool mmap_successful = false;

    #ifdef __linux__
    if (args.use_mmap) {
        int fd = open(args.filepath.c_str(), O_RDONLY);
        if (fd == -1) {
            std::cerr << "Error: Could not open file with POSIX open() for mmap: " << args.filepath << std::endl;
            perror("open details");
        } else {
            void* mapped_region = mmap(NULL, actual_file_size_bytes, PROT_READ, MAP_PRIVATE, fd, 0);

            // It's generally safer to close fd after checking mmap result, though man pages say it can be closed immediately.
            // For robustness, check mmap first.
            if (mapped_region == MAP_FAILED) {
                std::cerr << "Error: mmap failed for file: " << args.filepath << std::endl;
                perror("mmap details");
                close(fd); // Close fd even if mmap failed
            } else {
                close(fd); // Close fd after successful mmap
                std::cout << "File successfully memory-mapped using POSIX mmap." << std::endl;
                
                VocabMap vocab_map; // Create VocabMap instance
                if (!args.vocab_file_path.empty()) {
                    if (load_vocabulary(args.vocab_file_path, args.tokenizer_type_str, vocab_map)) {
                        std::cout << "Vocabulary loaded successfully from: " << args.vocab_file_path << std::endl;
                    } else {
                        std::cerr << "Warning: Failed to load vocabulary from: " << args.vocab_file_path << ". Continuing without token decoding." << std::endl;
                        // vocab_map will be empty, and that's handled by process_data_and_print_stats
                    }
                }

                if (args.dtype_str == "int32") {
                    process_data_and_print_stats(static_cast<const int32_t*>(mapped_region), args, total_tokens_in_file, vocab_map);
                } else if (args.dtype_str == "int16") {
                    process_data_and_print_stats(static_cast<const int16_t*>(mapped_region), args, total_tokens_in_file, vocab_map);
                }

                if (munmap(mapped_region, actual_file_size_bytes) == -1) {
                    perror("munmap warning");
                }
                mmap_successful = true;
            }
        }
        if (!mmap_successful && args.use_mmap) { // If mmap was intended but failed
            std::cout << "mmap failed. If you wish to use ifstream (slower), re-run with --no_mmap or ensure file is accessible for mmap." << std::endl;
            return 1;
        }
    }
    #else
    if(args.use_mmap) {
        std::cout << "Warning: POSIX mmap is typically used on Linux. For other systems, mmap support in this build is disabled." << std::endl;
        std::cout << "Falling back to ifstream-based reading (statistics will be limited to sequence printing)." << std::endl;
        args.use_mmap = false;
    }
    #endif

    // Fallback to ifstream if --no_mmap or if mmap was not successful (and mmap was the chosen path)
    if (!mmap_successful && ( args.use_mmap || !std::filesystem::exists(args.filepath) ) ) { // The second part of condition ensures we only enter here if mmap was the desired path or file doesn't exist to avoid double processing
        // If mmap was not used (either by choice --no_mmap, or it failed, or on non-Linux where it's disabled by default)
        // and we are not already in the explicit ifstream path later.
        // This block now primarily handles the case where mmap was intended but failed,
        // or if initial file checks for mmap might have been bypassed (e.g. file removed after initial check but before open).
        // The explicit `if (!args.use_mmap)` block later handles the --no_mmap case.
        // This logic becomes a bit complex due to multiple conditions for mmap failure vs. explicit choice.
        // Simpler: The `!mmap_successful && args.use_mmap` in the mmap block handles mmap failure.
        // The `if (!args.use_mmap)` block later handles explicit ifstream.
        // This current combined block might be redundant or lead to double messages.
        // Let's simplify: The existing `if (!mmap_successful && args.use_mmap)` after the mmap attempt
        // already prints an error and exits if mmap was intended but failed.
        // The subsequent `if (!args.use_mmap)` block (which I will rename for clarity)
        // should handle the case where mmap was *not* attempted (either --no_mmap or non-Linux).

    } // This block seems problematic, will be addressed by refining the ifstream logic below.

    // Fallback to ifstream if mmap was not used (due to --no_mmap or non-Linux without mmap support)
    // OR if mmap was attempted and failed (mmap_successful would be false, and args.use_mmap might still be true here if it failed)
    // The `mmap_successful` flag is key.
    if (!mmap_successful) { // This means mmap was either not attempted, or attempted and failed.
        if (args.output_format_str == "json") {
            // If mmap was the chosen path but failed, or if --no_mmap was specified with JSON output.
            nlohmann::json error_json;
            error_json["status"] = "statistics_not_available";
             if (args.use_mmap && !mmap_successful) { // mmap was attempted but failed
                 error_json["reason"] = "mmap failed; detailed statistics require mmap. ifstream fallback does not compute them.";
             } else { // mmap was not attempted (e.g. --no_mmap, or non-POSIX build without mmap)
                 error_json["reason"] = "File not processed with mmap (e.g., --no_mmap or unsupported platform); ifstream path does not compute detailed statistics.";
             }
            std::cout << error_json.dump(4) << std::endl;
        } else { // Text output for ifstream path
            // Message already printed if mmap failed and was intended.
            // If mmap was not intended (--no_mmap or non-Linux), print info.
            if (!args.use_mmap || (args.use_mmap && !mmap_successful && std::string(getenv("TERM") ? getenv("TERM") : "") != "dumb")) { // Avoid redundant message if mmap failure already reported
                 if (args.use_mmap && !mmap_successful) {
                    // This case is already handled by the message "mmap failed. If you wish to use ifstream..."
                    // So we only print the "Using std::ifstream..." if mmap was NOT the primary choice that failed.
                 } else {
                    std::cout << "Using std::ifstream for file reading (detailed statistics like padding count and top tokens will not be calculated)." << std::endl;
                 }
            }
            std::ifstream file_stream(args.filepath, std::ios::binary);
            if (!file_stream.is_open()) {
                std::cerr << "Error: Could not open file with ifstream: " << args.filepath << std::endl;
                return 1;
            }

            long long num_seq_to_display_ifstream = std::min(args.num_sequences, (long long)args.sequences_to_print);
            if (args.sequences_to_print > 0 && args.num_sequences > 0 && num_seq_to_display_ifstream == 0) num_seq_to_display_ifstream = 1;
            else if (args.sequences_to_print < 0) num_seq_to_display_ifstream = 0;

            if (num_seq_to_display_ifstream > 0) {
                std::cout << "Displaying first " << num_seq_to_display_ifstream << " sequence(s) via ifstream (token decoding not available for this mode):" << std::endl;
                for (long long seq_idx = 0; seq_idx < num_seq_to_display_ifstream; ++seq_idx) {
                    std::cout << "Sequence " << seq_idx << " (Tokens): [";
                    bool read_error = false;
                    for (int token_idx = 0; token_idx < args.max_length; ++token_idx) {
                        if (args.dtype_str == "int32") {
                            int32_t token_value;
                            if(!file_stream.read(reinterpret_cast<char*>(&token_value), sizeof(token_value))){read_error=true;break;}
                            std::cout << token_value;
                        } else if (args.dtype_str == "int16") {
                            int16_t token_value;
                            if(!file_stream.read(reinterpret_cast<char*>(&token_value), sizeof(token_value))){read_error=true;break;}
                            std::cout << token_value;
                        }
                        if(token_idx < args.max_length - 1) std::cout << ", ";
                        if((token_idx + 1) % args.tokens_per_line_print == 0 && token_idx < args.max_length - 1) std::cout << "\n                 ";
                    }
                    if(read_error && !file_stream.eof()) {std::cerr << "\nError: Read error during ifstream for sequence " << seq_idx << "." << std::endl;}
                    else if (read_error && file_stream.eof()){std::cout << "...<EOF>";}
                    std::cout << "]" << std::endl;
                    if (read_error && !file_stream.eof()) break;
                }
            }
            file_stream.close();
        }
    }

    if (args.output_format_str != "json") { // Only print "Analysis complete." for text mode, as JSON is self-contained.
        std::cout << "\nAnalysis complete." << std::endl;
    }
    return 0;
}
