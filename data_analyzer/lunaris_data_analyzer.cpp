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

template<typename T>
void process_data_and_print_stats(const T* data_ptr, const Args& args, size_t total_tokens_in_file) {
    // MODIFIED: Use args.pad_id instead of a hardcoded value
    long long padding_token_count = 0;
    long long out_of_vocab_count = 0;
    std::map<T, long long> token_frequencies;

    for (size_t i = 0; i < total_tokens_in_file; ++i) {
        T current_token = data_ptr[i];
        // MODIFIED: Compare with args.pad_id
        if (current_token == static_cast<T>(args.pad_id)) {
            padding_token_count++;
        }
        if (args.vocab_size > 0 && (current_token < 0 || current_token >= static_cast<T>(args.vocab_size))) {
            out_of_vocab_count++;
        }
        if (args.report_top_n_tokens > 0) {
            token_frequencies[current_token]++;
        }
    }

    std::cout << "\nData Statistics:" << std::endl;
    std::cout << "  Total tokens in file     : " << total_tokens_in_file << std::endl;
    // MODIFIED: Report the pad_id used for statistics
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
            std::cout << "    Token ID " << sorted_tokens[i].first
            << ": " << sorted_tokens[i].second << " occurrences" << std::endl;
        }
    }
    std::cout << std::endl;

    long long num_seq_to_display = std::min(args.num_sequences, (long long)args.sequences_to_print);
    // MODIFIED: Simplified logic for num_seq_to_display, ensures at least 1 if sequences_to_print > 0 and num_sequences > 0
    if (args.sequences_to_print > 0 && args.num_sequences > 0 && num_seq_to_display == 0) {
        num_seq_to_display = 1;
    } else if (args.sequences_to_print < 0) { // Handle negative input for print_seq
        num_seq_to_display = 0;
    }


    if (num_seq_to_display > 0) {
        std::cout << "Displaying first " << num_seq_to_display << " sequence(s):" << std::endl;
        for (long long seq_idx = 0; seq_idx < num_seq_to_display; ++seq_idx) {
            std::cout << "Sequence " << seq_idx << " (Tokens): [";
            for (int token_idx = 0; token_idx < args.max_length; ++token_idx) {
                size_t current_token_flat_index = seq_idx * args.max_length + token_idx;
                std::cout << data_ptr[current_token_flat_index];
                if (token_idx < args.max_length - 1) std::cout << ", ";
                if ((token_idx + 1) % args.tokens_per_line_print == 0 && token_idx < args.max_length - 1) {
                    std::cout << "\n                 ";
                }
            }
            std::cout << "]" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    Args args;
    if (!parse_arguments(argc, argv, args)) {
        return (argc <= 1 || (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) ? 0 : 1;
    }

    std::cout << "--- Lunaris Data Analyzer (C++ Version) ---" << std::endl;
    std::cout << "Target file: " << args.filepath << std::endl;
    std::cout << "Expected shape: (" << args.num_sequences << ", " << args.max_length << ")" << std::endl;
    std::cout << "Expected data type: " << get_type_name(args.dtype_str) << std::endl;
    // NEW: Report pad_id if it's not the default 0, or always report it.
    std::cout << "Padding ID for statistics: " << args.pad_id << std::endl;
    if (args.vocab_size > 0) std::cout << "Expected vocab size: " << args.vocab_size << std::endl;
    std::cout << "Attempting to use mmap: " << (args.use_mmap ? "Yes" : "No") << std::endl;
    std::cout << "-------------------------------------------\n" << std::endl;

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
                if (args.dtype_str == "int32") {
                    process_data_and_print_stats(static_cast<const int32_t*>(mapped_region), args, total_tokens_in_file);
                } else if (args.dtype_str == "int16") {
                    process_data_and_print_stats(static_cast<const int16_t*>(mapped_region), args, total_tokens_in_file);
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

    // Fallback to ifstream if --no_mmap or if mmap was not successful (and use_mmap was true on non-Linux, now false)
    if (!args.use_mmap) {
        std::cout << "Using std::ifstream for file reading (statistics like padding count and top tokens will not be calculated)." << std::endl;
        std::ifstream file_stream(args.filepath, std::ios::binary);
        if (!file_stream.is_open()) {
            std::cerr << "Error: Could not open file with ifstream: " << args.filepath << std::endl;
            return 1;
        }

        long long num_seq_to_display_ifstream = std::min(args.num_sequences, (long long)args.sequences_to_print);
        if (args.sequences_to_print > 0 && args.num_sequences > 0 && num_seq_to_display_ifstream == 0) num_seq_to_display_ifstream = 1;
        else if (args.sequences_to_print < 0) num_seq_to_display_ifstream = 0;


        if (num_seq_to_display_ifstream > 0) {
            std::cout << "Displaying first " << num_seq_to_display_ifstream << " sequence(s) via ifstream:" << std::endl;
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
                if (read_error && !file_stream.eof()) break; // Stop if there was a read error not at EOF
            }
        }
        file_stream.close();
    }

    std::cout << "\nAnalysis complete." << std::endl;
    return 0;
}
