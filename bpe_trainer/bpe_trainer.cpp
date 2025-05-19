#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <map>
#include <utility>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <cctype>
#include <cstdio>

#include <nlohmann/json.hpp>

// Namespace aliases
using json = nlohmann::json;
namespace fs = std::filesystem;

// Structure for BPE training statistics
struct BpeStats {
    size_t raw_corpus_char_count = 0;
    size_t initial_token_count_in_corpus = 0;
    size_t initial_vocab_size = 0;
    size_t final_vocab_size = 0;
    size_t total_merges = 0;
    std::chrono::milliseconds training_time{0};
};

// BPE Trainer Class
class BpeTrainer {
public:
    // Constructor now takes byte_level_mode
    BpeTrainer(bool verbose = false, bool byte_level_mode = true)
    : verbose_(verbose), byte_level_training_mode_(byte_level_mode) {}

    // train function no longer needs byte_level as a parameter
    bool train(const std::string& corpus_path,
               const std::string& output_dir,
               size_t target_vocab_size_param) {

        auto overall_start_time = std::chrono::high_resolution_clock::now();
        stats_ = BpeStats{}; // Reset stats for a new training run

        if (verbose_) {
            consoleLog("Starting BPE training process...");
            consoleLog("Mode: " + std::string(this->byte_level_training_mode_ ? "Byte-level" : "Word-level"));
            consoleLog("Loading corpus from file: " + corpus_path);
        }

        if (!ensureOutputDirectory(output_dir)) return false;

        // loadAndPretokenizeCorpus now uses the member variable this->byte_level_training_mode_
        if (!loadAndPretokenizeCorpus(corpus_path)) {
            consoleErr("Error: Failed to load or pretokenize the corpus.");
            return false;
        }

        // initializeVocabularyFromPretokenizedCorpus also uses the member variable
        initializeVocabularyFromPretokenizedCorpus();
        stats_.initial_vocab_size = vocab_.size();

        if (verbose_) {
            consoleLog("Corpus loaded. Raw characters read: " + std::to_string(stats_.raw_corpus_char_count));
            consoleLog("Initial tokens in pretokenized corpus: " + std::to_string(stats_.initial_token_count_in_corpus));
            consoleLog("Initial vocabulary size: " + std::to_string(stats_.initial_vocab_size) + " unique tokens.");
            consoleLog("Target vocabulary size parameter: " + std::to_string(target_vocab_size_param));
        }

        size_t actual_target_vocab_size = std::min(target_vocab_size_param, stats_.initial_vocab_size + 200000);
        if (target_vocab_size_param > stats_.initial_vocab_size + 200000 && verbose_) {
            consoleLog("Warning: Requested vocab_size is very large. Capping effective target to " + std::to_string(actual_target_vocab_size) + " (initial_vocab + 200k merges max).");
        }
        if (verbose_){
            consoleLog("Effective target vocabulary size for merging: " + std::to_string(actual_target_vocab_size));
        }


        if (actual_target_vocab_size <= vocab_.size()){
            if (verbose_) {
                consoleLog("Target vocabulary size (" + std::to_string(actual_target_vocab_size) +
                ") is already met or exceeded by initial vocabulary (" + std::to_string(vocab_.size()) + "). No merges will be performed.");
            }
        } else {
            if (verbose_) consoleLog("Starting BPE merge iterations...");
            while (vocab_.size() < actual_target_vocab_size) {
                auto pair_counts = countAdjacentPairs();
                if (pair_counts.empty()) {
                    if (verbose_) consoleLog("No more pairs to merge. Stopping iterations.");
                    break;
                }

                auto best_pair_info = findMostFrequentPair(pair_counts);
                if (best_pair_info.first.first.empty() || best_pair_info.second <= 1) { // Stop if best pair is insignificant
                    if (verbose_) consoleLog("No more significantly frequent pairs (frequency > 1) to merge. Stopping iterations.");
                    break;
                }

                const auto& pair_to_merge = best_pair_info.first;
                std::string new_token = pair_to_merge.first + pair_to_merge.second;

                if (vocab_.count(new_token)) {
                    if (verbose_ && stats_.total_merges < 5 + stats_.initial_vocab_size) { // Log first few occurrences of this
                        consoleLog("Skipping merge: new token '" + escapeTokenForDisplay(new_token) + "' already in vocab. This can happen.");
                    }
                    // To prevent potential infinite loops if the most frequent pair always creates an existing token:
                    // A more robust solution would be to remove this pair from consideration and find the next best.
                    // For now, we rely on the loop eventually breaking if no *new* tokens can be formed above freq 1.
                    // If all top pairs form existing tokens, best_pair_info.second might not decrease,
                    // but if only pairs with freq 1 remain, it will break.
                    // Let's check if any other viable merge exists
                    bool found_other_merge = false;
                    for(const auto& pc_entry : pair_counts){
                        if(pc_entry.second > 1 && !vocab_.count(pc_entry.first.first + pc_entry.first.second)){
                            found_other_merge = true;
                            break;
                        }
                    }
                    if(!found_other_merge && best_pair_info.second > 1){ // If best is still >1 but all new tokens from it are in vocab
                        if(verbose_) consoleLog("All remaining frequent pairs (freq > 1) result in existing tokens. Stopping iterations to prevent stall.");
                        break;
                    }
                    // If the best pair creates an existing token, but other *new* tokens could be formed from other pairs,
                    // ideally we'd remove the current best_pair from pair_counts and re-iterate findMostFrequentPair.
                    // For simplicity now, we let it try (it won't add to vocab, loop continues). If it gets stuck, the above check helps.
                }


                if (verbose_) {
                    consoleLog("Merge #" + std::to_string(stats_.total_merges + 1) +
                    ": '" + escapeTokenForDisplay(pair_to_merge.first) +
                    "' + '" + escapeTokenForDisplay(pair_to_merge.second) +
                    "' -> '" + escapeTokenForDisplay(new_token) +
                    "' (Frequency: " + std::to_string(best_pair_info.second) + ")");
                }

                applyMergeToCorpus(pair_to_merge.first, pair_to_merge.second, new_token);

                learned_merges_.push_back(pair_to_merge);
                stats_.total_merges++;
                vocab_.insert(new_token); // Will only insert if new_token isn't already there

                if (verbose_ && (stats_.total_merges % 100 == 0 || vocab_.size() % std::max((size_t)1, actual_target_vocab_size/100) == 0) ) {
                    consoleLog("Progress: " + std::to_string(vocab_.size()) + "/" + std::to_string(actual_target_vocab_size) +
                    " vocab tokens (" + formatDouble(vocab_.size() * 100.0 / actual_target_vocab_size, 2) + "%)");
                }
            }
        }


        stats_.final_vocab_size = vocab_.size();
        auto overall_end_time = std::chrono::high_resolution_clock::now();
        stats_.training_time = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time);

        if (verbose_) printStats();
        saveResults(output_dir);
        return true;
               }

private:
    bool verbose_;
    std::vector<std::vector<std::string>> pretokenized_corpus_;
    std::unordered_set<std::string> vocab_;
    std::vector<std::pair<std::string, std::string>> learned_merges_;
    BpeStats stats_;
    bool byte_level_training_mode_; // Stores the mode set at construction

    void consoleLog(const std::string& message) const {
        if(verbose_) std::cout << "[INFO] " << message << std::endl;
    }
    void consoleErr(const std::string& message) const {
        std::cerr << "[ERROR] " << message << std::endl;
    }

    std::string formatDouble(double val, int precision) const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision) << val;
        return oss.str();
    }

    std::string escapeTokenForDisplay(const std::string& token) const {
        std::string escaped_token;
        escaped_token.reserve(token.length());
        for (char c : token) {
            if (std::isprint(static_cast<unsigned char>(c))) {
                escaped_token += c;
            } else {
                char hex_repr[5]; // For "\\xHH\0"
                snprintf(hex_repr, sizeof(hex_repr), "\\x%02X", static_cast<unsigned char>(c));
                escaped_token += hex_repr;
            }
        }
        return escaped_token;
    }

    bool ensureOutputDirectory(const std::string& output_dir) {
        if (!fs::exists(output_dir)) {
            consoleLog("Output directory does not exist. Creating: " + output_dir);
            try {
                fs::create_directories(output_dir);
            } catch (const fs::filesystem_error& e) {
                consoleErr("Failed to create output directory '" + output_dir + "': " + e.what());
                return false;
            }
        }
        return true;
    }

    // loadAndPretokenizeCorpus now uses the member variable this->byte_level_training_mode_
    bool loadAndPretokenizeCorpus(const std::string& corpus_path) {
        std::ifstream file(corpus_path);
        if (!file.is_open()) {
            consoleErr("Could not open corpus file: " + corpus_path);
            return false;
        }

        pretokenized_corpus_.clear();
        stats_.initial_token_count_in_corpus = 0;
        stats_.raw_corpus_char_count = 0;

        std::string line;
        size_t line_number = 0;

        while (std::getline(file, line)) {
            line_number++;
            stats_.raw_corpus_char_count += line.length() + 1;

            if (line.empty()) {
                if(verbose_ && line_number % 50000 == 0) consoleLog("Processed " + std::to_string(line_number) + " lines...");
                continue;
            }

            std::vector<std::string> current_line_tokens;
            current_line_tokens.reserve(line.length()); // Pre-allocate assuming worst case (byte-level)

            if (this->byte_level_training_mode_) {
                for (unsigned char c : line) {
                    std::string byte_token_str;
                    if (c >= 32 && c <= 126) {
                        byte_token_str = std::string(1, c);
                    } else {
                        char hex_buf[5];
                        snprintf(hex_buf, sizeof(hex_buf), "\\x%02X", c);
                        byte_token_str = hex_buf;
                    }
                    current_line_tokens.push_back(byte_token_str);
                }
            } else {
                std::istringstream iss(line);
                std::string word_token;
                while (iss >> word_token) {
                    if (!word_token.empty()) {
                        current_line_tokens.push_back(word_token);
                    }
                }
            }

            if (!current_line_tokens.empty()) {
                pretokenized_corpus_.push_back(std::move(current_line_tokens));
                // Correctly update stats after move
                stats_.initial_token_count_in_corpus += pretokenized_corpus_.back().size();
            }
            if(verbose_ && line_number % 50000 == 0) consoleLog("Processed " + std::to_string(line_number) + " lines... Current initial tokens: " + std::to_string(stats_.initial_token_count_in_corpus));
        }
        file.close();

        if (pretokenized_corpus_.empty() && stats_.raw_corpus_char_count > 0) {
            consoleLog("Warning: Corpus was read but resulted in no pre-tokenized sequences. Check input or pre-tokenization logic.");
        }
        return !pretokenized_corpus_.empty() || stats_.raw_corpus_char_count == 0; // True if empty file or successfully tokenized
    }

    // initializeVocabularyFromPretokenizedCorpus now uses the member variable
    void initializeVocabularyFromPretokenizedCorpus() {
        vocab_.clear();
        if (this->byte_level_training_mode_) {
            for (int i = 0; i < 128; ++i) { // Base ASCII (0-127)
                if (i >= 32 && i <= 126) { // Printable ASCII
                    vocab_.insert(std::string(1, static_cast<char>(i)));
                } else { // Control characters and space (if not caught by printable)
                    char hex_buf[5];
                    snprintf(hex_buf, sizeof(hex_buf), "\\x%02X", static_cast<unsigned char>(i));
                    vocab_.insert(hex_buf);
                }
            }
        }
        for (const auto& sentence_tokens : pretokenized_corpus_) {
            for (const auto& token : sentence_tokens) {
                vocab_.insert(token);
            }
        }
    }

    std::map<std::pair<std::string, std::string>, size_t> countAdjacentPairs() {
        std::map<std::pair<std::string, std::string>, size_t> pair_counts;
        for (const auto& sentence_tokens : pretokenized_corpus_) {
            if (sentence_tokens.size() < 2) continue;
            for (size_t i = 0; i < sentence_tokens.size() - 1; ++i) {
                pair_counts[{sentence_tokens[i], sentence_tokens[i+1]}]++;
            }
        }
        return pair_counts;
    }

    std::pair<std::pair<std::string, std::string>, size_t> findMostFrequentPair(
        const std::map<std::pair<std::string, std::string>, size_t>& pair_counts) {
        std::pair<std::pair<std::string, std::string>, size_t> best_pair_info = {{std::string(), std::string()}, 0};
        if (pair_counts.empty()) return best_pair_info;

        for (const auto& entry : pair_counts) {
            if (entry.second > best_pair_info.second) {
                best_pair_info = entry;
            }
        }
        return best_pair_info;
        }

        void applyMergeToCorpus(const std::string& first, const std::string& second, const std::string& merged_token) {
            size_t new_total_token_count = 0;
            for (auto& sentence_tokens : pretokenized_corpus_) {
                if (sentence_tokens.empty()) continue;

                std::vector<std::string> new_sentence_tokens;
                new_sentence_tokens.reserve(sentence_tokens.size());

                size_t i = 0;
                while (i < sentence_tokens.size()) {
                    if (i + 1 < sentence_tokens.size() &&
                        sentence_tokens[i] == first &&
                        sentence_tokens[i+1] == second) {
                        new_sentence_tokens.push_back(merged_token);
                    i += 2;
                        } else {
                            new_sentence_tokens.push_back(sentence_tokens[i]);
                            i += 1;
                        }
                }
                sentence_tokens = std::move(new_sentence_tokens);
                new_total_token_count += sentence_tokens.size();
            }
            stats_.initial_token_count_in_corpus = new_total_token_count; // Update with new compressed count
        }

        void printStats() const {
            consoleLog("\n=== BPE Training Statistics ===");
            consoleLog("Raw characters read from corpus: " + std::to_string(stats_.raw_corpus_char_count));
            consoleLog("Token count in corpus (after current merges): " + std::to_string(stats_.initial_token_count_in_corpus));
            consoleLog("Initial vocabulary size (before any merges): " + std::to_string(stats_.initial_vocab_size));
            consoleLog("Final vocabulary size: " + std::to_string(stats_.final_vocab_size));
            consoleLog("Total merges performed: " + std::to_string(stats_.total_merges));
            consoleLog("Training time: " + formatDouble(stats_.training_time.count() / 1000.0, 3) + " seconds");
            consoleLog("===============================\n");
        }

        void saveResults(const std::string& output_dir) {
            json output_json;
            std::vector<std::string> vocab_list(vocab_.begin(), vocab_.end());
            std::sort(vocab_list.begin(), vocab_list.end());
            output_json["vocabulary"] = vocab_list;

            json merges_list_json = json::array();
            for (const auto& merge_pair : learned_merges_) {
                // Save merges with escaped tokens for display consistency
                merges_list_json.push_back({escapeTokenForDisplay(merge_pair.first), escapeTokenForDisplay(merge_pair.second)});
            }
            output_json["merges"] = merges_list_json;

            // Save the mode used for training, so a tokenizer loading this can know.
            output_json["bpe_config"] = {
                {"mode", this->byte_level_training_mode_ ? "byte" : "word"}
            };

            output_json["stats"] = {
                {"raw_corpus_char_count", stats_.raw_corpus_char_count},
                {"final_token_count_in_corpus", stats_.initial_token_count_in_corpus},
                {"initial_vocab_size", stats_.initial_vocab_size},
                {"final_vocab_size", stats_.final_vocab_size},
                {"total_merges", stats_.total_merges},
                {"training_time_ms", stats_.training_time.count()}
            };

            std::string json_model_path = fs::path(output_dir) / "bpe_model_lunaris.json";
            std::ofstream json_out_file(json_model_path);
            if (json_out_file.is_open()) {
                json_out_file << std::setw(2) << output_json << std::endl;
                consoleLog("BPE model (JSON) saved to: " + json_model_path);
            } else {
                consoleErr("Failed to save BPE model JSON to: " + json_model_path);
            }

            // Save vocabulary and merges also as plain text for easier inspection
            std::string vocab_text_path = fs::path(output_dir) / "vocabulary_lunaris.txt";
            std::ofstream vocab_out_file(vocab_text_path);
            if (vocab_out_file.is_open()) {
                for (const auto& token : vocab_list) { // vocab_list is already sorted
                    vocab_out_file << escapeTokenForDisplay(token) << "\n";
                }
                consoleLog("Vocabulary (TXT) saved to: " + vocab_text_path);
            } else {
                consoleErr("Failed to save vocabulary TXT to: " + vocab_text_path);
            }

            std::string merges_text_path = fs::path(output_dir) / "merges_lunaris.txt";
            std::ofstream merges_out_file(merges_text_path);
            if (merges_out_file.is_open()) {
                for (const auto& merge_pair : learned_merges_) {
                    merges_out_file << escapeTokenForDisplay(merge_pair.first) << " " << escapeTokenForDisplay(merge_pair.second) << "\n";
                }
                consoleLog("Merges (TXT) saved to: " + merges_text_path);
            } else {
                consoleErr("Failed to save merges TXT to: " + merges_text_path);
            }
        }
};


// CLI Argument Parsing Structure
struct CliArgs {
    std::string corpus_path;
    std::string output_dir = "./bpe_lunaris_model";
    size_t vocab_size = 32000;
    bool byte_level_mode = true; // Renamed to avoid conflict with BpeTrainer member
    bool verbose = false;
};

// CLI Help Function
void printHelp(const char* app_name) {
    std::cout << "Lunaris BPE Trainer - Byte Pair Encoding Tokenizer Trainer\n\n";
    std::cout << "Usage: " << app_name << " --corpus <filepath> [options]\n\n";
    std::cout << "Required Arguments:\n";
    std::cout << "  --corpus FILE      Path to the input corpus text file.\n\n";
    std::cout << "Optional Arguments:\n";
    std::cout << "  --output DIR       Output directory for model files (vocabulary, merges, JSON).\n"
    << "                     (Default: ./bpe_lunaris_model)\n";
    std::cout << "  --vocab-size N     Target vocabulary size.\n"
    << "                     (Default: 32000)\n";
    std::cout << "  --mode LEVEL       Tokenization mode: 'byte' or 'word'.\n"
    << "                     'byte': Initial tokens are individual bytes (UTF-8 safe hex for non-printable/non-ASCII).\n"
    << "                     'word': Initial tokens are space-separated words.\n"
    << "                     (Default: byte)\n";
    std::cout << "  --verbose          Enable verbose logging output during training.\n";
    std::cout << "  -h, --help         Display this help message and exit.\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << app_name << " --corpus my_data.txt --vocab-size 20000 --output ./my_bpe_model --verbose\n";
}

// CLI Argument Parsing Function
bool parseCliArguments(int argc, char* argv[], CliArgs& cli_args) {
    if (argc == 1) {
        printHelp(argv[0]);
        return false;
    }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            return false;
        } else if (arg == "--corpus" && i + 1 < argc) {
            cli_args.corpus_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            cli_args.output_dir = argv[++i];
        } else if (arg == "--vocab-size" && i + 1 < argc) {
            try {
                unsigned long val = std::stoul(argv[++i]); // Use unsigned long for stoul
                if (val == 0) { throw std::invalid_argument("Vocabulary size must be greater than zero.");}
                cli_args.vocab_size = static_cast<size_t>(val);
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Invalid vocabulary size: " << argv[i-1] << " (Value: " << argv[i] << "). Error: " << e.what() << "\n";
                return false;
            }
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode_str = argv[++i];
            std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower); // Convert to lowercase
            if (mode_str == "byte") {
                cli_args.byte_level_mode = true;
            } else if (mode_str == "word") {
                cli_args.byte_level_mode = false;
            } else {
                std::cerr << "[ERROR] Invalid mode: '" << mode_str << "'. Choose 'byte' or 'word'.\n";
                return false;
            }
        } else if (arg == "--verbose") {
            cli_args.verbose = true;
        } else {
            std::cerr << "[ERROR] Unknown argument or missing value: " << arg << "\n";
            printHelp(argv[0]);
            return false;
        }
    }

    if (cli_args.corpus_path.empty()) {
        std::cerr << "[ERROR] Corpus path (--corpus) is required.\n";
        printHelp(argv[0]);
        return false;
    }
    return true;
}


// Main function
int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    CliArgs cli_args;
    if (!parseCliArguments(argc, argv, cli_args)) {
        return 1;
    }

    // Pass the byte_level_mode from CLI to the BpeTrainer constructor
    BpeTrainer trainer(cli_args.verbose, cli_args.byte_level_mode);

    // The train method no longer needs byte_level as it's part of the trainer's state
    bool success = trainer.train(cli_args.corpus_path,
                                 cli_args.output_dir,
                                 cli_args.vocab_size);

    if(success && cli_args.verbose) { // Only print success if verbose
        std::cout << "[SUCCESS] BPE training completed successfully.\n";
    } else if (!success) { // Always print failure
        std::cout << "[FAILURE] BPE training encountered an error or produced no results.\n";
    }

    return success ? 0 : 1;
}
