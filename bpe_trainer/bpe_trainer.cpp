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

// BPE Processor Class
class BpeProcessor {
public:
    BpeProcessor(bool verbose = false)
    : verbose_(verbose), byte_level_processing_mode_(true), model_loaded_(false) {}

    // --- Training Functionality ---
    bool train(const std::string& corpus_path,
               const std::string& output_dir_or_prefix,
               size_t target_vocab_size_param,
               bool byte_level_training_mode) {

        auto overall_start_time = std::chrono::high_resolution_clock::now();
        stats_ = BpeStats{};
        training_vocab_.clear();
        learned_merges_for_saving_.clear();
        token_to_id_.clear();
        id_to_token_.clear();
        loaded_merges_for_tokenizing_.clear();
        model_loaded_ = false;

        this->byte_level_processing_mode_ = byte_level_training_mode;

        if (verbose_) {
            consoleLog("Starting BPE training process...");
            consoleLog("Mode: " + std::string(this->byte_level_processing_mode_ ? "Byte-level" : "Word-level"));
            consoleLog("Loading corpus from file: " + corpus_path);
        }

        std::string actual_output_dir;
        std::string file_prefix;
        if (output_dir_or_prefix.empty() || output_dir_or_prefix.back() == '/' || output_dir_or_prefix.back() == '\\') {
            actual_output_dir = output_dir_or_prefix;
            if (actual_output_dir.empty()) actual_output_dir = "./bpe_lunaris_model/";
            file_prefix = "";
        } else {
            fs::path p(output_dir_or_prefix);
            actual_output_dir = p.parent_path().string();
            if (actual_output_dir.empty()) actual_output_dir = ".";
            file_prefix = p.filename().string() + "_";
        }
        if (!ensureOutputDirectory(actual_output_dir)) return false;


        if (!this->loadAndPretokenizeCorpus(corpus_path)) { // CORRECTED
            consoleErr("Error: Failed to load or pretokenize the corpus.");
            return false;
        }

        this->initializeVocabularyFromPretokenizedCorpus(); // CORRECTED
        stats_.initial_vocab_size = training_vocab_.size();

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

        if (actual_target_vocab_size <= training_vocab_.size()){
            if (verbose_) {
                consoleLog("Target vocabulary size (" + std::to_string(actual_target_vocab_size) +
                ") is already met or exceeded by initial vocabulary (" + std::to_string(training_vocab_.size()) + "). No merges will be performed.");
            }
        } else {
            if (verbose_) consoleLog("Starting BPE merge iterations...");
            while (training_vocab_.size() < actual_target_vocab_size) {
                auto pair_counts = this->countAdjacentPairs(); // CORRECTED
                if (pair_counts.empty()) {
                    if (verbose_) consoleLog("No more pairs to merge. Stopping iterations.");
                    break;
                }

                auto best_pair_info = this->findMostFrequentPair(pair_counts); // CORRECTED
                if (best_pair_info.first.first.empty() || best_pair_info.second <= 1) {
                    if (verbose_) consoleLog("No more significantly frequent pairs (frequency > 1) to merge. Stopping iterations.");
                    break;
                }

                const auto& pair_to_merge = best_pair_info.first;
                std::string new_token = pair_to_merge.first + pair_to_merge.second;

                if (training_vocab_.count(new_token)) {
                    if (verbose_ && stats_.total_merges < 5 + stats_.initial_vocab_size) {
                        consoleLog("Skipping merge: new token '" + escapeTokenForDisplay(new_token) + "' already in vocab. This can happen.");
                    }
                    bool found_other_merge = false;
                    for(const auto& pc_entry : pair_counts){
                        if(pc_entry.second > 1 && !training_vocab_.count(pc_entry.first.first + pc_entry.first.second)){
                            found_other_merge = true;
                            break;
                        }
                    }
                    if(!found_other_merge && best_pair_info.second > 1){
                        if(verbose_) consoleLog("All remaining frequent pairs (freq > 1) result in existing tokens. Stopping iterations to prevent stall.");
                        break;
                    }
                }

                if (verbose_) {
                    consoleLog("Merge #" + std::to_string(stats_.total_merges + 1) +
                    ": '" + escapeTokenForDisplay(pair_to_merge.first) +
                    "' + '" + escapeTokenForDisplay(pair_to_merge.second) +
                    "' -> '" + escapeTokenForDisplay(new_token) +
                    "' (Frequency: " + std::to_string(best_pair_info.second) + ")");
                }

                this->applyMergeToCorpus(pair_to_merge.first, pair_to_merge.second, new_token); // CORRECTED
                learned_merges_for_saving_.push_back(pair_to_merge);
                stats_.total_merges++;
                training_vocab_.insert(new_token);

                if (verbose_ && (stats_.total_merges % 100 == 0 || training_vocab_.size() % std::max((size_t)1, actual_target_vocab_size/100) == 0) ) {
                    consoleLog("Progress: " + std::to_string(training_vocab_.size()) + "/" + std::to_string(actual_target_vocab_size) +
                    " vocab tokens (" + formatDouble(training_vocab_.size() * 100.0 / actual_target_vocab_size, 2) + "%)");
                }
            }
        }

        stats_.final_vocab_size = training_vocab_.size();
        auto overall_end_time = std::chrono::high_resolution_clock::now();
        stats_.training_time = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time);

        if (verbose_) this->printStats(); // CORRECTED
        saveTrainingResults(actual_output_dir, file_prefix);
        return true;
               }

               bool load_model(const std::string& model_path_or_dir) {
                   token_to_id_.clear();
                   id_to_token_.clear();
                   loaded_merges_for_tokenizing_.clear();
                   model_loaded_ = false;

                   fs::path model_fs_path(model_path_or_dir);
                   std::string json_model_file_path;

                   if (fs::is_directory(model_fs_path)) {
                       json_model_file_path = (model_fs_path / "bpe_model_lunaris.json").string();
                   } else if (model_fs_path.filename().string().find("bpe_model_lunaris.json") != std::string::npos) {
                       json_model_file_path = model_path_or_dir;
                   } else {
                       json_model_file_path = model_path_or_dir + "_bpe_model_lunaris.json";
                       if (!fs::exists(json_model_file_path)) {
                           json_model_file_path = (fs::path(model_path_or_dir).parent_path() / (fs::path(model_path_or_dir).filename().string() + "_bpe_model_lunaris.json")).string();
                           if (!fs::exists(json_model_file_path)) {
                               json_model_file_path = (fs::path(model_path_or_dir) / "bpe_model_lunaris.json").string();
                           }
                       }
                   }

                   if (!fs::exists(json_model_file_path)) {
                       consoleErr("Could not find BPE model JSON file at or derived from: " + model_path_or_dir + " (tried: " + json_model_file_path + ")");
                       return false;
                   }
                   consoleLog("Loading BPE model from: " + json_model_file_path);

                   std::ifstream json_file(json_model_file_path);
                   if (!json_file.is_open()) {
                       consoleErr("Could not open BPE model JSON file: " + json_model_file_path);
                       return false;
                   }

                   json model_data;
                   try {
                       json_file >> model_data;
                   } catch (json::parse_error& e) {
                       consoleErr("Error parsing BPE model JSON: " + std::string(e.what()));
                       return false;
                   }

                   if (model_data.contains("bpe_config") && model_data["bpe_config"].contains("mode")) {
                       std::string mode_str = model_data["bpe_config"]["mode"];
                       byte_level_processing_mode_ = (mode_str == "byte");
                       // CORRECTED string concatenation for log
                       consoleLog(std::string("  Model BPE mode loaded: ") + (byte_level_processing_mode_ ? "byte" : "word"));
                   } else {
                       consoleErr("Error: 'bpe_config.mode' not found in model JSON. Cannot determine tokenization mode.");
                       return false;
                   }

                   if (model_data.contains("vocabulary_map") && model_data["vocabulary_map"].is_object()) {
                       for (auto const& [token_str, id_val] : model_data["vocabulary_map"].items()) {
                           if (!id_val.is_number_integer()) {
                               consoleErr("  Error: Vocabulary ID for token '" + token_str + "' is not an integer.");
                               return false;
                           }
                           int token_id = id_val.get<int>();
                           token_to_id_[token_str] = token_id;
                           id_to_token_[token_id] = token_str;
                       }
                       consoleLog("  Vocabulary loaded. Size: " + std::to_string(token_to_id_.size()));
                   } else {
                       consoleErr("Error: 'vocabulary_map' (object) not found or invalid in model JSON.");
                       return false;
                   }

                   if (model_data.contains("merges") && model_data["merges"].is_array()) {
                       for (const auto& merge_item : model_data["merges"]) {
                           if (merge_item.is_array() && merge_item.size() == 2 &&
                               merge_item[0].is_string() && merge_item[1].is_string()) {
                               loaded_merges_for_tokenizing_.push_back({merge_item[0].get<std::string>(), merge_item[1].get<std::string>()});
                               } else {
                                   consoleErr("  Error: Invalid merge item format in model JSON.");
                                   return false;
                               }
                       }
                       consoleLog("  Merges loaded. Count: " + std::to_string(loaded_merges_for_tokenizing_.size()));
                   } else {
                       consoleErr("Error: 'merges' (array) not found or invalid in model JSON.");
                       return false;
                   }

                   model_loaded_ = true;
                   consoleLog("BPE model loaded successfully.");
                   return true;
               }

private:
    bool verbose_;
    std::vector<std::vector<std::string>> pretokenized_corpus_;
    std::unordered_set<std::string> training_vocab_;
    std::vector<std::pair<std::string, std::string>> learned_merges_for_saving_;
    BpeStats stats_;
    bool byte_level_processing_mode_;

    std::map<std::string, int> token_to_id_;
    std::map<int, std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> loaded_merges_for_tokenizing_;
    bool model_loaded_;

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
            if (std::isprint(static_cast<unsigned char>(c)) && c != '\\') {
                escaped_token += c;
            } else if (c == '\\'){
                escaped_token += "\\\\";
            }
            else {
                char hex_repr[5];
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
            current_line_tokens.reserve(line.length());
            if (this->byte_level_processing_mode_) {
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
                stats_.initial_token_count_in_corpus += pretokenized_corpus_.back().size();
            }
            if(verbose_ && line_number % 50000 == 0) consoleLog("Processed " + std::to_string(line_number) + " lines... Current initial tokens: " + std::to_string(stats_.initial_token_count_in_corpus));
        }
        file.close();
        if (pretokenized_corpus_.empty() && stats_.raw_corpus_char_count > 0) {
            consoleLog("Warning: Corpus was read but resulted in no pre-tokenized sequences. Check input or pre-tokenization logic.");
        }
        return !pretokenized_corpus_.empty() || stats_.raw_corpus_char_count == 0;
    }

    void initializeVocabularyFromPretokenizedCorpus() {
        training_vocab_.clear();
        if (this->byte_level_processing_mode_) {
            for (int i = 0; i < 128; ++i) {
                if (i >= 32 && i <= 126) {
                    training_vocab_.insert(std::string(1, static_cast<char>(i)));
                } else {
                    char hex_buf[5];
                    snprintf(hex_buf, sizeof(hex_buf), "\\x%02X", static_cast<unsigned char>(i));
                    training_vocab_.insert(hex_buf);
                }
            }
        }
        for (const auto& sentence_tokens : pretokenized_corpus_) {
            for (const auto& token : sentence_tokens) {
                training_vocab_.insert(token);
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
            stats_.initial_token_count_in_corpus = new_total_token_count;
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

        void saveTrainingResults(const std::string& actual_output_dir, const std::string& file_prefix) {
            json output_json;

            std::vector<std::string> sorted_vocab_list(training_vocab_.begin(), training_vocab_.end());
            std::sort(sorted_vocab_list.begin(), sorted_vocab_list.end());

            json vocab_map_json = json::object();
            token_to_id_.clear();
            id_to_token_.clear();
            for (int i = 0; i < static_cast<int>(sorted_vocab_list.size()); ++i) { // Cast to int for loop
                const std::string& token = sorted_vocab_list[i];
                vocab_map_json[token] = i;
                token_to_id_[token] = i;
                id_to_token_[i] = token;
            }
            output_json["vocabulary_map"] = vocab_map_json;
            output_json["vocabulary_size"] = sorted_vocab_list.size();

            json merges_list_json = json::array();
            for (const auto& merge_pair : learned_merges_for_saving_) {
                merges_list_json.push_back({merge_pair.first, merge_pair.second});
            }
            output_json["merges"] = merges_list_json;

            output_json["bpe_config"] = {
                {"mode", this->byte_level_processing_mode_ ? "byte" : "word"}
            };

            output_json["stats"] = {
                {"raw_corpus_char_count", stats_.raw_corpus_char_count},
                {"final_token_count_in_corpus", stats_.initial_token_count_in_corpus},
                {"initial_vocab_size", stats_.initial_vocab_size},
                {"final_vocab_size_from_training_set", stats_.final_vocab_size},
                {"actual_merges_performed", stats_.total_merges},
                {"training_time_ms", stats_.training_time.count()}
            };

            std::string json_model_filename = file_prefix + "bpe_model_lunaris.json";
            std::string json_model_path = (fs::path(actual_output_dir) / json_model_filename).string();
            std::ofstream json_out_file(json_model_path);
            if (json_out_file.is_open()) {
                json_out_file << std::setw(2) << output_json << std::endl;
                consoleLog("BPE model (JSON) saved to: " + json_model_path);
            } else {
                consoleErr("Failed to save BPE model JSON to: " + json_model_path);
            }

            std::string vocab_text_filename = file_prefix + "vocabulary_lunaris.txt";
            std::string vocab_text_path = (fs::path(actual_output_dir) / vocab_text_filename).string();
            std::ofstream vocab_out_file(vocab_text_path);
            if (vocab_out_file.is_open()) {
                for (int i = 0; i < static_cast<int>(sorted_vocab_list.size()); ++i) { // Cast to int for loop
                    vocab_out_file << escapeTokenForDisplay(sorted_vocab_list[i]) << "\n";
                }
                consoleLog("Vocabulary (TXT) saved to: " + vocab_text_path);
            } else {
                consoleErr("Failed to save vocabulary TXT to: " + vocab_text_path);
            }

            std::string merges_text_filename = file_prefix + "merges_lunaris.txt";
            std::string merges_text_path = (fs::path(actual_output_dir) / merges_text_filename).string();
            std::ofstream merges_out_file(merges_text_path);
            if (merges_out_file.is_open()) {
                for (const auto& merge_pair : learned_merges_for_saving_) {
                    merges_out_file << escapeTokenForDisplay(merge_pair.first) << " " << escapeTokenForDisplay(merge_pair.second) << "\n";
                }
                consoleLog("Merges (TXT) saved to: " + merges_text_path);
            } else {
                consoleErr("Failed to save merges TXT to: " + merges_text_path);
            }
        }
};

struct CliArgs {
    std::string action = "train";
    std::string corpus_path_or_input_text;
    std::string input_file_path;
    std::string model_path;
    std::string output_path_or_dir = "./bpe_lunaris_model";
    size_t vocab_size = 32000;
    bool byte_level_mode = true;
    bool verbose = false;
};

void printHelp(const char* app_name) {
    std::cout << "Lunaris BPE Processor - Train, Tokenize, Detokenize\n\n";
    std::cout << "Usage: " << app_name << " --action <train|tokenize|detokenize> [options...]\n\n";
    std::cout << "Actions:\n";
    std::cout << "  train: Trains a new BPE model.\n";
    std::cout << "    Required: --corpus <filepath>\n";
    std::cout << "    Optional: --output <dir_or_prefix>, --vocab-size <N>, --mode <byte|word>, --verbose\n";
    std::cout << "  tokenize: Tokenizes input text using a trained model.\n";
    std::cout << "    Required: --model_path <path_to_model_dir_or_prefix>, --input_text \"<text>\" (or --input_file <filepath>)\n";
    std::cout << "    Optional: --verbose\n";
    std::cout << "\nCommon Options:\n";
    std::cout << "  -h, --help         Display this help message and exit.\n";
}

bool parseCliArguments(int argc, char* argv[], CliArgs& cli_args) {
    if (argc == 1) { printHelp(argv[0]); return false; }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { printHelp(argv[0]); return false; }
        else if (arg == "--action" && i + 1 < argc) { cli_args.action = argv[++i]; }
        else if (arg == "--corpus" && i + 1 < argc) { cli_args.corpus_path_or_input_text = argv[++i]; }
        else if (arg == "--input_text" && i + 1 < argc) { cli_args.corpus_path_or_input_text = argv[++i];}
        else if (arg == "--input_file" && i + 1 < argc) { cli_args.input_file_path = argv[++i];}
        else if (arg == "--model_path" && i + 1 < argc) { cli_args.model_path = argv[++i]; }
        else if (arg == "--output" && i + 1 < argc) { cli_args.output_path_or_dir = argv[++i]; }
        else if (arg == "--vocab-size" && i + 1 < argc) { cli_args.vocab_size = std::stoul(argv[++i]); }
        else if (arg == "--mode" && i + 1 < argc) { cli_args.byte_level_mode = (std::string(argv[++i]) == "byte");}
        else if (arg == "--verbose") { cli_args.verbose = true; }
        else { // Basic unknown argument handling
            std::cerr << "[ERROR] Unknown or incomplete argument: " << arg << std::endl;
            printHelp(argv[0]);
            return false;
        }
    }
    if (cli_args.action == "train" && cli_args.corpus_path_or_input_text.empty()) {
        std::cerr << "[ERROR] For action 'train', --corpus is required.\n"; return false;
    }
    if (cli_args.action == "tokenize" && cli_args.model_path.empty()) {
        std::cerr << "[ERROR] For action 'tokenize', --model_path is required.\n"; return false;
    }
    if (cli_args.action == "tokenize" && cli_args.corpus_path_or_input_text.empty() && cli_args.input_file_path.empty()) {
        std::cerr << "[ERROR] For action 'tokenize', either --input_text or --input_file is required.\n"; return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    CliArgs cli_args;
    if (!parseCliArguments(argc, argv, cli_args)) {
        return 1;
    }

    BpeProcessor processor(cli_args.verbose);

    if (cli_args.action == "train") {
        bool success = processor.train(cli_args.corpus_path_or_input_text,
                                       cli_args.output_path_or_dir,
                                       cli_args.vocab_size,
                                       cli_args.byte_level_mode);
        if(success && cli_args.verbose) {
            std::cout << "[SUCCESS] BPE training completed successfully.\n";
        } else if (!success && !cli_args.verbose) { // Print failure even if not verbose
            std::cout << "[FAILURE] BPE training encountered an error or produced no results.\n";
        } else if (!success && cli_args.verbose){
            // verbose already printed errors
        }
        return success ? 0 : 1;
    } else if (cli_args.action == "tokenize") {
        if (!processor.load_model(cli_args.model_path)) {
            std::cerr << "[FAILURE] Could not load BPE model for tokenization.\n";
            return 1;
        }
        std::string text_to_tokenize;
        if (!cli_args.input_file_path.empty()) {
            std::ifstream ifs(cli_args.input_file_path);
            if (!ifs) { std::cerr << "Error opening input file: " << cli_args.input_file_path << std::endl; return 1;}
            std::stringstream buffer;
            buffer << ifs.rdbuf();
            text_to_tokenize = buffer.str();
        } else {
            text_to_tokenize = cli_args.corpus_path_or_input_text;
        }

        std::cout << "[INFO] Tokenize action called. Input (first 50 chars): '" << text_to_tokenize.substr(0, 50) << (text_to_tokenize.length() > 50 ? "..." : "") << "'. Actual tokenization not yet implemented in this step.\n";
        return 0;
    }
    else {
        std::cerr << "[ERROR] Unknown action: " << cli_args.action << std::endl;
        printHelp(argv[0]);
        return 1;
    }
}
