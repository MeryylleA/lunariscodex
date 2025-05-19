# Makefile for Lunaris Codex C++ Utilities

# Compiler
CXX := g++

# Common C++ Standard
CPP_STD := -std=c++17

# Common Warning Flags
WARNING_FLAGS := -Wall -Wextra -pedantic

# Release Flags (default)
OPTIMIZATION_FLAGS := -O2
DEFINE_FLAGS := -DNDEBUG # Define NDEBUG to disable asserts, etc. in release
CXXFLAGS_RELEASE := $(CPP_STD) $(WARNING_FLAGS) $(OPTIMIZATION_FLAGS) $(DEFINE_FLAGS)

# Debug Flags
CXXFLAGS_DEBUG := $(CPP_STD) $(WARNING_FLAGS) -g -O0 # -g for debug symbols, -O0 to disable optimizations

# Select build mode flags (Can be overridden from command line: make CXXFLAGS_MODE=DEBUG)
CXXFLAGS_MODE ?= RELEASE # Default to RELEASE if not set

ifeq ($(CXXFLAGS_MODE),DEBUG)
    CURRENT_CXXFLAGS := $(CXXFLAGS_DEBUG)
    BUILD_TYPE_MSG := "Debug build"
else
    CURRENT_CXXFLAGS := $(CXXFLAGS_RELEASE)
    BUILD_TYPE_MSG := "Release build - Default"
endif

# --- Text Cleaner ---
TC_DIR := text_cleaner
TC_SRC := $(TC_DIR)/lunaris_text_cleaner.cpp
TC_TARGET_NAME := lunaris_text_cleaner
TC_TARGET := $(TC_DIR)/$(TC_TARGET_NAME)
TC_CI_TARGET_NAME := lunaris_text_cleaner_ci_executable
TC_CI_TARGET := $(TC_DIR)/$(TC_CI_TARGET_NAME)

# --- Data Analyzer ---
DA_DIR := data_analyzer
DA_SRC := $(DA_DIR)/lunaris_data_analyzer.cpp
DA_TARGET_NAME := lunaris_data_analyzer
DA_TARGET := $(DA_DIR)/$(DA_TARGET_NAME)
DA_CI_TARGET_NAME := lda_ci_executable
DA_CI_TARGET := $(DA_DIR)/$(DA_CI_TARGET_NAME)

# --- BPE Trainer ---
BPE_DIR := bpe_trainer
BPE_SRC := $(BPE_DIR)/bpe_trainer.cpp
BPE_TARGET_NAME := bpe_trainer
BPE_TARGET := $(BPE_DIR)/$(BPE_TARGET_NAME)
BPE_CI_TARGET_NAME := bpe_trainer_ci_executable
BPE_CI_TARGET := $(BPE_DIR)/$(BPE_CI_TARGET_NAME)


# Default target: build all standard utilities
all: text_cleaner data_analyzer bpe_trainer
	@echo "----------------------------------------------------"
	@echo "All standard utilities built: $(BUILD_TYPE_MSG)."
	@echo "Executables created:"
	@echo "  $(TC_TARGET)"
	@echo "  $(DA_TARGET)"
	@echo "  $(BPE_TARGET)"
	@echo "To build with debug flags, run: make CXXFLAGS_MODE=DEBUG"
	@echo "----------------------------------------------------"

# Target to build all CI-named executables
all_ci: text_cleaner_ci data_analyzer_ci bpe_trainer_ci
	@echo "----------------------------------------------------"
	@echo "All CI executables built: $(BUILD_TYPE_MSG) (CI typically uses Release)."
	@echo "Executables created for CI:"
	@echo "  $(TC_CI_TARGET)"
	@echo "  $(DA_CI_TARGET)"
	@echo "  $(BPE_CI_TARGET)"
	@echo "----------------------------------------------------"

# --- Text Cleaner Targets ---
text_cleaner: $(TC_TARGET)
$(TC_TARGET): $(TC_SRC) Makefile
	@echo "Building Text Cleaner: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(TC_SRC) -o $(TC_TARGET)
	@echo "Text Cleaner built as $(TC_TARGET)"

text_cleaner_ci: $(TC_CI_TARGET)
$(TC_CI_TARGET): $(TC_SRC) Makefile
	@echo "Building Text Cleaner for CI: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(TC_SRC) -o $(TC_CI_TARGET)
	@echo "Text Cleaner for CI built as $(TC_CI_TARGET)"

# --- Data Analyzer Targets ---
data_analyzer: $(DA_TARGET)
$(DA_TARGET): $(DA_SRC) Makefile
	@echo "Building Data Analyzer: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(DA_SRC) -o $(DA_TARGET)
	@echo "Data Analyzer built as $(DA_TARGET)"

data_analyzer_ci: $(DA_CI_TARGET)
$(DA_CI_TARGET): $(DA_SRC) Makefile
	@echo "Building Data Analyzer for CI: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(DA_SRC) -o $(DA_CI_TARGET)
	@echo "Data Analyzer for CI built as $(DA_CI_TARGET)"

# --- BPE Trainer Targets ---
bpe_trainer: $(BPE_TARGET)
$(BPE_TARGET): $(BPE_SRC) Makefile # Assumes nlohmann/json.hpp is in system include paths
	@echo "Building BPE Trainer: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(BPE_SRC) -o $(BPE_TARGET) -lstdc++fs # Added -lstdc++fs for filesystem
	@echo "BPE Trainer built as $(BPE_TARGET)"

bpe_trainer_ci: $(BPE_CI_TARGET)
$(BPE_CI_TARGET): $(BPE_SRC) Makefile # Assumes nlohmann/json.hpp is in system include paths
	@echo "Building BPE Trainer for CI: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(BPE_SRC) -o $(BPE_CI_TARGET) -lstdc++fs # Added -lstdc++fs for filesystem
	@echo "BPE Trainer for CI built as $(BPE_CI_TARGET)"

# --- Debug Build Targets (Convenience) ---
debug_text_cleaner: ; $(MAKE) text_cleaner CXXFLAGS_MODE=DEBUG
debug_data_analyzer: ; $(MAKE) data_analyzer CXXFLAGS_MODE=DEBUG
debug_bpe_trainer: ; $(MAKE) bpe_trainer CXXFLAGS_MODE=DEBUG
debug_all: ; $(MAKE) all CXXFLAGS_MODE=DEBUG

# --- Clean Target ---
# Removes all built executables (standard and CI versions)
clean:
	@echo "Cleaning C++ utilities..."
	rm -f $(TC_TARGET) $(TC_CI_TARGET) \
	      $(DA_TARGET) $(DA_CI_TARGET) \
	      $(BPE_TARGET) $(BPE_CI_TARGET)
	@echo "Cleanup complete."

# Phony targets (targets that are not actual files)
.PHONY: all text_cleaner data_analyzer bpe_trainer \
        text_cleaner_ci data_analyzer_ci bpe_trainer_ci \
        all_ci clean \
        debug_text_cleaner debug_data_analyzer debug_bpe_trainer debug_all
