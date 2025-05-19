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
# CI will typically use the default (RELEASE)
CXXFLAGS_MODE ?= RELEASE

ifeq ($(CXXFLAGS_MODE),DEBUG)
    CURRENT_CXXFLAGS := $(CXXFLAGS_DEBUG)
    BUILD_TYPE_MSG := "Debug build"
else
    CURRENT_CXXFLAGS := $(CXXFLAGS_RELEASE)
    BUILD_TYPE_MSG := "Release build"
endif

# --- Tool Directories and Source Files ---
TC_DIR := text_cleaner
TC_SRC := $(TC_DIR)/lunaris_text_cleaner.cpp
TC_TARGET := $(TC_DIR)/lunaris_text_cleaner

DA_DIR := data_analyzer
DA_SRC := $(DA_DIR)/lunaris_data_analyzer.cpp
DA_TARGET := $(DA_DIR)/lunaris_data_analyzer

BPE_DIR := bpe_trainer
BPE_SRC := $(BPE_DIR)/bpe_trainer.cpp
BPE_TARGET := $(BPE_DIR)/bpe_trainer
# Ensure nlohmann/json.hpp is available. If it's vendored (e.g., in a 'third_party' dir),
# you might need to add -I./third_party/nlohmann/include or similar.
# For now, assuming it's in system paths or handled by user.
BPE_LIBS := -lstdc++fs # For std::filesystem

# Default target: build all standard utilities
all: $(TC_TARGET) $(DA_TARGET) $(BPE_TARGET)
	@echo "----------------------------------------------------"
	@echo "All standard utilities built: $(BUILD_TYPE_MSG)."
	@echo "Executables created:"
	@echo "  $(TC_TARGET)"
	@echo "  $(DA_TARGET)"
	@echo "  $(BPE_TARGET)"
	@echo "To build with debug flags, run: make CXXFLAGS_MODE=DEBUG"
	@echo "----------------------------------------------------"

# --- Text Cleaner Target ---
$(TC_TARGET): $(TC_SRC) Makefile
	@echo "Building Text Cleaner: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(TC_SRC) -o $(TC_TARGET)
	@echo "Text Cleaner built as $(TC_TARGET)"

# --- Data Analyzer Target ---
$(DA_TARGET): $(DA_SRC) Makefile
	@echo "Building Data Analyzer: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(DA_SRC) -o $(DA_TARGET)
	@echo "Data Analyzer built as $(DA_TARGET)"

# --- BPE Trainer Target ---
$(BPE_TARGET): $(BPE_SRC) Makefile
	@echo "Building BPE Trainer: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(BPE_SRC) -o $(BPE_TARGET) $(BPE_LIBS)
	@echo "BPE Trainer built as $(BPE_TARGET)"

# --- Convenience Targets ---
# Expose individual tool builds directly
text_cleaner: $(TC_TARGET)
data_analyzer: $(DA_TARGET)
bpe_trainer: $(BPE_TARGET)

debug_text_cleaner: ; $(MAKE) text_cleaner CXXFLAGS_MODE=DEBUG
debug_data_analyzer: ; $(MAKE) data_analyzer CXXFLAGS_MODE=DEBUG
debug_bpe_trainer: ; $(MAKE) bpe_trainer CXXFLAGS_MODE=DEBUG
debug_all: ; $(MAKE) all CXXFLAGS_MODE=DEBUG

# --- Clean Target ---
# Removes all built executables
clean:
	@echo "Cleaning C++ utilities..."
	rm -f $(TC_TARGET) $(DA_TARGET) $(BPE_TARGET)
	@echo "Cleanup complete."

# Phony targets (targets that are not actual files)
.PHONY: all text_cleaner data_analyzer bpe_trainer clean \
        debug_text_cleaner debug_data_analyzer debug_bpe_trainer debug_all
