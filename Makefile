# Makefile for Lunaris Codex C++ Utilities

# Compiler
CXX := g++

# Common C++ Standard
CPP_STD := -std=c++17

# Common Warning Flags
WARNING_FLAGS := -Wall -Wextra -pedantic

# Release Flags (default)
OPTIMIZATION_FLAGS := -O2
DEFINE_FLAGS := -DNDEBUG 
CXXFLAGS_RELEASE := $(CPP_STD) $(WARNING_FLAGS) $(OPTIMIZATION_FLAGS) $(DEFINE_FLAGS)

# Debug Flags
CXXFLAGS_DEBUG := $(CPP_STD) $(WARNING_FLAGS) -g -O0 
CXXFLAGS_MODE ?= RELEASE

ifeq ($(CXXFLAGS_MODE),DEBUG)
    CURRENT_CXXFLAGS := $(CXXFLAGS_DEBUG)
    BUILD_TYPE_MSG := "Debug build"
else
    CURRENT_CXXFLAGS := $(CXXFLAGS_RELEASE)
    BUILD_TYPE_MSG := "Release build"
endif

# --- Tool Definitions ---
TC_DIR := text_cleaner
TC_SRC := $(TC_DIR)/lunaris_text_cleaner.cpp
TC_TARGET := $(TC_DIR)/lunaris_text_cleaner

DA_DIR := data_analyzer
DA_SRC := $(DA_DIR)/lunaris_data_analyzer.cpp
DA_TARGET := $(DA_DIR)/lunaris_data_analyzer

# BPE Processor specific definitions
BPE_PROC_DIR := bpe_trainer # The directory name can remain bpe_trainer
BPE_PROC_SRC_FILENAME := bpe_processor.cpp # Actual name of your .cpp file
BPE_PROC_TARGET_NAME := bpe_processor      # Desired name of the executable
BPE_PROC_SRC := $(BPE_PROC_DIR)/$(BPE_PROC_SRC_FILENAME)
BPE_PROC_TARGET := $(BPE_PROC_DIR)/$(BPE_PROC_TARGET_NAME)
BPE_PROC_LIBS := -lstdc++fs

# Default target: build all standard utilities
all: $(TC_TARGET) $(DA_TARGET) $(BPE_PROC_TARGET)
	@echo "----------------------------------------------------"
	@echo "All standard utilities built: $(BUILD_TYPE_MSG)."
	@echo "Executables created:"
	@echo "  $(TC_TARGET)"
	@echo "  $(DA_TARGET)"
	@echo "  $(BPE_PROC_TARGET)"
	@echo "To build with debug flags, run: make CXXFLAGS_MODE=DEBUG"
	@echo "----------------------------------------------------"

# --- Build Rules ---
$(TC_TARGET): $(TC_SRC) Makefile
	@echo "Building Text Cleaner: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(TC_SRC) -o $(TC_TARGET)
	@echo "Text Cleaner built as $(TC_TARGET)"

$(DA_TARGET): $(DA_SRC) Makefile
	@echo "Building Data Analyzer: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(DA_SRC) -o $(DA_TARGET)
	@echo "Data Analyzer built as $(DA_TARGET)"

$(BPE_PROC_TARGET): $(BPE_PROC_SRC) Makefile
	@echo "Building BPE Processor: $(BUILD_TYPE_MSG)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(BPE_PROC_SRC) -o $(BPE_PROC_TARGET) $(BPE_PROC_LIBS)
	@echo "BPE Processor built as $(BPE_PROC_TARGET)"

# --- Convenience Targets (explicitly define them) ---
text_cleaner: $(TC_TARGET)
data_analyzer: $(DA_TARGET)
bpe_processor: $(BPE_PROC_TARGET)

debug_text_cleaner:
	$(MAKE) text_cleaner CXXFLAGS_MODE=DEBUG
debug_data_analyzer:
	$(MAKE) data_analyzer CXXFLAGS_MODE=DEBUG
debug_bpe_processor:
	$(MAKE) bpe_processor CXXFLAGS_MODE=DEBUG
debug_all:
	$(MAKE) all CXXFLAGS_MODE=DEBUG

# --- Clean Target ---
clean:
	@echo "Cleaning C++ utilities..."
	rm -f $(TC_TARGET) $(DA_TARGET) $(BPE_PROC_TARGET)
	@echo "Cleanup complete."

# Phony targets
.PHONY: all text_cleaner data_analyzer bpe_processor clean \
        debug_text_cleaner debug_data_analyzer debug_bpe_processor debug_all
