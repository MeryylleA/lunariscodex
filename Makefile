# Makefile for Lunaris Codex C++ Utilities (Linux/Ubuntu Focus)

CXX := g++
CPP_STD := -std=c++17
WARNING_FLAGS := -Wall -Wextra -pedantic

# No Ubuntu, nlohmann-json3-dev instala os headers em locais padrão.
# Não é necessário -I./third_party se estiver usando a versão do sistema.
INCLUDE_PATHS := # Pode ser deixado vazio ou removido se não houver outros includes de terceiros

OPTIMIZATION_FLAGS_RELEASE := -O2 -DNDEBUG
OPTIMIZATION_FLAGS_DEBUG := -g -O0

CXXFLAGS_MODE ?= RELEASE
ifeq ($(CXXFLAGS_MODE),DEBUG)
    CURRENT_CXXFLAGS := $(CPP_STD) $(WARNING_FLAGS) $(OPTIMIZATION_FLAGS_DEBUG) $(INCLUDE_PATHS)
    BUILD_TYPE_MSG := "Debug build"
else
    CURRENT_CXXFLAGS := $(CPP_STD) $(WARNING_FLAGS) $(OPTIMIZATION_FLAGS_RELEASE) $(INCLUDE_PATHS)
    BUILD_TYPE_MSG := "Release build"
endif

# EXE_SUFFIX não é necessário no Linux
EXE_SUFFIX :=

# --- Tool Definitions ---
TEXT_CLEANER_SRC_FILE := lunaris_text_cleaner.cpp
TEXT_CLEANER_DIR := text_cleaner
TEXT_CLEANER_SRC := $(TEXT_CLEANER_DIR)/$(TEXT_CLEANER_SRC_FILE)
TEXT_CLEANER_TARGET := $(TEXT_CLEANER_DIR)/lunaris_text_cleaner$(EXE_SUFFIX)

DATA_ANALYZER_SRC_FILE := lunaris_data_analyzer.cpp
DATA_ANALYZER_DIR := data_analyzer
DATA_ANALYZER_SRC := $(DATA_ANALYZER_DIR)/$(DATA_ANALYZER_SRC_FILE)
DATA_ANALYZER_TARGET := $(DATA_ANALYZER_DIR)/lunaris_data_analyzer$(EXE_SUFFIX)

BPE_PROCESSOR_SRC_FILE := bpe_processor.cpp
BPE_PROCESSOR_DIR := bpe_trainer
BPE_PROCESSOR_SRC := $(BPE_PROCESSOR_DIR)/$(BPE_PROCESSOR_SRC_FILE)
BPE_PROCESSOR_TARGET := $(BPE_PROCESSOR_DIR)/bpe_processor$(EXE_SUFFIX)
# Para g++ >= 9 no Linux, -lstdc++fs geralmente funciona para <filesystem>
BPE_PROCESSOR_LIBS := -lstdc++fs


# --- Debugging ---
$(info Makefile - CURRENT_CXXFLAGS: $(CURRENT_CXXFLAGS))
$(info Makefile - TEXT_CLEANER_TARGET: $(TEXT_CLEANER_TARGET))
$(info Makefile - BPE_PROCESSOR_TARGET: $(BPE_PROCESSOR_TARGET))


# Default target
all: $(TEXT_CLEANER_TARGET) $(DATA_ANALYZER_TARGET) $(BPE_PROCESSOR_TARGET)
	@echo "----------------------------------------------------"
	@echo "All utilities built: $(BUILD_TYPE_MSG)."
	@echo "  $(TEXT_CLEANER_TARGET)"
	@echo "  $(DATA_ANALYZER_TARGET)"
	@echo "  $(BPE_PROCESSOR_TARGET)"
	@echo "----------------------------------------------------"

# --- Build Rules ---
# Não precisamos mais de $(JSON_HEADER) como pré-requisito se usamos a versão do sistema
$(TEXT_CLEANER_TARGET): $(TEXT_CLEANER_SRC) Makefile
	@echo "Building Text Cleaner (target: $@, source: $<)..."
	$(CXX) $(CURRENT_CXXFLAGS) $< -o $@

$(DATA_ANALYZER_TARGET): $(DATA_ANALYZER_SRC) Makefile
	@echo "Building Data Analyzer (target: $@, source: $<)..."
	$(CXX) $(CURRENT_CXXFLAGS) $< -o $@

$(BPE_PROCESSOR_TARGET): $(BPE_PROCESSOR_SRC) Makefile
	@echo "Building BPE Processor (target: $@, source: $<)..."
	@echo "Command: $(CXX) $(CURRENT_CXXFLAGS) $< -o $@ $(BPE_PROCESSOR_LIBS)"
	$(CXX) $(CURRENT_CXXFLAGS) $< -o $@ $(BPE_PROCESSOR_LIBS)


# --- Convenience Targets ---
text_cleaner: $(TEXT_CLEANER_TARGET)
data_analyzer: $(DATA_ANALYZER_TARGET)
bpe_processor: $(BPE_PROCESSOR_TARGET)

debug_text_cleaner: ; $(MAKE) text_cleaner CXXFLAGS_MODE=DEBUG
debug_data_analyzer: ; $(MAKE) data_analyzer CXXFLAGS_MODE=DEBUG
debug_bpe_processor: ; $(MAKE) bpe_processor CXXFLAGS_MODE=DEBUG
debug_all: ; $(MAKE) all CXXFLAGS_MODE=DEBUG

# --- Clean Target ---
clean:
	@echo "Cleaning C++ utilities..."
	rm -f $(TEXT_CLEANER_TARGET) \
	      $(DATA_ANALYZER_TARGET) \
	      $(BPE_PROCESSOR_TARGET)
	@echo "Cleanup complete."

.PHONY: all text_cleaner data_analyzer bpe_processor clean debug_text_cleaner debug_data_analyzer debug_bpe_processor debug_all
