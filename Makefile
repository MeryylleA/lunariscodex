# Makefile for Lunaris Codex C++ Utilities - Explicit Version

CXX := g++
CPP_STD := -std=c++17
WARNING_FLAGS := -Wall -Wextra -pedantic

# Adicione o caminho para os headers de terceiros
INCLUDE_PATHS := -I./third_party

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

# OS_TYPE será passado pelo CI (ex: Windows_NT, Linux, Darwin)
# Default para OS_TYPE se não for passado (útil para builds locais)
OS_TYPE ?= $(shell uname -s)

EXE_SUFFIX :=
ifeq ($(OS_TYPE),Windows_NT)
    EXE_SUFFIX := .exe
else ifeq ($(OS_TYPE),MINGW32_NT-$(shell uname -m | cut -d'-' -f1)) # Git Bash no Windows reporta MINGW...
	OS_TYPE := Windows_NT # Normaliza para Windows_NT
    EXE_SUFFIX := .exe
else ifeq ($(OS_TYPE),MINGW64_NT-$(shell uname -m | cut -d'-' -f1))
	OS_TYPE := Windows_NT # Normaliza para Windows_NT
    EXE_SUFFIX := .exe
endif

# --- Tool Definitions ---
# Verifique se estes caminhos correspondem EXATAMENTE à sua estrutura de diretórios
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
BPE_PROCESSOR_LIBS := -lstdc++fs


# --- Debugging ---
$(info Makefile - OS_TYPE (inicial/passado): $(OS_TYPE))
$(info Makefile - EXE_SUFFIX: $(EXE_SUFFIX))
$(info Makefile - CURRENT_CXXFLAGS: $(CURRENT_CXXFLAGS))
$(info Makefile - TEXT_CLEANER_SRC: $(TEXT_CLEANER_SRC))
$(info Makefile - TEXT_CLEANER_TARGET: $(TEXT_CLEANER_TARGET))
$(info Makefile - DATA_ANALYZER_SRC: $(DATA_ANALYZER_SRC))
$(info Makefile - DATA_ANALYZER_TARGET: $(DATA_ANALYZER_TARGET))
$(info Makefile - BPE_PROCESSOR_SRC: $(BPE_PROCESSOR_SRC))
$(info Makefile - BPE_PROCESSOR_TARGET: $(BPE_PROCESSOR_TARGET))

# Path para o header JSON (para usar como pré-requisito)
JSON_HEADER := third_party/nlohmann/json.hpp

# Default target
all: $(TEXT_CLEANER_TARGET) $(DATA_ANALYZER_TARGET) $(BPE_PROCESSOR_TARGET)
	@echo "----------------------------------------------------"
	@echo "All utilities built: $(BUILD_TYPE_MSG)."
	@echo "  $(TEXT_CLEANER_TARGET)"
	@echo "  $(DATA_ANALYZER_TARGET)"
	@echo "  $(BPE_PROCESSOR_TARGET)"
	@echo "----------------------------------------------------"

# --- Build Rules ---
# Garanta que os diretórios dos alvos existam (se necessário, embora o make geralmente lide bem com isso)
# $(shell mkdir -p $(TEXT_CLEANER_DIR) $(DATA_ANALYZER_DIR) $(BPE_PROCESSOR_DIR))

$(TEXT_CLEANER_TARGET): $(TEXT_CLEANER_SRC) $(JSON_HEADER) Makefile
	@echo "Building Text Cleaner (target: $@, source: $<)..."
	@echo "Command: $(CXX) $(CURRENT_CXXFLAGS) $< -o $@"
	$(CXX) $(CURRENT_CXXFLAGS) $< -o $@

$(DATA_ANALYZER_TARGET): $(DATA_ANALYZER_SRC) $(JSON_HEADER) Makefile
	@echo "Building Data Analyzer (target: $@, source: $<)..."
	@echo "Command: $(CXX) $(CURRENT_CXXFLAGS) $< -o $@"
	$(CXX) $(CURRENT_CXXFLAGS) $< -o $@

$(BPE_PROCESSOR_TARGET): $(BPE_PROCESSOR_SRC) $(JSON_HEADER) Makefile
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
	rm -f $(TEXT_CLEANER_TARGET) $(TEXT_CLEANER_DIR)/lunaris_text_cleaner \
	      $(DATA_ANALYZER_TARGET) $(DATA_ANALYZER_DIR)/lunaris_data_analyzer \
	      $(BPE_PROCESSOR_TARGET) $(BPE_PROCESSOR_DIR)/bpe_processor
	@echo "Cleanup complete."

.PHONY: all text_cleaner data_analyzer bpe_processor clean debug_text_cleaner debug_data_analyzer debug_bpe_processor debug_all
