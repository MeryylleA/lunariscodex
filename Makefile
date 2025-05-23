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

# OS-specific executable suffix
EXE_SUFFIX :=
ifeq ($(OS),Windows_NT)
    EXE_SUFFIX := .exe
    # Para <filesystem> com MinGW, pode ser necessário linkar explicitamente
    # BPE_PROCESSOR_LIBS := -lstdc++fs # Já definido abaixo, mas certifique-se que seu MinGW suporta.
endif

# --- Tool Definitions ---
TEXT_CLEANER_SRC := text_cleaner/lunaris_text_cleaner.cpp
TEXT_CLEANER_TARGET := text_cleaner/lunaris_text_cleaner$(EXE_SUFFIX)

DATA_ANALYZER_SRC := data_analyzer/lunaris_data_analyzer.cpp
DATA_ANALYZER_TARGET := data_analyzer/lunaris_data_analyzer$(EXE_SUFFIX)

BPE_PROCESSOR_SRC := bpe_trainer/bpe_processor.cpp
BPE_PROCESSOR_TARGET := bpe_trainer/bpe_processor$(EXE_SUFFIX)
# Para g++ >= 9, -lstdc++fs pode ser suficiente. Para MinGW, pode precisar de -static-libstdc++ -static-libgcc se quiser portabilidade e o filesystem não estiver linkando bem.
# No entanto, o -lstdc++fs é o padrão a tentar primeiro.
BPE_PROCESSOR_LIBS := -lstdc++fs

# --- Debugging ---
$(info INFO: OS = $(OS))
$(info INFO: EXE_SUFFIX = $(EXE_SUFFIX))
$(info INFO: CURRENT_CXXFLAGS = $(CURRENT_CXXFLAGS))
$(info INFO: TEXT_CLEANER_TARGET = $(TEXT_CLEANER_TARGET))
$(info INFO: DATA_ANALYZER_TARGET = $(DATA_ANALYZER_TARGET))
$(info INFO: BPE_PROCESSOR_TARGET = $(BPE_PROCESSOR_TARGET))

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
# Adicionado $(JSON_HEADER) como pré-requisito para cada alvo que possa usá-lo.
# Se um alvo específico não usa JSON, você pode remover $(JSON_HEADER) da sua lista de pré-requisitos.
$(TEXT_CLEANER_TARGET): $(TEXT_CLEANER_SRC) $(JSON_HEADER) Makefile
	@echo "Building Text Cleaner..."
	$(CXX) $(CURRENT_CXXFLAGS) $(TEXT_CLEANER_SRC) -o $(TEXT_CLEANER_TARGET)

$(DATA_ANALYZER_TARGET): $(DATA_ANALYZER_SRC) $(JSON_HEADER) Makefile
	@echo "Building Data Analyzer..."
	$(CXX) $(CURRENT_CXXFLAGS) $(DATA_ANALYZER_SRC) -o $(DATA_ANALYZER_TARGET)

$(BPE_PROCESSOR_TARGET): $(BPE_PROCESSOR_SRC) $(JSON_HEADER) Makefile
	@echo "Building BPE Processor..."
	$(CXX) $(CURRENT_CXXFLAGS) $(BPE_PROCESSOR_SRC) -o $(BPE_PROCESSOR_TARGET) $(BPE_PROCESSOR_LIBS)

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
	rm -f text_cleaner/lunaris_text_cleaner* \
	      data_analyzer/lunaris_data_analyzer* \
	      bpe_trainer/bpe_processor*
	@echo "Cleanup complete."

.PHONY: all text_cleaner data_analyzer bpe_processor clean debug_text_cleaner debug_data_analyzer debug_bpe_processor debug_all
