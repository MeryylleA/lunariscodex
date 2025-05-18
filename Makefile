# Makefile for Lunaris Codex C++ Utilities

# Compiler
CXX := g++

# Common C++ Standard
CPP_STD := -std=c++17

# Common Warning Flags
WARNING_FLAGS := -Wall -Wextra -pedantic

# Release Flags (default)
OPTIMIZATION_FLAGS := -O2
DEFINE_FLAGS := -DNDEBUG # Define NDEBUG to disable asserts, etc.
CXXFLAGS_RELEASE := $(CPP_STD) $(WARNING_FLAGS) $(OPTIMIZATION_FLAGS) $(DEFINE_FLAGS)

# Debug Flags
CXXFLAGS_DEBUG := $(CPP_STD) $(WARNING_FLAGS) -g -O0 # -g for debug symbols, -O0 to disable optimizations

# Default flags to use (can be overridden)
# Por padrão, vamos usar as flags de release.
# Para compilar com debug, use: make CXXFLAGS_MODE=DEBUG <target>
# Ou defina CXXFLAGS diretamente: make CXXFLAGS="-std=c++17 -g" <target>
CXXFLAGS_MODE ?= RELEASE # Default to RELEASE if not set from command line

ifeq ($(CXXFLAGS_MODE),DEBUG)
    CURRENT_CXXFLAGS := $(CXXFLAGS_DEBUG)
    BUILD_TYPE_MSG := "Debug"
else
    CURRENT_CXXFLAGS := $(CXXFLAGS_RELEASE)
    BUILD_TYPE_MSG := "Release (Default)"
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


# Default target: build all utilities with default (Release) flags
all: text_cleaner data_analyzer
	@echo "----------------------------------------------------"
	@echo "All standard utilities built with $(BUILD_TYPE_MSG) flags."
	@echo "Executables created:"
	@echo "  $(TC_TARGET)"
	@echo "  $(DA_TARGET)"
	@echo "To build with debug flags, run: make CXXFLAGS_MODE=DEBUG"
	@echo "----------------------------------------------------"

# Target to build all CI executables
all_ci: text_cleaner_ci data_analyzer_ci
	@echo "----------------------------------------------------"
	@echo "All CI executables built with $(BUILD_TYPE_MSG) flags (CI usually uses Release)."
	@echo "Executables created for CI:"
	@echo "  $(TC_CI_TARGET)"
	@echo "  $(DA_CI_TARGET)"
	@echo "----------------------------------------------------"


# --- Individual Build Targets ---

# Target to build only the text cleaner (default: release)
text_cleaner: $(TC_TARGET)

$(TC_TARGET): $(TC_SRC) Makefile
	@echo "Building Text Cleaner ($(BUILD_TYPE_MSG))..."
	$(CXX) $(CURRENT_CXXFLAGS) $(TC_SRC) -o $(TC_TARGET)
	@echo "Text Cleaner built as $(TC_TARGET)"

# Target to build text cleaner specifically for CI (uses default flags, typically Release)
text_cleaner_ci: $(TC_CI_TARGET)

$(TC_CI_TARGET): $(TC_SRC) Makefile
	@echo "Building Text Cleaner for CI (using $(BUILD_TYPE_MSG) flags)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(TC_SRC) -o $(TC_CI_TARGET)
	@echo "Text Cleaner for CI built as $(TC_CI_TARGET)"


# Target to build only the data analyzer (default: release)
data_analyzer: $(DA_TARGET)

$(DA_TARGET): $(DA_SRC) Makefile
	@echo "Building Data Analyzer ($(BUILD_TYPE_MSG))..."
	$(CXX) $(CURRENT_CXXFLAGS) $(DA_SRC) -o $(DA_TARGET)
	@echo "Data Analyzer built as $(DA_TARGET)"

# Target to build data analyzer specifically for CI (uses default flags, typically Release)
data_analyzer_ci: $(DA_CI_TARGET)

$(DA_CI_TARGET): $(DA_SRC) Makefile
	@echo "Building Data Analyzer for CI (using $(BUILD_TYPE_MSG) flags)..."
	$(CXX) $(CURRENT_CXXFLAGS) $(DA_SRC) -o $(DA_CI_TARGET)
	@echo "Data Analyzer for CI built as $(DA_CI_TARGET)"


# --- Debug Build Targets (Convenience) ---
debug_text_cleaner:
	$(MAKE) text_cleaner CXXFLAGS_MODE=DEBUG

debug_data_analyzer:
	$(MAKE) data_analyzer CXXFLAGS_MODE=DEBUG

debug_all:
	$(MAKE) all CXXFLAGS_MODE=DEBUG

# --- Clean Target ---
clean:
	@echo "Cleaning C++ utilities..."
	rm -f $(TC_TARGET) $(TC_CI_TARGET) $(DA_TARGET) $(DA_CI_TARGET)
	@echo "Cleanup complete."

# Phony targets
.PHONY: all text_cleaner data_analyzer text_cleaner_ci data_analyzer_ci all_ci clean debug_text_cleaner debug_data_analyzer debug_all
