# Makefile for Lunaris Codex C++ Utilities

# Compiler and common flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -O2
# -Wall: Enable most common warnings
# -Wextra: Enable some extra warnings
# -pedantic: Issue all warnings demanded by strict ISO C++
# -O2: Optimization level 2

# --- Text Cleaner ---
TC_DIR := text_cleaner
TC_SRC := $(TC_DIR)/lunaris_text_cleaner.cpp
TC_TARGET_NAME := lunaris_text_cleaner
TC_TARGET := $(TC_DIR)/$(TC_TARGET_NAME)
# For CI, we might want a specific name like _ci_executable, or just use the default.
# Let's use a distinct name for CI builds if needed by the CI script,
# otherwise, the default TC_TARGET_NAME is fine.
TC_CI_TARGET_NAME := lunaris_text_cleaner_ci_executable
TC_CI_TARGET := $(TC_DIR)/$(TC_CI_TARGET_NAME)

# --- Data Analyzer ---
DA_DIR := data_analyzer
DA_SRC := $(DA_DIR)/lunaris_data_analyzer.cpp
DA_TARGET_NAME := lunaris_data_analyzer
DA_TARGET := $(DA_DIR)/$(DA_TARGET_NAME)
DA_CI_TARGET_NAME := lda_ci_executable # As per your CI
DA_CI_TARGET := $(DA_DIR)/$(DA_CI_TARGET_NAME)


# Default target: build all utilities
all: text_cleaner data_analyzer

# Target to build only the text cleaner
text_cleaner: $(TC_TARGET)

$(TC_TARGET): $(TC_SRC)
	@echo "Building Text Cleaner..."
	$(CXX) $(CXXFLAGS) $(TC_SRC) -o $(TC_TARGET)
	@echo "Text Cleaner built as $(TC_TARGET)"

# Target to build text cleaner specifically for CI (if different name needed)
text_cleaner_ci: $(TC_CI_TARGET)

$(TC_CI_TARGET): $(TC_SRC)
	@echo "Building Text Cleaner for CI..."
	$(CXX) $(CXXFLAGS) $(TC_SRC) -o $(TC_CI_TARGET)
	@echo "Text Cleaner for CI built as $(TC_CI_TARGET)"


# Target to build only the data analyzer
data_analyzer: $(DA_TARGET)

$(DA_TARGET): $(DA_SRC)
	@echo "Building Data Analyzer..."
	$(CXX) $(CXXFLAGS) $(DA_SRC) -o $(DA_TARGET)
	@echo "Data Analyzer built as $(DA_TARGET)"

# Target to build data analyzer specifically for CI
data_analyzer_ci: $(DA_CI_TARGET)

$(DA_CI_TARGET): $(DA_SRC)
	@echo "Building Data Analyzer for CI..."
	$(CXX) $(CXXFLAGS) $(DA_SRC) -o $(DA_CI_TARGET)
	@echo "Data Analyzer for CI built as $(DA_CI_TARGET)"


# Target to clean all built C++ utilities
clean:
	@echo "Cleaning C++ utilities..."
	rm -f $(TC_TARGET) $(TC_CI_TARGET) $(DA_TARGET) $(DA_CI_TARGET)
	@echo "Cleanup complete."

# Phony targets (targets that are not actual files)
.PHONY: all text_cleaner data_analyzer text_cleaner_ci data_analyzer_ci clean
