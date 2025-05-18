# data_analyzer/Makefile

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -O2
# -Wall: Turns on all most common warnings
# -Wextra: Turns on some extra warnings not covered by -Wall
# -pedantic: Issues all warnings demanded by strict ISO C++
# -O2: Optimization level 2

# Target executable name
TARGET := lunaris_data_analyzer

# Source files
SRCS := lunaris_data_analyzer.cpp
# SRCS := lunaris_data_analyzer.cpp another_file.cpp yet_another.cpp

# Object files (auto-generated from SRCS)
OBJS := $(SRCS:.cpp=.o)

# Default target: build the executable
all: $(TARGET)

# Rule to link the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# Rule to compile source files into object files
# %.o: %.cpp
#	$(CXX) $(CXXFLAGS) -c $< -o $@
# Esta regra explícita para .cpp -> .o é muitas vezes implícita, mas é bom tê-la.
# No entanto, para um único arquivo de origem,
# o g++ pode compilar e linkar em um único passo, como no exemplo abaixo
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Target to clean up build files
clean:
	rm -f $(TARGET) $(OBJS)

# Phony targets (targets that are not files)
.PHONY: all clean
