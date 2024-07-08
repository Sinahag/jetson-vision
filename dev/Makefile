# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = `pkg-config --cflags opencv4` -std=c++11 -pthread -I/usr/include/opencv4

# Linker flags
LDFLAGS = `pkg-config --libs opencv4`

# Source files
SRCS = person_detect.cpp

# Output binary
TARGET = person_detect

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# Rule to clean up the build
clean:
	rm -f $(TARGET)

# Rule to run the target
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

