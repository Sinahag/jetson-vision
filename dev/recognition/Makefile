# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 `pkg-config --cflags opencv4`

# Linker flags
LDFLAGS = `pkg-config --libs opencv4` -lopencv_dnn

# Source files
SRCS = person_recognition.cpp

# Executable name
TARGET = person_recognition

# Build target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Clean target
clean:
	rm -f $(TARGET)
