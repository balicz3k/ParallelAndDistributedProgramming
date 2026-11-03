CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

TARGET = topk_ref
SRC    = topk_ref.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET) results.txt
