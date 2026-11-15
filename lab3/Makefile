CXX      = g++
CXXFLAGS = -O3 -fopenmp -std=c++17 -Wall -Wextra
TARGET   = lab3
SRC      = lab3_conv.cpp
REPS     = 5

.PHONY: all benchmark clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Trzy konfiguracje (N, K); surowe czasy trafiaja do results_raw.csv
benchmark: $(TARGET)
	./$(TARGET) 512 3 $(REPS)    >  results_raw.csv
	./$(TARGET) 2048 7 $(REPS)   >> results_raw.csv
	./$(TARGET) 4096 15 $(REPS)  >> results_raw.csv
	@echo "Zapisano results_raw.csv"

clean:
	rm -f $(TARGET)
