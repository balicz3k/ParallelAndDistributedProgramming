CXX      = g++
CXXFLAGS = -O3 -fopenmp -std=c++17 -Wall -Wextra
TARGET   = lab2
SRC      = lab2_openmp.cpp
DATA     = data/plik_3kolumny.txt
REPS     = 15

.PHONY: all bench correctness clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Pelny benchmark -> results_raw.csv (surowe), komunikaty na stderr
bench: $(TARGET)
	./$(TARGET) $(DATA) bench $(REPS) > results_raw.csv
	@echo "Zapisano results_raw.csv"

# Sprawdzenie poprawnosci seq vs par (maks. watkow)
correctness: $(TARGET)
	./$(TARGET) $(DATA) correctness

clean:
	rm -f $(TARGET)
