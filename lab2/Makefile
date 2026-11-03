CXX      = g++
CXXFLAGS = -O2 -fopenmp -std=c++17
TARGET   = lab2
SRC      = lab2_openmp.cpp

.PHONY: lab2 benchmark clean

# ------------------------------------------------
# Kompilacja
# ------------------------------------------------
lab2: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)
	@echo "Skompilowano: ./$(TARGET)"

# ------------------------------------------------
# Benchmark: 1, 2, 4, 8 wątków
# Zakłada że plik danych jest w data/chmura_punktow.txt
# lub można nadpisać: make benchmark DATA=ścieżka/do/pliku.txt
# ------------------------------------------------
DATA ?= data/chmura_punktow.txt

benchmark: lab2
	@if [ ! -f "$(DATA)" ]; then \
		echo "Błąd: brak pliku danych: $(DATA)"; \
		echo "Ustaw ścieżkę: make benchmark DATA=<ścieżka>"; \
		exit 1; \
	fi
	@echo ""
	@echo "=== Benchmark: $(DATA) ==="
	@echo "Wątki | Wynik"
	@echo "------|------"
	@for t in 1 2 4 8; do \
		echo "--- $$t wątek(wątki) ---"; \
		./$(TARGET) $(DATA) $$t; \
		echo ""; \
	done

# ------------------------------------------------
clean:
	rm -f $(TARGET)
	@echo "Usunięto binarkę."
