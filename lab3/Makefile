CXX      = g++
CXXFLAGS = -O2 -fopenmp -std=c++17
TARGET   = lab3
SRC      = lab3_conv.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

benchmark: $(TARGET)
	@echo "=== 512x512, 3x3 ==="
	@OMP_NUM_THREADS=1 ./$(TARGET) 512 3 1
	@OMP_NUM_THREADS=4 ./$(TARGET) 512 3 4
	@OMP_NUM_THREADS=8 ./$(TARGET) 512 3 8
	@echo ""
	@echo "=== 2048x2048, 7x7 ==="
	@OMP_NUM_THREADS=1 ./$(TARGET) 2048 7 1
	@OMP_NUM_THREADS=4 ./$(TARGET) 2048 7 4
	@OMP_NUM_THREADS=8 ./$(TARGET) 2048 7 8
	@echo ""
	@echo "=== 4096x4096, 15x15 ==="
	@OMP_NUM_THREADS=1 ./$(TARGET) 4096 15 1
	@OMP_NUM_THREADS=4 ./$(TARGET) 4096 15 4
	@OMP_NUM_THREADS=8 ./$(TARGET) 4096 15 8

clean:
	rm -f $(TARGET)

.PHONY: benchmark clean
