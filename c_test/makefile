CC = gcc
NVCC = nvcc
CFLAGS = -O3 -march=native -Ic_frame -fopenmp
NVCCFLAGS = -O3 -Ic_frame
LDFLAGS = -lgsl -lm -lcudart -lOpenCL

C_SRC = main.c $(wildcard c_frame/*.c)
CUDA_SRC = c_frame/cubic_interp_cuda.cu

BUILD_DIR = build
C_OBJ = $(patsubst %.c,$(BUILD_DIR)/%.o,$(C_SRC))
CUDA_OBJ = $(patsubst %.cu,$(BUILD_DIR)/%.o,$(CUDA_SRC))
OBJ = $(C_OBJ) $(CUDA_OBJ)

TARGET = main.out

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run