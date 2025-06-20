CC = gcc
NVCC = nvcc
CFLAGS = -O3 -march=native -Ic_frame -fopenmp
CFLAGS_OFFLOAD = -O2 -Ic_frame -fopenmp -foffload=nvptx-none -foffload="-O3 -w -fcf-protection=none" -foffload=-lm -fno-stack-protector -fcf-protection=none -fno-lto
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
	$(CC) $(CFLAGS_OFFLOAD) -o $@ $(OBJ) $(LDFLAGS)

$(BUILD_DIR)/main.o: main.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS_OFFLOAD) -c $< -o $@

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