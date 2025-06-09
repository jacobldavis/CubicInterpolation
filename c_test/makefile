CC = gcc
NVCC = nvcc
CFLAGS = -O3 -march=native -Ic_frame
NVCCFLAGS = -O3 -Ic_frame
LDFLAGS = -lgsl -lm -lcudart

C_SRC = main.c $(wildcard c_frame/*.c)
CUDA_SRC = c_frame/kernel.cu
C_OBJ = $(C_SRC:.c=.o)
CUDA_OBJ = $(CUDA_SRC:.cu=.o)
OBJ = $(C_OBJ) $(CUDA_OBJ)

TARGET = main.out

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run