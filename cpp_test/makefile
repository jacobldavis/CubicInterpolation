CC = g++
CFLAGS = -O3 -march=native
LDFLAGS = -lgsl -lm
SRC = main.c $(wildcard cpp_frame/*.cpp)
OBJ = $(SRC:.c=.o)
TARGET = main.out

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

run:
	./main.out
