CC = gcc
CFLAGS = -O3 -march=native
LDFLAGS = -lgsl -lm
SRC = main.c $(wildcard c_frame/*.c)
OBJ = $(SRC:.c=.o)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

run:
	./main
