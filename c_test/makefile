CC = gcc
CFLAGS = -O3 -march=native
LDFLAGS = -lgsl -lm
SRC = main.c $(wildcard c_frame/*.c)
OBJ = $(SRC:.c=.o)
DEP = $(OBJ:.o=.d)
TARGET = main.out

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.d: %.c
	$(CC) $(CFLAGS) -M -o $@ $<

clean:
	rm -f $(OBJ) $(DEP) $(TARGET)

run: $(TARGET)
	./$(TARGET)

-include $(DEP)
