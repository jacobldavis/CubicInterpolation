CC = g++
CFLAGS = -O3 -march=native
LIBS = -lgsl -lm
SRC = main.cpp
CPP_FRAME_SRC = $(wildcard cpp_frame/*.cpp)
OBJ = main.o $(CPP_FRAME_SRC:.cpp=.o)
TARGET = main.out

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

run:
	./$(TARGET)
