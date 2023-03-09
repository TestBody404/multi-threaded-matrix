CC=g++
CFLAGS=-c -Wall -fopenmp -pthread
LDFLAGS=-fopenmp -pthread
 
SOURCES=matrix.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=out
 
all: $(SOURCES) $(EXECUTABLE)
 
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ 
 
.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
