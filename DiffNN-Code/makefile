CC=gcc
CFLAGS += -Wall
LDFLAGS= -lopenblas -lpthread -lm
DEPS= matrix.h nnet.h split.h

all: delta_network_test
all: CFLAGS += -O3
all: LDFLAGS += -O3 

bench:
	$(eval CFLAGS += -DMAX_THREAD=10)

depth:
	$(eval CFLAGS += -DDEPTH_VERIFIED)

debug: run_network delta_network_test test_mnist test_HAR
debug: CFLAGS += -g -O0
debug: LDFLAGS += -g -O0

stats: 
	$(eval CFLAGS += -DNEURONSTATE -DNEURONEQ -DNEURONBOUNDS -DFWDINTERVAL -DLOGGING)

delta_network_test: matrix.o nnet.o delta_network_test.o split.o
	$(CC) -o $@ $^ $(LDFLAGS)

c.o: 
	$(CC) $(CFLAGS) $<  -o $@

clean:
	rm -f *.o delta_network_test

