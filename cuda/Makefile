NVCC = nvcc
CFLAGS = -I/usr/include
LDFLAGS = -L/mnt/c/Users/Kevin/Desktop/wsl/neuralnetworkcpp/cpp_imp/cuda_lib
LIBS = -lcudart -lcublas -lcurand

all: test a.out

test: activation.o layer.o model.o helpers.o test.o
	$(NVCC) $(CFLAGS) $(LDFLAGS) activation.o helpers.o layer.o model.o test.o  -o test $(LIBS)

a.out: activation.o layer.o model.o module.o helpers.o
	$(NVCC) $(CFLAGS) $(LDFLAGS) activation.o helpers.o layer.o model.o module.o  -o a.out $(LIBS)

activation.o: activation.cu activation.h
	$(NVCC) -c activation.cu -o activation.o $(CFLAGS)

helpers.o: helpers.cu helpers.h
	$(NVCC) -c helpers.cu -o helpers.o $(CFLAGS)

layer.o: layer.cu layer.h
	$(NVCC) -c layer.cu -o layer.o $(CFLAGS)

model.o: model.cu model.h
	$(NVCC) -c model.cu -o model.o $(CFLAGS)

test.o: test.cu
	$(NVCC) -c test.cu -o test.o $(CFLAGS)

module.o: module.cu module.h
	$(NVCC) -c module.cu -o module.o $(CFLAGS)

clean:
	rm -f *.o test





