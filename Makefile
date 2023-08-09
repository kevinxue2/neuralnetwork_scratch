NVCC = nvcc
CFLAGS = -I/usr/include
LDFLAGS = -L/mnt/c/Users/Kevin/Desktop/wsl/neuralnetworkcpp/cpp_imp/cuda_lib
LIBS = -lcudart -lcublas -lcurand

all: test

test: activation.o layer.o model.o module.o helpers.o
	$(NVCC) $(CFLAGS) $(LDFLAGS) activation.o helpers.o layer.o model.o module.o  -o test $(LIBS)

activation.o: activation.cu activation.h
	$(NVCC) -c activation.cu -o activation.o $(CFLAGS)

helpers.o: helpers.cu helpers.h
	$(NVCC) -c helpers.cu -o helpers.o $(CFLAGS)

layer.o: layer.cu layer.h
	$(NVCC) -c layer.cu -o layer.o $(CFLAGS)

model.o: model.cu model.h
	$(NVCC) -c model.cu -o model.o $(CFLAGS)

module.o: module.cu module.h
	$(NVCC) -c module.cu -o module.o $(CFLAGS)

clean:
	rm -f *.o test





