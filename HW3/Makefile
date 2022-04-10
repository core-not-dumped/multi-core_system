
NVCC        = nvcc
NVCC_FLAGS  = --std=c++03
#LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
#LD_FLAGS    = -L/usr/local/cuda/lib64
EXE	        = convolution
OBJ	        = support.o main.o 

default: $(EXE)

main.o: main.cu kernel.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
