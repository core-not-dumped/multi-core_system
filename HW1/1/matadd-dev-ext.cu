#include <iostream>
#include <cstdlib>

#ifdef DEBUG
#define CUDA_CHECK(x) do {\
	(x);\
	cudaError_t e = cudaGetLastError();\
	if(cudaSuccess !=e){\
		printf("cuda failure \"%s\" at %s:%d\n",\
			cudaGetErrorString(e),\
			__FILE__,__LINE__);\
		exit(1);\
	}\
}while(0)
#else
#define CUDA_CHECK(x)	(x)
#endif

// kernel program for the device (GPU): compiled by NVCC
__global__ void addKernel(int* c, const int* a, const int* b, const int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = y * width + x; // [y][x] = y * WIDTH + x;
	c[i] = a[i] + b[i];
}

// main program for the CPU: compiled by MS-VC++
int main(int argc, char* argv[]) {

	// host-side data
	const int HEIGHT = atoi(argv[1]);
	const float TILE_HEIGHT = 16;
	const int WIDTH = atoi(argv[2]);
	const float TILE_WIDTH = 16;
	int a[HEIGHT][WIDTH];
	int b[HEIGHT][WIDTH];
	int c[HEIGHT][WIDTH] = {0};

	// make a, b matrices
	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			a[y][x] = y * 10 + x;
			b[y][x] = (y * 10 + x) * 10000;
		}
	}

	// device-side data
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;

	// allocate device memory
	CUDA_CHECK( cudaMalloc((void**)&dev_a, HEIGHT * WIDTH * sizeof(int)) );
	CUDA_CHECK( cudaMalloc((void**)&dev_b, HEIGHT * WIDTH * sizeof(int)) );
	CUDA_CHECK( cudaMalloc((void**)&dev_c, HEIGHT * WIDTH * sizeof(int)) );

	// copy from host to device
	CUDA_CHECK( cudaMemcpy(dev_a, a, HEIGHT * WIDTH * sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(dev_b, b, HEIGHT * WIDTH * sizeof(int), cudaMemcpyHostToDevice) );

	// launch a kernel on the GPU with one thread for each element.
	dim3 dimGrid(ceil(WIDTH/TILE_WIDTH), ceil(HEIGHT/TILE_HEIGHT), 1);
	dim3 dimBlock(int(TILE_WIDTH), int(TILE_HEIGHT), 1); // x, y, z
	addKernel <<< dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, WIDTH); // dev_c = dev_a + dev_b;
	CUDA_CHECK( cudaPeekAtLastError() );

	// copy from device to host
	CUDA_CHECK( cudaMemcpy(c, dev_c, HEIGHT * WIDTH * sizeof(int), cudaMemcpyDeviceToHost) );

	// free device memory
	CUDA_CHECK( cudaFree(dev_c) );
	CUDA_CHECK( cudaFree(dev_a) );
	CUDA_CHECK( cudaFree(dev_b) );

	// print the result
	for (int y = 0; y < HEIGHT; ++y) {
		for (int x = 0; x < WIDTH; ++x)
			printf("%8d", c[y][x]);
		printf("\n");
	}
	// done
	return 0;
}



