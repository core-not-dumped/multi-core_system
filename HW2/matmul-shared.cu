#include <iostream>
#include <cstdlib>
#include <sys/time.h>

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

//CUDA kernel size settings
const int TILE_WIDTH = 32; // block will be (TILE_WIDTH,TILE_WIDTH)

//random data generation
void genData(float* ptr, unsigned int size) {
	while (size) {
		*ptr++ = (float)size/(float)1000;
		size--;
	}
}

__global__ void matmul(float* g_C, const float* g_A, const float* g_B, const int width1, const int width2, const int width3) {
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

	int by = blockIdx.y; int bx = blockIdx.x;
	int ty = threadIdx.y; int tx = threadIdx.x;

	int gy = by * TILE_WIDTH + ty; // global y index
	int gx = bx * TILE_WIDTH + tx; // global x index

	float sum = 0.0F;
	for (register int m = 0; m < ceil(width2 / (float)TILE_WIDTH); ++m) {
	
		// read into the shared memory blocks
		if((m * TILE_WIDTH + tx) >= width2 || gy >= width1)		s_A[ty][tx] = 0.0F;
		else													s_A[ty][tx] = g_A[gy * width2 + (m * TILE_WIDTH + tx)];
		if((m * TILE_WIDTH + ty) >= width2 || gx >= width3)		s_B[ty][tx] = 0.0F;
		else													s_B[ty][tx] = g_B[(m * TILE_WIDTH + ty) * width3 + gx];
		__syncthreads();

		for (register int k = 0; k < TILE_WIDTH; ++k) {
			sum += s_A[ty][k] * s_B[k][tx];
		}
		__syncthreads();
	}
	if(gy < width1 && gx < width3)	g_C[gy * width3 + gx] = sum;
}

int main(int argc, char* argv[]) {

	// host-side data
	const int WIDTH1 = atoi(argv[1]);
	const int WIDTH2 = atoi(argv[2]);
	const int WIDTH3 = atoi(argv[3]);
	const int GRID_WIDTH1 = ceil(WIDTH1 / (float)TILE_WIDTH); // grid will be (GRID_WDITH,GRID_WDITH)
	const int GRID_WIDTH2 = ceil(WIDTH2 / (float)TILE_WIDTH);
	const int GRID_WIDTH3 = ceil(WIDTH3 / (float)TILE_WIDTH); 

	float* pA = NULL;
	float* pB = NULL;
	float* pC = NULL;
	struct timeval start_time, end_time;

	// malloc memories on the host-side
	pA = (float*)malloc(WIDTH2 * WIDTH1 * sizeof(float));
	pB = (float*)malloc(WIDTH3 * WIDTH2 * sizeof(float));
	pC = (float*)malloc(WIDTH3 * WIDTH1 * sizeof(float));
	for(int i=0;i < WIDTH1 * WIDTH3;i++)	pC[i] = 0.0;

	// generate source data
	genData(pA, WIDTH2 * WIDTH1);
	genData(pB, WIDTH3 * WIDTH2);

	// CUDA: allocate device memory
	float* pAdev = NULL;
	float* pBdev = NULL;
	float* pCdev = NULL;
	CUDA_CHECK( cudaMalloc((void**)&pAdev, WIDTH2 * WIDTH1 * sizeof(float)) );
	CUDA_CHECK( cudaMalloc((void**)&pBdev, WIDTH3 * WIDTH2 * sizeof(float)) );
	CUDA_CHECK( cudaMalloc((void**)&pCdev, WIDTH3 * WIDTH1 * sizeof(float)) );

	// copy from host to device
	CUDA_CHECK( cudaMemcpy(pAdev, pA, WIDTH2 * WIDTH1 * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(pBdev, pB, WIDTH3 * WIDTH2 * sizeof(float), cudaMemcpyHostToDevice) );

	//get current time
	cudaThreadSynchronize();
	gettimeofday(&start_time, NULL);

	// CUDA: launch the kernel
	dim3 dimGrid(GRID_WIDTH3, GRID_WIDTH1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	matmul <<< dimGrid, dimBlock >>> (pCdev, pAdev, pBdev, WIDTH1, WIDTH2, WIDTH3);
	CUDA_CHECK( cudaPeekAtLastError() );

	//get current time
	cudaThreadSynchronize();
	gettimeofday(&end_time, NULL);
	double operating_time = (double)(end_time.tv_sec)+(double)(end_time.tv_usec)/1000000.0-((double)(start_time.tv_sec)+(double)(start_time.tv_usec)/1000000.0);
	printf("Elapsed: %f seconds\n", (double)operating_time);

	// copy from device to host
	CUDA_CHECK( cudaMemcpy(pC, pCdev, WIDTH1 * WIDTH3 * sizeof(float), cudaMemcpyDeviceToHost) );

	// free device memory
	CUDA_CHECK( cudaFree(pAdev) );
	CUDA_CHECK( cudaFree(pBdev) );
	CUDA_CHECK( cudaFree(pCdev) );

	// print sample cases
	int i, j;
	for(i=0;i<WIDTH1;i++)
	{
		for(j=0;j<WIDTH3;j++)
			std::cout << pC[i*WIDTH3+j] << " ";
		std::cout << std::endl;
	}


	// done
	return 0;
}
