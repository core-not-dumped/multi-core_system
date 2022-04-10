// using constant memeory

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
using namespace std;

#define NUM_RANGE 101

//INSERT CODE HERE---------------------------------
__global__ void histogram(unsigned int *hist, int *pSource, int input_size)
{
	__shared__ int histShared[NUM_RANGE];
	if(threadIdx.x < NUM_RANGE)
		histShared[threadIdx.x] = 0;

	__syncthreads();
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < input_size)
	{
		int pixelVal = pSource[i];
		atomicAdd(&(histShared[pixelVal]), 1);
	}

	__syncthreads();
	if (threadIdx.x < NUM_RANGE)
		atomicAdd(&(hist[threadIdx.x]), histShared[threadIdx.x]);
}

__global__ void prefix(unsigned int *hist, unsigned int *pre)
{
	int x = threadIdx.x;
	int num = blockDim.x;
	__shared__ unsigned int histShared[128];
	if(x < NUM_RANGE)	histShared[x] = hist[x];
	else				histShared[x] = 0;

	int stride = 1;
	while(stride < num)
	{
		int index = (x + 1) * stride * 2 - 1;
		if(index < num)
			histShared[index] += histShared[index - stride];
		stride *= 2;

		__syncthreads();
	}

	stride = num / 2;
	while(stride > 0)
	{
		int index = (x + 1) * stride * 2 - 1;
		if(index < num && (index+stride) < num)
			histShared[index+stride] += histShared[index];
		stride /= 2;
		__syncthreads();
	}

	if(x < NUM_RANGE)
		pre[x] = histShared[x];	
}

__global__ void fill_matrix(int *result, unsigned int *pre)
{
	int i = threadIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ unsigned int preShared[NUM_RANGE + 1];
	if(i == 0)					preShared[i] = 0;
	else if(i < NUM_RANGE)		preShared[i] = pre[i-1];
	__syncthreads();

	int left = 0, right = NUM_RANGE, h = (left + right) / 2;
	while(left + 1 < right)
	{
		if(preShared[h] > x)			right = h;
		else							left = h;
		h = (left + right) / 2;
	}
	result[x] = h;
}

void verify(int* src, int*result, int input_size){
	sort(src, src+input_size);
	long long match_cnt=0;
	for(int i=0; i<input_size;i++)
	{
		if(src[i]==result[i])
			match_cnt++;
	}

	if(match_cnt==input_size)
		printf("TEST PASSED\n\n");
	else
		printf("TEST FAILED\n\n");

}

void genData(int* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (int)(rand() % 101);
	}
}

int main(int argc, char* argv[]) {
	int* pSource = NULL;
	int* pResult = NULL;
	int input_size=0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (argc == 2)
		input_size=atoi(argv[1]);
	else
	{
    		printf("\n    Invalid input parameters!"
	   		"\n    Usage: ./sort <input_size>"
           		"\n");
        	exit(0);
	}

	//allocate host memory
	pSource=(int*)malloc(input_size*sizeof(int));
	pResult=(int*)malloc(input_size*sizeof(int));
	// generate source data
	genData(pSource, input_size);
	
	// start timer
	cudaEventRecord(start, 0);

	/////////////////////////////// histogram /////////////////////////////////////////////////////////////

	// allocate host memory
	unsigned int *hist;
	unsigned int *pre;
	hist = (unsigned int *)malloc(NUM_RANGE*sizeof(unsigned int));
	pre = (unsigned int *)malloc(NUM_RANGE*sizeof(unsigned int));
	for(int i=0;i<NUM_RANGE;i++)	hist[i] = 0;
	for(int i=0;i<NUM_RANGE;i++)	pre[i] = 0;

	// allocate device memory
	int *pSourcedev = NULL;
	int *pResultdev = NULL;
	unsigned int *histdev = NULL;
	unsigned int *predev = NULL;
	cudaMalloc((void **)&pSourcedev, input_size * sizeof(int));
	cudaMalloc((void **)&pResultdev, input_size * sizeof(int));
	cudaMalloc((void **)&histdev, NUM_RANGE * sizeof(unsigned int));
	cudaMalloc((void **)&predev, NUM_RANGE * sizeof(unsigned int));

	// copy from host to device
	cudaMemcpy(pSourcedev, pSource, input_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(histdev, hist, NUM_RANGE * sizeof(unsigned int), cudaMemcpyHostToDevice);
	// cudaMemcpy(predev, pre, NUM_RANGE * sizeof(unsigned int), cudaMemcpyHostToDevice); // if prefix device kernel active

	// launch the kernel -> histogram
	int BLOCK_SIZE_HIST = 512;
	int GRID_SIZE_HIST = ceil(input_size / (float)BLOCK_SIZE_HIST);
	dim3 dimgrid_hist(GRID_SIZE_HIST, 1, 1);
	dim3 dimblock_hist(BLOCK_SIZE_HIST, 1, 1);
	histogram <<< dimgrid_hist, dimblock_hist >>> (histdev, pSourcedev, input_size);

	// copy from device to host
	cudaMemcpy(hist, histdev, NUM_RANGE * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//////////////////////////////////// prefix //////////////////////////////////////////////////////////////////////

	/****************** prefix device kernel **********************/
	// launch ther kernel -> prefix
	/*int BLOCK_SIZE_PRE = 128;
	int GRID_SIZE_PRE = 1;
	dim3 dimgrid_pre(GRID_SIZE_PRE, 1, 1);
	dim3 dimblock_pre(BLOCK_SIZE_PRE, 1, 1);
	prefix <<< dimgrid_pre, dimblock_pre >>> (histdev, predev);

	// copy from device to host
	cudaMemcpy(pre, predev, NUM_RANGE * sizeof(unsigned int), cudaMemcpyDeviceToHost);*/
	/***************************************************************/
	
	pre[0] = hist[0];
	for(int i=0;i<101;i++)
		pre[i] = pre[i-1] + hist[i];

	////////////////////////////////////// fill matrix ///////////////////////////////////////////////////////////////////

	// copy from host to device
	cudaMemcpy(predev, pre, NUM_RANGE * sizeof(unsigned int), cudaMemcpyHostToDevice); // if prefix device kernel not active

	// launch kernel
	int BLOCK_SIZE_FILL = 512;
	int GRID_SIZE_FILL = ceil(input_size / (float)BLOCK_SIZE_FILL);
	dim3 dimgrid_fill(GRID_SIZE_FILL, 1, 1);
	dim3 dimblock_fill(BLOCK_SIZE_FILL, 1, 1);
	fill_matrix <<< dimgrid_fill, dimblock_fill >>> (pResultdev, predev);

	cudaMemcpy(pResult, pResultdev, input_size * sizeof(int), cudaMemcpyDeviceToHost);

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	// end timer
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("elapsed time = %f msec\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// test code ////////////////
	/*printf("hist: ");
	for(int i = 0;i<101;i++)
		printf("%u ", hist[i]);
	printf("\n");

	printf("pre: ");
	for(int i = 0;i<101;i++)
		printf("%d: %u ", i, pre[i]);
	printf("\n");

	printf("Result: ");
	for(int i = 0;i<=100;i++)
	{
		int j = 0;
		while(1)
		{
			if(i < pResult[j])
			{
				printf("%d: %d ", i, j);
				break;
			}
			j++;
		}
	}
	printf("\n");

	printf("pre: ");
	for(int i = 0;i<input_size;i++)
		printf("%d ", pResult[i]);
	printf("\n");*/
	/////////////////////////////

	printf("Verifying results..."); fflush(stdout);
	verify(pSource, pResult, input_size);
	fflush(stdout);
}
