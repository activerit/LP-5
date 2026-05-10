//%%writefile vector_add.cu

#include <stdio.h>
#include <cuda.h>

#define N 10

// Kernel Function (Runs on GPU)
__global__ void vectorAdd(int *a, int *b, int *c)
{
    int id = threadIdx.x;

    if(id < N)
    {
        c[id] = a[id] + b[id];
    }
}

int main()
{
    int a[N], b[N], c[N];

    int *d_a, *d_b, *d_c;

    // Initialize vectors
    for(int i = 0; i < N; i++)
    {
        a[i] = i + 1;
        b[i] = (i + 1) * 2;
    }

    // Allocate memory on GPU
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch Kernel
    vectorAdd<<<1, N>>>(d_a, d_b, d_c);

    // Copy result back to CPU
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print Result
    printf("Vector Addition Result:\n");

    for(int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free GPU Memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}