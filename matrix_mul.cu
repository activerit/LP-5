%%writefile matrix_mul.cu

#include <stdio.h>
#include <cuda.h>

#define WIDTH 2

// Kernel Function
__global__ void matrixMul(int *A, int *B, int *C)
{
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;

    for(int k = 0; k < WIDTH; k++)
    {
        sum += A[row * WIDTH + k] * B[k * WIDTH + col];
    }

    C[row * WIDTH + col] = sum;
}

int main()
{
    int A[WIDTH][WIDTH] = {
        {1, 2},
        {3, 4}
    };

    int B[WIDTH][WIDTH] = {
        {5, 6},
        {7, 8}
    };

    int C[WIDTH][WIDTH];

    int *d_A, *d_B, *d_C;

    int size = WIDTH * WIDTH * sizeof(int);

    // Allocate GPU Memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices from CPU to GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Create Threads
    dim3 threads(WIDTH, WIDTH);

    // Launch Kernel
    matrixMul<<<1, threads>>>(d_A, d_B, d_C);

    // Copy Result back to CPU
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print Result
    printf("Matrix Multiplication Result:\n");

    for(int i = 0; i < WIDTH; i++)
    {
        for(int j = 0; j < WIDTH; j++)
        {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    // Free GPU Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}