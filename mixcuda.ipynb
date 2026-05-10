%%writefile vector_add.cu

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












//////////////////////////

//matrix_mul//

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



