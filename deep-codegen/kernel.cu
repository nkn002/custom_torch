#include "kernel.h"

/**
 * CUDA kernel for matrix transpose operation.
 * Reference: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
 */
__global__ void transpose_kernel(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}


/**
 * CUDA kernel for matrix multiplication operation.
 * Reference: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
 */
__global__ void matmul(float *a,float *b, float *c, int m, int n, int k)
{ 
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y; 
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

/**
 * CUDA kernel for matrix summation operation.
 */
__global__ void sumMatrices(float* A, float* B, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] + B[index];
    }
}


void transpose(array2d_t<float>& a, array2d_t<float>& output){
    int row = a.row_count;
    int col = a.col_count;
    dim3 dimBlock(16, 16);
    dim3 dimGrid((dimBlock.x + col - 1) / dimBlock.x, (dimBlock.y + row - 1) / dimBlock.y);
    transpose_kernel<<<dimGrid, dimBlock>>>(a.data_ptr, output.data_ptr, row, col);
    cudaDeviceSynchronize();
}

void mm(array2d_t<float>& a, array2d_t<float>& b, array2d_t<float>& output){
    int m = a.row_count;
    int n = a.col_count;
    int k = b.col_count;
    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((dimBlock.x + k - 1) / dimBlock.x, (dimBlock.y + m - 1) / dimBlock.y);
    matmul<<<dimGrid, dimBlock>>>(a.data_ptr, b.data_ptr, output.data_ptr, m, n, k);
    cudaDeviceSynchronize();
}

void sum_two_tensors(array2d_t<float>& a, array2d_t<float>& b, array2d_t<float>& output){
    int m = a.row_count;
    int n = a.col_count;
    dim3 dimBlock(16, 16);
    dim3 dimGrid((dimBlock.x + n - 1) / dimBlock.x, (dimBlock.y + m - 1) / dimBlock.y);
    sumMatrices<<<dimGrid, dimBlock>>>(a.data_ptr, b.data_ptr, output.data_ptr, m, n);
    cudaDeviceSynchronize();
}
void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm){;}
void gspmmve(graph_t& graph, array2d_t<float>& input1, array1d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmme(graph_t& graph, array1d_t<float>& edge_input, array1d_t<float>& output, op_t op, bool reverse){;}
void gspmme2d(graph_t& graph, array2d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmmve2d(graph_t& graph, array3d_t<float>& input1, array2d_t<float>& edge_input, array3d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve(graph_t& graph, array1d_t<float>& input_left, array1d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve2d(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv2d(graph_t& graph, array3d_t<float>& input_left, array3d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void test_2out(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test3(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test4(array3d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, int t){;}
