#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
      307 * rgb[ 3 * ( j * cols + i ) ]
      + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
      + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
    ) / 1024;
  }
}

int main()
{
  cv::Mat img_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = img_in.data;
  auto rows = img_in.rows;
  auto cols = img_in.cols;
  std::vector< unsigned char > g( rows * cols );
  cv::Mat img_out( rows, cols, CV_8UC1, g.data() );
  unsigned char * rgb_d;
  unsigned char * g_d;
  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &g_d, rows * cols );
  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  dim3 dim1( 32, 32 );
  dim3 dim2(( cols - 1) / dim1.x + 1 , ( rows - 1 ) / dim1.y + 1 );
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  grayscale<<< dim2, dim1 >>>( rgb_d, g_d, cols, rows );
  cudaMemcpy(g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  auto cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess){
    std::cout << cudaGetErrorName(cudaError) << std::endl;
    std::cout << cudaGetErrorString(cudaError) << std::endl;
  }
  else {
    std::cout << "No Errors!" << std::endl;
  }
  cudaEventRecord( stop );
  cudaEventSynchronize( stop );
  float duration = 0.0f;
  cudaEventElapsedTime( &duration, start, stop );
  std::cout << "Total: " << duration << "ms\n";
  cv::imwrite( "GSOutput.jpg", img_out );
  cudaFree( rgb_d);
  cudaFree( g_d);
  return 0;
}
