#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscaleShared ( unsigned char * in,   unsigned char * out, std::size_t w, std::size_t h) {

  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  auto op1 = threadIdx.x;
  auto op2 = threadIdx.y;

  extern __shared__ unsigned char sharedExt[];

  if( i < w && j < h ) {
		sharedExt[ (op2 * blockDim.x + op1) ] = (
		  307 * in[ 3 * ( j * w + i ) ]
		  + 604 * in[ 3 * ( j * w + i ) + 1 ]
		  + 113 * in[  3 * ( j * w + i ) + 2 ]
		) / 1024;
		__syncthreads();
		out[(j * w + i)] = sharedExt[(op2 * blockDim.x + op1)];
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
  unsigned char * out;
  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &out, rows * cols );
  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  dim3 dim1( 32, 32 );
  dim3 dim2(  (( cols - 1) / (dim1.x-2) + 1) , ( rows - 1 ) / (dim1.y-2) + 1 );
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  grayscaleShared<<< dim2, dim1, dim1.x*dim1.y >>>( rgb_d, out, cols, rows );
  cudaMemcpy(g.data(), out, rows * cols, cudaMemcpyDeviceToHost);
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
  cv::imwrite( "GrayscaleSharedOutput.jpg", img_out );
  cudaFree( rgb_d);
  cudaFree ( out);
  return 0;
}
