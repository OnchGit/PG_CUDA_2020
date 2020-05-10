#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscaleShared ( unsigned char * in,   unsigned char * out, std::size_t w, std::size_t h) {
  auto op1 = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto op2 = blockIdx.y * (blockDim.y-2) + threadIdx.y;
  auto op3 = threadIdx.x;
  auto op4 = threadIdx.y;
  extern __shared__ unsigned char sharedExt[];
  if( op1 < w && op2 < h ) {
		sharedExt[ (op4 * blockDim.x + op3) ] = (
		  307 * in[ 3 * ( op2 * w + op1 ) ]
		  + 604 * in[ 3 * ( op2 * w + op1 ) + 1 ]
		  + 113 * in[  3 * ( op2 * w + op1 ) + 2 ]
		) / 1024;
		__syncthreads();
		out[(op2 * w + op1)] = sharedExt[(op4 * blockDim.x + op3)];
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
  std::size_t size = img_in.cols * img_in.rows;
  std::size_t sizeRGB = 3 * img_in.cols * img_in.rows;
  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &out, rows * cols );
  cudaStream_t streams[ 2 ];
  cudaStreamCreate( &streams[ 0 ] );
  cudaStreamCreate( &streams[ 1 ] );
  cudaMemcpyAsync( rgb_d, rgb, sizeRGB/2, cudaMemcpyHostToDevice, streams[ 0 ] );
  cudaMemcpyAsync( rgb_d+sizeRGB/2, rgb+sizeRGB/2, sizeRGB/2, cudaMemcpyHostToDevice, streams[ 1 ] );
  dim3 dim1( 32, 32 );
  dim3 dim2(  (( cols ) / ((dim1.x - 2) + 1) ), (( rows ) / ((dim1.y - 2) + 1) ));
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  grayscaleShared<<< dim2, dim1, dim1.x * dim1.y, streams[ 0 ] >>>( rgb_d, out, cols + 2, rows);
  grayscaleShared<<< dim2, dim1, dim1.x * dim1.y, streams[ 1 ] >>>( rgb_d+sizeRGB/2, out+size/2, cols + 2, rows);
  cudaMemcpyAsync( g.data(), out, size/2, cudaMemcpyDeviceToHost, streams[ 0 ] );
  cudaMemcpyAsync( g.data()+size/2, out+size/2, size/2, cudaMemcpyDeviceToHost, streams[ 1 ] );
  cudaDeviceSynchronize();
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
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
  cv::imwrite( "outGrayscaleShared.jpg", img_out );
  cudaFree( rgb_d);
  cudaFree ( out);


  return 0;
}
