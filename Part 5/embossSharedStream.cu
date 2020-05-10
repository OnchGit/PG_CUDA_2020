#include <opencv2/opencv.hpp>
#include <vector>

__global__ void embossShared ( unsigned char * data,   unsigned char * out, std::size_t w, std::size_t h) {
  auto op1 = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto op2 = blockIdx.y * (blockDim.y-2) + threadIdx.y;
  auto op3 = threadIdx.x;
  auto op4 = threadIdx.y;
  extern __shared__ unsigned char sharedExt[];
  if( op1 < w && op2 < h ) {
    sharedExt[3 * (op4 * blockDim.x + op3) ] = data[ 3 * ( op2 * w + op1 ) ];
    sharedExt[3 * (op4 * blockDim.x + op3) + 1 ] = data[ 3 * ( op2 * w + op1 ) + 1];
    sharedExt[3 * (op4 * blockDim.x + op3) + 2 ] = data[ 3 * ( op2 * w + op1 ) + 2 ];
    __syncthreads();
    auto op5 = blockDim.x;
    if( op3 > 0 && op3 < (blockDim.x - 1) && op4 > 0 && op4 < (blockDim.y - 1) )
    {
      for (auto c = 0; c < 3; ++c){
			auto op6 = sharedExt[((op4 - 1) * op5 + op3 - 1) * 3 + c] * -18 + sharedExt[((op4 - 1) * op5 + op3 + 1) * 3 + c] * 0
			+ sharedExt[( op4* op5 + op3 - 1) * 3 + c] * -9 + sharedExt[( op4 * op5 + op3 + 1) * 3 + c] * 9
			+ sharedExt[((op4 + 1) * op5 + op3 - 1) * 3 + c] * 0  + sharedExt[((op4 + 1) * op5 + op3 + 1) * 3 + c] * 18
			+ sharedExt[(( op4 - 1) * op5 + op3) * 3 + c] * -9 + 9 * sharedExt[( op4 * op5 + op3) * 3 + c]
			+ sharedExt[(( op4 + 1) * op5 + op3) * 3 + c] * 9;
			out[(op2 * w + op1) * 3 + c] = (op6 / 9);
		}
    }
  }
}

int main()
{
  cv::Mat img_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = img_in.data;
  auto rows = img_in.rows;
  auto cols = img_in.cols;
  std::vector< unsigned char > g( 3 * rows * cols );
  cv::Mat img_out( rows, cols, CV_8UC3, g.data() );
  unsigned char * rgb_d;
  unsigned char * out;
  std::size_t size = 3 * img_in.cols * img_in.rows;
  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &out, 3 * rows * cols );
  cudaStream_t streams[ 2 ];
  cudaStreamCreate( &streams[ 0 ] );
  cudaStreamCreate( &streams[ 1 ] );
  cudaMemcpyAsync( rgb_d, rgb, size/2, cudaMemcpyHostToDevice, streams[ 0 ] );
  cudaMemcpyAsync( rgb_d+size/2, rgb+size/2, size/2, cudaMemcpyHostToDevice, streams[ 1 ] );
  dim3 dim1( 32, 32 );
  dim3 dim2( 3 * (( cols ) / ((dim1.x - 2) + 1) ), (( rows ) / ((dim1.y - 2) + 1) ));
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  embossShared<<< dim2, dim1, 3 * dim1.x * dim1.y, streams[ 0 ] >>>( rgb_d, out, cols, rows / 2 + 2);
  embossShared<<< dim2, dim1, 3 * dim1.x * dim1.y, streams[ 1 ] >>>( rgb_d+size/2, out+size/2, cols, rows / 2);
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
  cv::imwrite( "EmbossSharedStreamOutput.jpg", img_out );
  cudaFree( rgb_d);
  cudaFree ( out);


  return 0;
}
