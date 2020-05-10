#include <opencv2/opencv.hpp>
#include <vector>

__global__ void sobelShared ( unsigned char * data,   unsigned char * end_result, std::size_t w, std::size_t h) {

  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;
  auto op1 = threadIdx.x;
  auto op2 = threadIdx.y;
  extern __shared__ unsigned char sharedExt[];

  if( i < w && j < h ) {
    sharedExt[3 * (op2 * blockDim.x + op1) ] = data[ 3 * ( j * w + i ) ];
    sharedExt[3 * (op2 * blockDim.x + op1) + 1 ] = data[ 3 * ( j * w + i ) + 1];
    sharedExt[3 * (op2 * blockDim.x + op1) + 2 ] = data[ 3 * ( j * w + i ) + 2 ];
    __syncthreads();
    auto op3 = blockDim.x;
    if( op1 > 0 && op1 < (blockDim.x - 1) && op2 > 0 && op2 < (blockDim.y - 1) ){
		for (auto a = 0; a < 3; ++a){
			auto hh = sharedExt[ ((op2-1)*op3 + op1 - 1)* 3 + a ] - sharedExt[ ((op2-1)*op3 + op1 + 1) * 3 + a ]
			+ 2 * sharedExt[ (op2*op3 + op1 - 1) * 3 + a ] - 2* sharedExt[ (op2*op3+op1+1) * 3 + a]
			+ sharedExt[ ((op2+1)*op3 + op1 -1) * 3 + a] - sharedExt[ ((op2+1)*op3 +op1 + 1) * 3 + a];
			auto vv = sharedExt[ ((op2-1)*op3 + op1 - 1) * 3 + a ] - sharedExt[ ((op2+1)*op3 + op1 - 1) * 3 + a ]
			+ 2 * sharedExt[ ((op2-1)*op3 + op1) * 3 + a ] - 2* sharedExt[ ((op2+1)*op3+op1) * 3 + a ]
			+ sharedExt[ ((op2-1)*op3 + op1 +1) * 3 + a] - sharedExt[ ((op2+1)*op3 +op1 + 1) * 3 + a];
			auto res = hh * hh + vv * vv;
			res = res > 255*255 ? res = 255*255 : res;
			end_result[ (j * w + i) * 3 + a ] = sqrt( (float)res );

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
  unsigned char * end_result;
  cudaMalloc( &rgb_d, 3 * rows * cols);
  cudaMalloc( &end_result, 3 * rows * cols );
  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  dim3 dim1( 32, 32 );
  dim3 dim2( 3 * (( cols - 1) / (dim1.x-2) + 1) , ( rows - 1 ) / (dim1.y-2) + 1 );
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  sobelShared<<< dim2, dim1, 3*dim1.x*dim1.y >>>( rgb_d, end_result, cols, rows );
  cudaMemcpy(g.data(), end_result, 3 * rows * cols, cudaMemcpyDeviceToHost);
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
  cv::imwrite( "SobelSharedOutput.jpg", img_out );
  cudaFree( rgb_d);
  cudaFree ( end_result);
  return 0;
}
