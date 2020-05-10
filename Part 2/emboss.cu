#include <opencv2/opencv.hpp>
#include <vector>

__global__ void emboss ( unsigned char * data,   unsigned char * end_result, std::size_t cols, std::size_t rows) {

  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if ( i > 0 && i < (cols - 1) && j > 0 && j < (rows - 1)) {
    for (auto a = 0; a < 3; ++a){
		  auto op1 = data[((j - 1) * cols + i - 1) * 3 + a] * -18 + data[((j - 1) * cols + i + 1) * 3 + a] * 0
          + data[( j * cols + i - 1) * 3 + a] * -9 + data[( j * cols + i + 1) * 3 + a] * 9
    	  + data[((j + 1) * cols + i - 1) * 3 + a] * 0  + data[((j + 1) * cols + i + 1) * 3 + a] * 18
          + data[(( j - 1) * cols + i) * 3 + a] * -9 +  9 * data[( j* cols + i) * 3 + a]
          + data[(( j + 1) * cols + i) * 3 + a] * 9;
		  end_result[(j * cols + i) * 3 + a] = (op1 / 9);
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
  dim3 dim2(( cols - 1) / dim1.x + 1 , ( rows - 1 ) / dim1.y + 1 );
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );
  emboss<<< dim2, dim1 >>>( rgb_d, end_result, cols, rows );
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
  cv::imwrite( "EmbossOutput.jpg", img_out );
  cudaFree( rgb_d);
  cudaFree ( end_result);
  return 0;
}
