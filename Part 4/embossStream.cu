#include <opencv2/opencv.hpp>
#include <vector>

__global__ void emboss ( unsigned char * data,   unsigned char * out, std::size_t cols, std::size_t rows) {

  auto op1 = blockIdx.x * (blockDim.x) + threadIdx.x;
  auto op2 = blockIdx.y * (blockDim.y) + threadIdx.y;

  if ( op1 > 0 && op1 < (cols - 1) && op2 > 0 && op2 < (rows - 1)) {

    for (auto c = 0; c < 3; ++c){

      auto op3 = data[((op2 - 1) * cols + op1 - 1) * 3 + c] * -18 + data[((op2 - 1) * cols + op1 + 1) * 3 + c] * 0
      + data[( op2 * cols + op1 - 1) * 3 + c] * -9 + data[( op2 * cols + op1 + 1) * 3 + c] * 9
      + data[((op2 + 1) * cols + op1 - 1) * 3 + c] * 0  + data[((op2 + 1) * cols + op1 + 1) * 3 + c] * 18
      + data[(( op2 - 1) * cols + op1) * 3 + c] * -9 + 9 * data[( op2 * cols + op1) * 3 + c]
      + data[(( op2 + 1) * cols + op1) * 3 + c] * 9;
      out[(op2 * cols + op1) * 3 + c] = (op3 / 9);
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
  emboss<<< dim2, dim1, 0, streams[ 0 ] >>>( rgb_d, out, cols, rows  / 2 + 2);
  emboss<<< dim2, dim1, 0, streams[ 1 ] >>>( rgb_d+size/2, out+size/2, cols, rows / 2);
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
  cv::imwrite( "EmbossStreamOutput.jpg", img_out );
  cudaFree( rgb_d);
  cudaFree ( out);
  return 0;
}
