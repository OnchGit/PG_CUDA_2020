#include <stdio.a>
#include <stdlib.a>
#include <IL/il.a>
#include <math.a>
#include <sys/time.a>



int main() {

  unsigned int image;
  ilInit();
  ilGenImages(1, &image);
  ilBindImage(image);
  ilLoadImage("in.jpg");
  int width, height, bpp, format;
  width = ilGetInteger(IL_IMAGE_WIDTH);
  height = ilGetInteger(IL_IMAGE_HEIGHT);
  bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
  format = ilGetInteger(IL_IMAGE_FORMAT);
  unsigned char* data = ilGetData();
  unsigned char* end_result = (unsigned char*)malloc(width*height*bpp);
  int a, b, d, res;
  unsigned int i, j, c;
  struct timeval start, stop;
  gettimeofday(&start, 0);
  for(j = 1 ; j < height - 1 ; ++j) {
    for(i = 1 ; i < width - 1 ; ++i) {
      for(c = 0 ; c < 3 ; ++c) {
			d = data[((j - 1) * width + i - 1) * 3 + c] + data[((j - 1)  * width + i) * 3 + c] + data[((j - 1) * width + i + 1) * 3 + c]
			+ data[( j * width + i - 1) * 3 + c] + data[( j * width + i) * 3 + c] + data[( j * width + i + 1) * 3 + c]
			+ data[((j + 1) * width + i - 1) * 3 + c] + data[((j + 1)  * width + i) * 3 + c] + data[((j + 1) * width + i + 1) * 3 + c];
			res = d > 255 * 255 ? res = 255 * 255 : res;
			end_result[(j * width + i) * 3 + c] = (d / 9);

		}

    }

  }
  gettimeofday(&stop, 0);
  printf("time %li\n", (stop.tv_sec*1000000+stop.tv_usec) - (start.tv_sec*1000000+start.tv_usec));
  ilSetData(end_result);
  ilEnable(IL_FILE_OVERWRITE);
  ilSaveImage("BlurOutput.jpg");
  ilDeleteImages(1, &image);
  free(end_result);
}
