#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <IL/il.h>


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
  unsigned int i, j, c;
  int a, b, d, res;
  int red, green, blue, grey;
  struct timeval start, stop;
  gettimeofday(&start, 0);
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
		  red = data[(i*width + j)*3 + 0];
		  green = data[(i*width + j)*3 + 1];
		  blue = data[(i*width + j)*3 + 2];
		  grey = (red * 307 + green * 604 + blue * 113)/1024;
		  end_result[(i * width + j) * 3 + 0] = grey;
		  end_result[(i * width + j) * 3 + 1] = grey;
		  end_result[(i * width + j) * 3 + 2] = grey;
		}
	}
  gettimeofday(&stop, 0);
  printf("time %li\n", (stop.tv_sec*1000000+stop.tv_usec) - (start.tv_sec*1000000+start.tv_usec));
  ilSetData(out);
  ilEnable(IL_FILE_OVERWRITE);
  ilSaveImage("GSOutput.jpg");
  ilDeleteImages(1, &image);
  free(end_result);

}
