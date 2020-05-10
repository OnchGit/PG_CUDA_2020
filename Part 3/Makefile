CXX=g++
CXXFLAGS= -g -G -O3 -march=native
LDLIBS=`pkg-config --libs opencv`

sobel: sobelShared.cu
	nvcc -o $@ $< $(LDLIBS)
	
grayscale: grayscaleShared.cu
		nvcc -o $@ $< $(LDLIBS)

blur: blurShared.cu
		nvcc -o $@ $< $(LDLIBS)

emboss: embossShared.cu
		nvcc -o $@ $< $(LDLIBS)

erode: erodeShared.cu
		nvcc -o $@ $< $(LDLIBS)

dilate: dilateShared.cu
		nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm sobel blur emboss erode dilate grayscale *.jpg
