CXX=g++
CXXFLAGS= -g -G -O3 -march=native
LDLIBS=`pkg-config --libs opencv`

sobel: sobelSharedStream.cu
		nvcc -o $@ $< $(LDLIBS)

blur: blurSharedStream.cu
	nvcc -o $@ $< $(LDLIBS)

erode: erodeSharedStream.cu
		nvcc -o $@ $< $(LDLIBS)

dilate: dilateSharedStream.cu
		nvcc -o $@ $< $(LDLIBS)

emboss: embossSharedStream.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm sobel blur emboss erode dilate *.jpg
