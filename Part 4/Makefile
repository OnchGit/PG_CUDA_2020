CXX=g++
CXXFLAGS= -g -G -O3 -march=native
LDLIBS=`pkg-config --libs opencv`

sobel: sobelStream.cu
		nvcc -o $@ $< $(LDLIBS)

blur: blurStream.cu
	nvcc -o $@ $< $(LDLIBS)

erode: erodeStream.cu
		nvcc -o $@ $< $(LDLIBS)

dilate: dilateStream.cu
		nvcc -o $@ $< $(LDLIBS)

emboss: embossStream.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm sobel blur emboss erode dilate *.jpg
