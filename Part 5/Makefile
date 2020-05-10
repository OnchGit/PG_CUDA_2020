CXX=g++
CXXFLAGS= -g -G -O3 -march=native
LDLIBS=`pkg-config --libs opencv`

grayscaleShared: grayscaleSharedStream.cu
	nvcc -o $@ $< $(LDLIBS)

sobelShared: sobelSharedStream.cu
		nvcc -o $@ $< $(LDLIBS)

blurShared: blurSharedStream.cu
	nvcc -o $@ $< $(LDLIBS)

erodeShared: erodeSharedStream.cu
		nvcc -o $@ $< $(LDLIBS)

dilateShared: dilateSharedStream.cu
		nvcc -o $@ $< $(LDLIBS)

embossShared: embossSharedStream.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm sobelShared blurShared embossShared erodeShared dilateShared grayscaleShared *.jpg
