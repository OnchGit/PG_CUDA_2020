CXX=g++
CXXFLAGS= -g -G -O3 -march=native
LDLIBS=`pkg-config --libs opencv`

sobelShared: sobelShared.cu
	nvcc -o $@ $< $(LDLIBS)


grayscaleShared: grayscaleShared.cu
		nvcc -o $@ $< $(LDLIBS)

blurShared: blurShared.cu
		nvcc -o $@ $< $(LDLIBS)

embossShared: embossShared.cu
		nvcc -o $@ $< $(LDLIBS)

erodeShared: erodeShared.cu
		nvcc -o $@ $< $(LDLIBS)

dilateShared: dilateShared.cu
		nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm sobelShared blurShared embossShared erodeShared dilateShared grayscaleShared *.jpg
