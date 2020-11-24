EXE_CUDA:=$(basename $(wildcard cuda-*.cu))
EXE:=$(EXE_CUDA)
NVCC=nvcc
NVCFLAGS+=--gpu-architecture compute_30 -D_XOPEN_SOURCE=600
NVLDLIBS+=-lm

ALL: $(EXE)

% : %.cu
	$(NVCC) $(NVCFLAGS) $< $(NVLDLIBS) -o $@

.PHONY: clean

clean:
	\rm -f *~ $(EXE) coupled-oscillators.ppm
