# ============================================================
# CUDA Assignment Makefile
# GPU: NVIDIA H100 (sm_90 / Compute Capability 9.0)
# CUDA: 12.4
# ============================================================

NVCC        := nvcc
ARCH        := -arch=sm_90
SRCDIR      := src
INCDIR      := include
BUILDDIR    := build
TARGET      := cuda_assignment

# Flags
RELEASE_FLAGS := -O3 -use_fast_math
DEBUG_FLAGS   := -g -G -O0           # -G = device debug info, required for cuda-gdb
COMMON_FLAGS  := -I$(INCDIR) $(ARCH) -Xcompiler -Wall

# Source files
SRCS := $(wildcard $(SRCDIR)/*.cu)

# Default: release build
all: release

release: $(SRCS)
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(COMMON_FLAGS) $(RELEASE_FLAGS) -o $(BUILDDIR)/$(TARGET) $(SRCS)
	@echo "\n✅ Release build complete: $(BUILDDIR)/$(TARGET)"

# Debug build (needed for cuda-gdb)
debug: $(SRCS)
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(COMMON_FLAGS) $(DEBUG_FLAGS) -o $(BUILDDIR)/$(TARGET)_debug $(SRCS)
	@echo "\n✅ Debug build complete: $(BUILDDIR)/$(TARGET)_debug"

# Run with memory/race checker
sanitize: debug
	compute-sanitizer --tool memcheck $(BUILDDIR)/$(TARGET)_debug
	compute-sanitizer --tool racecheck $(BUILDDIR)/$(TARGET)_debug

# Profile with Nsight Systems
profile: release
	nsys profile --stats=true -o results/profile $(BUILDDIR)/$(TARGET)

clean:
	rm -rf $(BUILDDIR) results/*.nsys-rep results/*.sqlite

.PHONY: all release debug sanitize profile clean
