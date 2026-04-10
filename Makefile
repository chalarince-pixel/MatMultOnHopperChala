# ============================================================
# CUDA Assignment Makefile
# GPU: NVIDIA H100 (sm_90 / Compute Capability 9.0)
# CUDA: 12.4
#
# Sources live in per-approach subdirectories:
#   01_tiled_shmem/01_tiled_shmem.cu  → build/01_tiled_shmem
#   02_cublas/02_cublas.cu            → build/02_cublas
#   03_wmma/03_wmma.cu                → build/03_wmma
#   04_thread_block_clusters/...      → build/04_thread_block_clusters
#
# Adding a new approach: create a new folder + .cu file.
# The Makefile discovers it automatically.
# ============================================================

NVCC      := nvcc
ARCH      := -arch=sm_90a
INCDIR    := include
BUILDDIR  := build

RELEASE_FLAGS := -O3 -use_fast_math -lineinfo
DEBUG_FLAGS   := -g -G -O0 -lineinfo
COMMON_FLAGS  := -I$(INCDIR) $(ARCH) -Xcompiler -Wall
LDFLAGS       := -lcublas -lcuda   # needed by 02_cublas & driver APIs

# Discover all .cu files one level deep in subdirectories
SRCS := $(wildcard */*.cu)

# All discovered sources are now buildable

# Derive binary names from filenames (strip folder/ and .cu)
BINS := $(patsubst %.cu,%,$(notdir $(SRCS)))

# ── Default: release build of every target ───────────────────
all: $(BINS)

run_all: all
	@for b in $(BINS); do \
	    echo "\n🚀 Running $$b ..."; \
	    ./$(BUILDDIR)/$$b || exit 1; \
	done

# Generate explicit build rules for each target
define MAKE_TARGET
$(2): $(1) | $$(BUILDDIR)
	$$(NVCC) $$(COMMON_FLAGS) $$(RELEASE_FLAGS) $$(LDFLAGS) -o $$(BUILDDIR)/$$@ $$<
	@echo "\n✅  Release build complete: $$(BUILDDIR)/$$@"

debug_$(2): $(1) | $$(BUILDDIR)
	$$(NVCC) $$(COMMON_FLAGS) $$(DEBUG_FLAGS) $$(LDFLAGS) -o $$(BUILDDIR)/$(2)_debug $$<
	@echo "\n✅  Debug build complete: $$(BUILDDIR)/$(2)_debug"
endef

$(foreach src,$(SRCS),$(eval $(call MAKE_TARGET,$(src),$(patsubst %.cu,%,$(notdir $(src))))))

# ── Debug build target  ──────────────────────────────────────
debug: $(addprefix debug_,$(BINS))

# ── Convenience run targets ───────────────────────────────────
$(addprefix run_,$(BINS)): run_%: %
	./$(BUILDDIR)/$*

# ── Memory / race checker ─────────────────────────────────────
sanitize: debug
	@for b in $(BINS); do \
	    echo "\n🔍  Sanitizing $$b ..."; \
	    compute-sanitizer --tool memcheck  $(BUILDDIR)/$${b}_debug; \
	    compute-sanitizer --tool racecheck $(BUILDDIR)/$${b}_debug; \
	done

# ── Nsight Systems profiling ─────────────────────────────────
profile_nsys: all
	@mkdir -p results
	@for b in $(BINS); do \
	    echo "\n📊 CPU/GPU Timeline (nsys) for $$b ..."; \
	    nsys profile --stats=true --force-overwrite true -o results/$${b}_nsys ./$(BUILDDIR)/$$b; \
	done

# ── Nsight Compute profiling ─────────────────────────────────
profile_ncu: all
	@mkdir -p results
	@for b in $(BINS); do \
	    echo "\n🔬 Kernel Detailed Profiling (ncu) for $$b ..."; \
	    ncu --set full -f -o results/$${b}_ncu ./$(BUILDDIR)/$$b; \
	done

# ── Per-target profiling ─────────────────────────────────────
profile_nsys_%: %
	@mkdir -p results
	nsys profile --stats=true --force-overwrite true -o results/$*_nsys ./$(BUILDDIR)/$*

profile_ncu_%: %
	@mkdir -p results
	ncu --set full -f -o results/$*_ncu ./$(BUILDDIR)/$*

# ── Helpers ───────────────────────────────────────────────────
$(BUILDDIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILDDIR) results/*.nsys-rep results/*.sqlite results/*.ncu-rep

.PHONY: all debug sanitize profile_nsys profile_ncu clean \
        $(BINS) \
        $(addprefix run_,$(BINS)) \
        $(addprefix debug_,$(BINS))
