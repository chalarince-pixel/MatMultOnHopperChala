# ============================================================
# CUDA Assignment Makefile
# GPU: NVIDIA H100 (sm_90 / Compute Capability 9.0)
# CUDA: 12.4
#
# Each .cu file in src/ is built into its own binary in build/.
#   make              → release build of all targets
#   make debug        → debug build of all targets
#   make 01_tiled_shmem   → build only that target (release)
#   make run_01       → build + run 01_tiled_shmem
# ============================================================

NVCC        := nvcc
ARCH        := -arch=sm_90
SRCDIR      := src
INCDIR      := include
BUILDDIR    := build

# Flags
RELEASE_FLAGS := -O3 -use_fast_math
DEBUG_FLAGS   := -g -G -O0
COMMON_FLAGS  := -I$(INCDIR) $(ARCH) -Xcompiler -Wall

# Discover all source files; derive binary names (strip src/ and .cu)
SRCS    := $(wildcard $(SRCDIR)/*.cu)
TARGETS := $(patsubst $(SRCDIR)/%.cu,%,$(SRCS))

# ── Default: release build of every target ──────────────────
all: $(TARGETS)

# Pattern rule — each .cu → its own binary in build/
$(TARGETS): %: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(COMMON_FLAGS) $(RELEASE_FLAGS) -o $(BUILDDIR)/$@ $<
	@echo "\n✅  Release build complete: $(BUILDDIR)/$@"

# ── Debug build of every target ─────────────────────────────
debug: $(addprefix debug_,$(TARGETS))

debug_%: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(COMMON_FLAGS) $(DEBUG_FLAGS) -o $(BUILDDIR)/$*_debug $<
	@echo "\n✅  Debug build complete: $(BUILDDIR)/$*_debug"

# ── Convenience run targets ──────────────────────────────────
$(addprefix run_,$(TARGETS)): run_%: %
	./$(BUILDDIR)/$*

# ── Memory / race checker ────────────────────────────────────
sanitize: debug
	@for t in $(TARGETS); do \
	    echo "\n🔍  Sanitizing $$t ..."; \
	    compute-sanitizer --tool memcheck  $(BUILDDIR)/$${t}_debug; \
	    compute-sanitizer --tool racecheck $(BUILDDIR)/$${t}_debug; \
	done

# ── Profile with Nsight Systems ─────────────────────────────
profile: all
	@mkdir -p results
	@for t in $(TARGETS); do \
	    echo "\n📊  Profiling $$t ..."; \
	    nsys profile --stats=true -o results/profile_$$t $(BUILDDIR)/$$t; \
	done

# ── Create build dir ─────────────────────────────────────────
$(BUILDDIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILDDIR) results/*.nsys-rep results/*.sqlite

.PHONY: all debug sanitize profile clean \
        $(addprefix run_,$(TARGETS)) \
        $(addprefix debug_,$(TARGETS))
