# Build automation for the Vivvy Go bindings (go/): compiles the vivvy-ffi
# Rust crate into the pre-compiled static libraries go/cgo.go links against,
# so `go get` / `go build` / `go test` never require a Rust toolchain.
#
# Usage:
#   make native        # build only the host's own GOOS/GOARCH into go/lib/
#   make linux-amd64    # cross/native-build one specific target
#   make go-test         # go test -v ./... against whatever is in go/lib/
#   make help            # list all targets
#
# Each per-platform target requires the matching Rust std target and a
# working C linker/SDK for it to be installed:
#   linux-amd64    x86_64-unknown-linux-gnu   — usually present out of the box
#   linux-arm64    aarch64-unknown-linux-gnu  — needs an aarch64 cross gcc
#                                                (e.g. `apt install gcc-aarch64-linux-gnu`)
#   darwin-amd64   x86_64-apple-darwin        — needs Xcode/macOS SDK (build on macOS)
#   darwin-arm64   aarch64-apple-darwin       — needs Xcode/macOS SDK (build on macOS)
#   windows-amd64  x86_64-pc-windows-gnu      — needs mingw-w64 (crossable from Linux/macOS);
#                                                the CI workflow instead builds
#                                                x86_64-pc-windows-msvc natively on a
#                                                windows-latest runner, which is the
#                                                better-supported production target.
# This Makefile targets Unix-like hosts (Linux/macOS) with `make`, `install`
# and `cargo` on PATH. The Windows CI job builds directly with cargo/pwsh
# instead of invoking this file — see .github/workflows/go-bindings.yml.

CARGO ?= cargo
GO ?= go
FFI_CRATE := vivvy-ffi
FFI_LIB_NAME := vivvy_ffi
GO_DIR := go
GO_LIB_DIR := go/lib

.PHONY: all help native go-libs go-build go-test go-test-race go-vet go-fmt-check \
        header leak-check clean-libs \
        linux-amd64 linux-arm64 darwin-amd64 darwin-arm64 windows-amd64

all: native go-build go-test

help:
	@echo "Targets:"
	@echo "  make native          Build the FFI staticlib for the host's own GOOS/GOARCH into go/lib/"
	@echo "  make linux-amd64     Build target x86_64-unknown-linux-gnu -> go/lib/libvivvy_linux_amd64.a"
	@echo "  make linux-arm64     Build target aarch64-unknown-linux-gnu -> go/lib/libvivvy_linux_arm64.a"
	@echo "  make darwin-amd64    Build target x86_64-apple-darwin -> go/lib/libvivvy_darwin_amd64.a"
	@echo "  make darwin-arm64    Build target aarch64-apple-darwin -> go/lib/libvivvy_darwin_arm64.a"
	@echo "  make windows-amd64   Build target x86_64-pc-windows-gnu -> go/lib/libvivvy_windows_amd64.a"
	@echo "  make go-build        cd go && go build ./..."
	@echo "  make go-vet          cd go && go vet ./..."
	@echo "  make go-fmt-check    Fail if any go/*.go file is not gofmt-clean"
	@echo "  make go-test         cd go && go test -v ./..."
	@echo "  make go-test-race    cd go && go test -v -race ./..."
	@echo "  make leak-check      Run the go test binary under valgrind (Linux, advisory)"
	@echo "  make clean-libs      Remove go/lib/*.a"

# -- one Rust staticlib per (GOOS, GOARCH) go/cgo.go's #cgo lines expect ---------------------

linux-amd64:
	rustup target add x86_64-unknown-linux-gnu 2>/dev/null || true
	$(CARGO) build -p $(FFI_CRATE) --release --target x86_64-unknown-linux-gnu
	install -m644 target/x86_64-unknown-linux-gnu/release/lib$(FFI_LIB_NAME).a $(GO_LIB_DIR)/libvivvy_linux_amd64.a

linux-arm64:
	rustup target add aarch64-unknown-linux-gnu 2>/dev/null || true
	$(CARGO) build -p $(FFI_CRATE) --release --target aarch64-unknown-linux-gnu
	install -m644 target/aarch64-unknown-linux-gnu/release/lib$(FFI_LIB_NAME).a $(GO_LIB_DIR)/libvivvy_linux_arm64.a

darwin-amd64:
	rustup target add x86_64-apple-darwin 2>/dev/null || true
	$(CARGO) build -p $(FFI_CRATE) --release --target x86_64-apple-darwin
	install -m644 target/x86_64-apple-darwin/release/lib$(FFI_LIB_NAME).a $(GO_LIB_DIR)/libvivvy_darwin_amd64.a

darwin-arm64:
	rustup target add aarch64-apple-darwin 2>/dev/null || true
	$(CARGO) build -p $(FFI_CRATE) --release --target aarch64-apple-darwin
	install -m644 target/aarch64-apple-darwin/release/lib$(FFI_LIB_NAME).a $(GO_LIB_DIR)/libvivvy_darwin_arm64.a

windows-amd64:
	rustup target add x86_64-pc-windows-gnu 2>/dev/null || true
	$(CARGO) build -p $(FFI_CRATE) --release --target x86_64-pc-windows-gnu
	install -m644 target/x86_64-pc-windows-gnu/release/lib$(FFI_LIB_NAME).a $(GO_LIB_DIR)/libvivvy_windows_amd64.a

# Builds only the static lib matching this machine's own GOOS/GOARCH — the
# fast path for local development and for CI's per-OS native jobs.
native:
	@os=$$($(GO) env GOOS); arch=$$($(GO) env GOARCH); \
	target="$$os-$$arch"; \
	echo "==> building $$target (host GOOS/GOARCH)"; \
	$(MAKE) $$target

# -- Go side -----------------------------------------------------------------------------------

go-build:
	cd $(GO_DIR) && CGO_ENABLED=1 $(GO) build ./...

go-vet:
	cd $(GO_DIR) && CGO_ENABLED=1 $(GO) vet ./...

go-fmt-check:
	@unformatted=$$(gofmt -l $(GO_DIR)); \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt needed on:"; echo "$$unformatted"; exit 1; \
	fi

go-test:
	cd $(GO_DIR) && CGO_ENABLED=1 $(GO) test -v ./...

go-test-race:
	cd $(GO_DIR) && CGO_ENABLED=1 $(GO) test -v -race ./...

# Advisory only: the Go runtime's own arenas/goroutine stacks show up as
# "still reachable" noise under valgrind, so this is meant to be read by a
# human looking for genuine "definitely lost" blocks from the FFI layer, not
# wired into CI as a hard gate without first tuning a suppressions file.
leak-check:
	cd $(GO_DIR) && CGO_ENABLED=1 $(GO) test -c -o vivvy.test .
	cd $(GO_DIR) && valgrind --leak-check=full --show-leak-kinds=definite ./vivvy.test -test.v
	rm -f $(GO_DIR)/vivvy.test

# The checked-in go/include/vivvy.h is hand-maintained (see its own header
# comment) to stay exactly in sync with ffi/src/store.rs. If cbindgen is
# installed, this target regenerates a candidate header for comparison —
# diff it against the real one by hand before ever replacing it.
header:
	cd ffi && cbindgen --config cbindgen.toml --output /tmp/vivvy-cbindgen-candidate.h
	@echo "Candidate header written to /tmp/vivvy-cbindgen-candidate.h — diff against go/include/vivvy.h before using it."

clean-libs:
	rm -f $(GO_LIB_DIR)/*.a
