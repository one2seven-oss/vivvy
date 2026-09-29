package vivvy

// Links against the pre-compiled Vivvy static library for the host's
// GOOS/GOARCH, so consumers of this module need neither a Rust toolchain
// nor a running daemon — `go build`/`go test` just work once lib/ is
// populated (see the repository Makefile's `make go-libs`).
//
// Each staticlib bundles its own copy of SQLite (rusqlite's "bundled"
// feature) and the Rust standard library's runtime needs, so every
// platform line below pulls in the same category of system libraries a
// plain C program linking libsqlite3 + libstd would: math, threading,
// dynamic loading on Unix; the handful of Win32 import libraries the Rust
// std toolchain links against on Windows; and the security/CF frameworks
// macOS's libstd backend needs for randomness and TLS bootstrap.

/*
#cgo CFLAGS: -I${SRCDIR}/include

#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib -lvivvy_linux_amd64 -lm -ldl -lpthread
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/lib -lvivvy_linux_arm64 -lm -ldl -lpthread

#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/lib -lvivvy_darwin_amd64 -framework Security -framework CoreFoundation
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/lib -lvivvy_darwin_arm64 -framework Security -framework CoreFoundation

#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/lib -lvivvy_windows_amd64 -lws2_32 -luserenv -lntdll -lbcrypt -ladvapi32 -lkernel32

#include "vivvy.h"
#include <stdlib.h>
*/
import "C"
