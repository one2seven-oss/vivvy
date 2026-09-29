# lib/

This directory holds the pre-compiled `vivvy-ffi` static libraries that
`go/cgo.go` links against — one per supported platform:

```
libvivvy_linux_amd64.a
libvivvy_linux_arm64.a
libvivvy_darwin_amd64.a
libvivvy_darwin_arm64.a
libvivvy_windows_amd64.a
```

They are **not committed to git** (see `.gitignore`) — each is tens of
megabytes and is fully reproducible from source. Populate this directory one
of two ways:

1. **Build locally** (requires a Rust toolchain): from the repository root,
   run `make native` to build only your machine's own platform, or
   `make linux-amd64` / `make darwin-arm64` / etc. for a specific one. See
   the top-level `Makefile` for the full list and their cross-compilation
   requirements.
2. **Download a release build**: grab the `.a` file for your platform from
   the repository's GitHub Releases page (published by
   `.github/workflows/go-ffi.yml` on tagged releases) and drop it in here.

Once populated, `go build ./...` / `go test ./...` from the `go/` directory
work with no further setup — no Rust toolchain needed at that point.
