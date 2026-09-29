# vivvy-go

Idiomatic Go bindings for [Vivvy](../README.md) — an in-process vector
engine and long-term agent memory runtime — via a pre-compiled static C-FFI
library (cgo + staticlib). No Rust toolchain or external daemon is required
to build or run applications that depend on this module; Vivvy runs
in-process, linked directly into your binary.

## Install

```sh
go get github.com/one2seven-oss/vivvy-go
```

`go build`/`go test` need the platform-matching static library present in
`lib/` (see [`lib/README.md`](lib/README.md) — either build it locally with
`make` from the repository root, or download it from a tagged GitHub
Release). Once `lib/` is populated, everything else is a normal Go build:
no Rust toolchain, no `cgo` flags to configure by hand, no running service
to connect to.

## Quick start

```go
package main

import (
	"fmt"
	"log"

	"github.com/one2seven-oss/vivvy-go"
)

func main() {
	store, err := vivvy.Open("./agent_data", vivvy.OpenConfig{
		Dimensions:     3,
		EmbeddingModel: "test-model",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer store.Close()

	id, err := store.Insert(vivvy.MemoryRecord{
		TenantID:  "acme",
		Namespace: "support",
		Content:   "the customer prefers email over phone",
		Kind:      vivvy.KindPreference,
	}, []float32{0.1, 0.2, 0.3})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("stored:", id)

	res, err := store.Recall(vivvy.RecallOptions{
		TenantID:  "acme",
		Namespace: "support",
		QueryText: "how should we contact them?",
		TopK:      5,
	}, []float32{0.1, 0.2, 0.3})
	if err != nil {
		log.Fatal(err)
	}
	for _, item := range res.Items {
		fmt.Printf("[%.2f] %s\n", item.Score, item.Content)
	}
}
```

## API

- `Open(path, OpenConfig) (*Store, error)`
- `(*Store) Insert(MemoryRecord, vector []float32) (id string, error)`
- `(*Store) Recall(RecallOptions, vector []float32) (*RecallResult, error)` — hybrid vector + full-text search
- `(*Store) FormatContext(RecallOptions, vector []float32, FormatOptions) (string, error)` — token-budgeted prompt rendering
- `(*Store) Backup(targetPath string) error` — zero-downtime, crash-consistent copy
- `(*Store) Vacuum(batchSize int) (purged int, error)` — physically purge tombstoned/expired records
- `(*Store) Close() error`

A `*Store` is safe for concurrent use by multiple goroutines — Insert,
Recall, FormatContext, Backup and Vacuum may all run concurrently against
the same Store. Errors are `*vivvy.Error`, carrying a stable numeric `Code`
(see the constants in `vivvy.go`, mirroring `include/vivvy.h`) alongside a
human-readable `Message`; use `errors.As` to recover one.

## Repository layout

- `include/vivvy.h` — the hand-maintained C99 header this module's cgo
  preamble includes; canonical documentation for the underlying C ABI.
- `cgo.go` — per-platform `#cgo LDFLAGS` linking against `lib/`.
- `vivvy.go` — the public Go API and its JSON marshaling to/from the C layer.
- `vivvy_test.go` — the test suite (`go test -v ./...`; add `-race` for the
  concurrency tests, `-short` to skip the multi-second memory-growth smoke
  test).
- `lib/` — pre-compiled static libraries (not committed; see above).

## Building the static library from source

See the top-level [`Makefile`](../Makefile) (`make help`) and
[`.github/workflows/go-ffi.yml`](../.github/workflows/go-ffi.yml). The
library itself lives in [`../ffi`](../ffi), a thin `#[no_mangle] extern "C"`
layer over the `vivvy-memory` Rust crate.
