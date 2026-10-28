# Releasing Vivvy `0.1.0-beta.1`

This repository releases a unified SDK and focused Rust crates at the same
version. Publishing is deliberately a human-authorized action; these steps do
not run automatically from CI.

## Public release surface

| Name | Registry / ecosystem | Audience |
| --- | --- | --- |
| `vivvy` | crates.io, PyPI, npm | Unified batteries-included API |
| `vivvy-core` | crates.io | Low-level vector search |
| `vivvy-memory` | crates.io | Durable agent memory |
| `vivvy-embedder` | crates.io | Provider-neutral Rust embedding trait |
| `github.com/one2seven-oss/vivvy/go` | Go module / Git tag | Unified Go API |

`vivvy-ffi` is an internal implementation artifact and is not published as a
standalone developer package.

## Required checks

```sh
make verify
cargo clippy --workspace -- -D warnings
```

Run package-specific publish dry runs immediately before each upload, after
that package's newly-versioned registry dependencies are available:

```sh
cargo publish -p vivvy-core --dry-run
cargo publish -p vivvy-embedder --dry-run
```

## Publication order

Publish Rust crates in dependency order, waiting for each version to become
available on crates.io before moving to the next:

1. `vivvy-core`
2. `vivvy-embedder`
3. `vivvy-memory`
4. `vivvy`

Before steps 3 and 4, wait for crates.io to index the preceding packages,
then run that package's own `cargo publish --dry-run`. A local dry run cannot
resolve a new path dependency at its release version until it is available in
the registry; this is expected, not a packaging defect.

Then publish the language SDKs at their ecosystem-equivalent beta version:

- PyPI: `vivvy==0.1.0b1`
- npm: `vivvy@0.1.0-beta.1`
- Go: tag `v0.1.0-beta.1` and publish the release; the existing release
  workflow attaches native static libraries for supported platforms.

Do not publish any package until its manifest, changelog/release notes, and
public API documentation describe the same release surface.
