# vivvy

The unified Rust SDK for Vivvy. It re-exports `vivvy-core`, `vivvy-memory`,
and `vivvy-embedder` behind feature flags. Use the focused crates directly
when an application only needs one service.

```toml
vivvy = "0.1.0-beta.1"
```

Use `default-features = false` plus `features = ["core"]`, `memory`, or
`embedder` to select a smaller surface.
