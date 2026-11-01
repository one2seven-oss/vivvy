# vivvy-embedder

`vivvy-embedder` defines the provider-neutral `Embedder` trait used by Vivvy
applications. It does not impose an HTTP client, API key strategy, or vendor
dependency. Implement it with the provider client appropriate for your Rust
application, or use the native provider adapters supplied by the Go and
TypeScript SDKs.
