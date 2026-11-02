# Contributing to Vivvy

Thank you for helping improve Vivvy. All changes enter through a pull request (PR) to `main`; `main` is the only permanent integration branch. Maintainers review, test, approve, and merge every contribution.

The upstream repository intentionally retains only `main` and branches that back active pull requests or active maintainer work. Merged working branches are deleted; their commits and PR history remain available in GitHub.

## Before you start

Fork the repository on GitHub, then clone your fork and add the canonical project as `upstream`:

```bash
git clone git@github.com:<your-account>/vivvy.git
cd vivvy
git remote add upstream git@github.com:one2seven-oss/vivvy.git
git fetch upstream
```

Start each contribution from the current upstream `main`. Do not base new work on another contributor's branch unless a maintainer asks you to.

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
```

## Branch names

Use lowercase kebab-case. Create a branch only for the contribution you are making; a branch identifies both the kind of work and the subsystem it changes.

| Category | Purpose | Domains |
| --- | --- | --- |
| `feat/<domain>/<description>` | New production capability | `ltm`, `embedding`, `security`, `sdk`, `api`, `integrations`, `storage`, `runtime`, `vector-index` |
| `fix/<domain>/<description>` | Defect or regression repair | `ltm`, `storage`, `security`, `vector-index`, `runtime`, `api`, `sdk`, `integrations` |
| `improve/<domain>/<description>` | Non-breaking improvement | `ltm`, `storage`, `vector-index`, `performance`, `architecture`, `sdk`, `runtime`, `api` |
| `experiment/<domain>/<description>` | Research or prototype, not production-ready by default | `retrieval`, `embedding`, `vector-index`, `architecture`, `performance`, `storage` |
| `docs/<description>` or `docs/<domain>/<description>` | Documentation | `architecture`, `api`, `guides` |
| `test/<domain>/<description>` | Test-only work | `ltm`, `storage`, `security`, `vector-index`, `integration` |
| `chore/<description>` or `chore/<domain>/<description>` | Tooling, dependencies, or build maintenance | `dependencies`, `build`, `tooling` |

`recovery/<security|storage|ltm>/<description>` and `review/<description>` are reserved maintainer workflows. Do not use them for ordinary community contributions.

The prefixes above are namespaces, not permanent branches. Never create placeholder branches such as `feat`, `fix`, or `experiment`. Delete your working branch after its PR merges.

Examples:

```bash
git switch -c feat/ltm/hierarchical-recall
git switch -c fix/vector-index/hnsw-search-bug
git switch -c improve/storage/persistence-performance
git switch -c experiment/retrieval/hybrid-search
```

The branch-name check validates this convention for PRs to `main`.

## Develop and test

Keep each branch focused. Do not mix refactors, generated files, or unrelated fixes into a contribution. The repository contains Rust, Python, Go, and TypeScript components; run the checks relevant to the paths you change.

```bash
# Rust workspace: formatting, linting, unit and integration tests
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test --workspace

# Cross-language verification on a Unix-like host
make verify

# Python bindings
python -m pip install maturin numpy pytest
maturin develop --manifest-path py/Cargo.toml
python -m pytest py/tests
```

The CI workflow runs the Python correctness tests while excluding the hardware-sensitive NumPy timing microbenchmark; run that benchmark separately when evaluating performance. CI also runs the tenant-isolation, persistence, and recovery regression suites. Changes under `go/` and `ts/` receive their existing FFI/SDK checks. Update tests and user-facing documentation with behavior changes.

## Commits and pull requests

Use concise, imperative commit subjects with a conventional type where practical, for example `feat: add hierarchical recall`, `fix: preserve tenant isolation`, or `docs: clarify storage recovery`.

Push the branch and open a PR against `main`:

```bash
git add <files>
git commit -m "feat: add hierarchical recall"
git push -u origin HEAD
```

Complete the PR template, describe the problem and implementation, include relevant test results, and keep the PR scoped. Every PR must target `main`, pass required checks, and receive maintainer review and approval. Contributors cannot merge their own PRs or push directly to `main`; only authorized maintainers may merge after validation. Auto-merge is not used.

## Security and experiments

For suspected vulnerabilities, especially tenant-isolation, encryption, storage, or data-loss issues, do not publish exploit details in a public issue or PR. Use the repository's private security-reporting channel or contact a maintainer first. Add regression coverage once a fix can be disclosed.

`experiment/` branches are for evaluation, benchmarks, and prototypes. They never become production code automatically. Record findings, validate correctness and performance, then implement any accepted result cleanly on a new `feat/` or `improve/` branch and submit a normal PR.

## Review and cleanup

Maintainers may request design changes, tests, benchmarks, or documentation before approval. After a PR is merged, delete its working branch from your fork and local checkout. The upstream branch is also removed once it is no longer needed, keeping the branch list focused on active work. Do not delete unmerged work, recovery branches, or active experiments without maintainer approval. Recovery branches remain until their contents have been verified; review branches are temporary.

See [the detailed Git workflow](docs/development/git-workflow.md) for the branch lifecycle and maintainer controls.
