# Vivvy Git workflow

`main` is Vivvy's only permanent integration branch. Every feature, fix, improvement, documentation change, test change, maintenance task, and experiment is developed on a short-lived categorized branch and proposed to `main` in a pull request.

## Architecture

The following are naming namespaces, not long-lived parent branches:

```text
feat/{ltm,embedding,security,sdk,api,integrations,storage,runtime,vector-index}/<description>
fix/{ltm,storage,security,vector-index,runtime,api,sdk,integrations}/<description>
improve/{ltm,storage,vector-index,performance,architecture,sdk,runtime,api}/<description>
experiment/{retrieval,embedding,vector-index,architecture,performance,storage}/<description>
docs/[architecture|api|guides/]<description>
test/{ltm,storage,security,vector-index,integration}/<description>
chore/[dependencies|build|tooling/]<description>
recovery/{security,storage,ltm}/<description>     # maintainers only
review/<description>                              # maintainers only
```

Use lowercase kebab-case for every description. Examples: `feat/embedding/new-adapter`, `fix/runtime/op-id-leak`, `improve/vector-index/batch-upsert`, and `experiment/retrieval/hybrid-search`.

## Contributor flow

```text
Create from main -> develop -> test -> push branch -> PR to main
                                                    |
                                      CI -> maintainer review -> approval -> merge
```

```bash
git switch main
git pull --ff-only origin main
git switch -c fix/vector-index/hnsw-search-bug
# implement and test
git add <files>
git commit -m "fix: repair HNSW search"
git push -u origin HEAD
```

Open the PR against `main`, not another working branch. CI checks the branch name, Rust formatting/lint/tests, critical security and persistence regressions, Python bindings, and the existing language-binding workflows where their paths apply.

## Maintainer controls

The required repository configuration is documented in `CONTRIBUTING.md` and should ensure that direct pushes to `main`, self-approval, self-merge, bypassing required checks, and automatic merging are prohibited. Security-sensitive changes receive deliberate maintainer review. Experimental code must be promoted through a clean `feat/` or `improve/` PR after its findings are documented and validated.

## Lifecycle and cleanup

```text
main -> categorized working branch -> pull request -> CI and review -> maintainer merge -> delete merged working branch
```

Delete branches only after their PR is merged and the branch is no longer needed. Keep unmerged work, active experiments, and recovery branches. A maintainer decides when to archive or delete experiments and recovery branches.
