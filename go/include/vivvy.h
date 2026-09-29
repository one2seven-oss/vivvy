/*
 * vivvy.h — C99 interface to the Vivvy in-process vector engine & long-term
 * agent memory runtime (vivvy-ffi crate).
 *
 * This header is hand-maintained to stay in sync with ffi/src/store.rs and
 * ffi/src/error.rs. It is the single source of truth consumed by the Go
 * bindings in go/cgo.go; do not regenerate it blindly over local edits.
 *
 * ---------------------------------------------------------------------------
 * Conventions
 * ---------------------------------------------------------------------------
 *
 * Return codes: every vivvy_store_* function returns an int. 0 means success.
 * A nonzero value is one of the codes below; call vivvy_last_error_message()
 * immediately afterward (on the same thread, before any other vivvy_* call)
 * to get a human-readable message.
 *
 * Negative codes are FFI-boundary failures:
 *   -1  VIVVY_ERR_NULL_POINTER    a required pointer argument was NULL
 *   -2  VIVVY_ERR_INVALID_UTF8    a C string argument was not valid UTF-8
 *   -3  VIVVY_ERR_INVALID_JSON    a *_json argument failed to parse/serialize
 *   -4  VIVVY_ERR_PANIC           a Rust panic was caught at the boundary
 *   -5  VIVVY_ERR_IO              a low-level I/O error occurred
 *
 * Positive codes mirror vivvy_memory::ErrorCode one-to-one:
 *    1  VIVVY_ERR_INVALID_SCOPE
 *    2  VIVVY_ERR_DIMENSION_MISMATCH
 *    3  VIVVY_ERR_EMBEDDING_MODEL_MISMATCH
 *    4  VIVVY_ERR_NOT_FOUND
 *    5  VIVVY_ERR_REVISION_CONFLICT
 *    6  VIVVY_ERR_STORE_BUSY
 *    7  VIVVY_ERR_RECOVERY_REQUIRED
 *    8  VIVVY_ERR_CORRUPT_STORE
 *    9  VIVVY_ERR_ENCRYPTION_KEY_UNAVAILABLE
 *   10  VIVVY_ERR_POLICY_DENIED
 *   11  VIVVY_ERR_INVALID_FILTER
 *   12  VIVVY_ERR_INVALID_INPUT
 *   13  VIVVY_ERR_STORE_IO_ERROR
 *   14  VIVVY_ERR_DATABASE_ERROR
 *
 * Strings: any pointer returned through an `out_*` parameter is heap-allocated
 * by Rust and MUST be released with vivvy_free_string() exactly once. The
 * pointer returned by vivvy_last_error_message() is the opposite: it is
 * borrowed from thread-local storage, remains valid only until the next
 * vivvy_* call on the same thread, and must NEVER be passed to
 * vivvy_free_string().
 *
 * Vectors: embeddings cross the boundary as a raw `const float*` plus a
 * `size_t` length — never as JSON — since encoding thousands of floats as
 * JSON numbers would be wasteful on both sides.
 *
 * Thread-safety: a `VivvyStore*` may be shared and used concurrently from
 * multiple threads for vivvy_store_insert/recall/format_context/backup/
 * vacuum — the underlying store guards its own state. vivvy_store_close must
 * be called exactly once, after which the pointer is invalid and must not be
 * used by any other in-flight call.
 */

#ifndef VIVVY_H
#define VIVVY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* FFI-boundary error codes. */
#define VIVVY_ERR_NULL_POINTER (-1)
#define VIVVY_ERR_INVALID_UTF8 (-2)
#define VIVVY_ERR_INVALID_JSON (-3)
#define VIVVY_ERR_PANIC (-4)
#define VIVVY_ERR_IO (-5)

/* Domain error codes, mirroring vivvy_memory::ErrorCode. */
#define VIVVY_ERR_INVALID_SCOPE 1
#define VIVVY_ERR_DIMENSION_MISMATCH 2
#define VIVVY_ERR_EMBEDDING_MODEL_MISMATCH 3
#define VIVVY_ERR_NOT_FOUND 4
#define VIVVY_ERR_REVISION_CONFLICT 5
#define VIVVY_ERR_STORE_BUSY 6
#define VIVVY_ERR_RECOVERY_REQUIRED 7
#define VIVVY_ERR_CORRUPT_STORE 8
#define VIVVY_ERR_ENCRYPTION_KEY_UNAVAILABLE 9
#define VIVVY_ERR_POLICY_DENIED 10
#define VIVVY_ERR_INVALID_FILTER 11
#define VIVVY_ERR_INVALID_INPUT 12
#define VIVVY_ERR_STORE_IO_ERROR 13
#define VIVVY_ERR_DATABASE_ERROR 14

/* Opaque handle to an open store. Only ever accessed through a pointer. */
typedef struct VivvyStore VivvyStore;

/*
 * Opens (or creates) a durable memory store at `path`.
 *
 * config_json: {"dimensions": <uint>, "embedding_model": <string>,
 *               "max_recall_limit": <uint, optional, default 100>}
 *
 * On success, *out_store receives a handle that must eventually be released
 * with vivvy_store_close(). Returns 0 on success, nonzero on failure.
 */
int vivvy_store_open(const char *path, const char *config_json,
                      VivvyStore **out_store);

/*
 * Closes a store opened with vivvy_store_open(), releasing all resources.
 * `store` may be NULL (no-op). After this call the pointer is invalid.
 */
void vivvy_store_close(VivvyStore *store);

/*
 * Inserts a new memory record.
 *
 * record_json: {"tenant_id": <string>, "namespace": <string>,
 *               "agent_id": <string, optional>, "user_id": <string, optional>,
 *               "content": <string>,
 *               "kind": <"fact"|"preference"|"instruction"|"context"|"episodic", optional, default "fact">,
 *               "importance": <float 0..1, optional, default 0.5>,
 *               "expires_at_ms": <int64, optional>,
 *               "operation_id": <string, optional, for idempotent retries>,
 *               "metadata": <object, optional>, "source": <object, optional>}
 *
 * vector/dim: the embedding, supplied separately from record_json.
 *
 * On success, *out_id receives the newly assigned record ID as a
 * NUL-terminated string owned by the caller (free with vivvy_free_string).
 * Returns 0 on success, nonzero on failure.
 */
int vivvy_store_insert(const VivvyStore *store, const char *record_json,
                        const float *vector, size_t dim, char **out_id);

/*
 * Hybrid (vector + optional full-text) recall.
 *
 * options_json: {"tenant_id": <string>, "namespace": <string>,
 *                "agent_id": <string, optional>, "user_id": <string, optional>,
 *                "kinds": <array of kind strings, optional>,
 *                "min_importance": <float, optional>,
 *                "metadata_eq": <object, optional, exact-match filter>,
 *                "created_after_ms": <int64, optional>,
 *                "created_before_ms": <int64, optional>,
 *                "include_explanations": <bool, optional, default true>,
 *                "mmr_lambda": <float 0..1, optional, enables MMR diversity reranking>}
 *
 * query_vector/dim: the query embedding.
 * query_text: optional NUL-terminated string; pass NULL for vector-only search.
 * top_k: maximum number of results to return (must be > 0).
 *
 * On success, *out_results_json receives a NUL-terminated JSON document
 * owned by the caller (free with vivvy_free_string):
 *   {"total_candidates": <uint>,
 *    "items": [{"id":.., "tenant_id":.., "namespace":.., "agent_id":.., "user_id":..,
 *               "kind":.., "content":.., "importance":.., "created_at_ms":..,
 *               "updated_at_ms":.., "expires_at_ms":.., "revision":..,
 *               "metadata":{...}, "source":{...}, "score":..,
 *               "explanation": {"total_score":.., "similarity_score":..,
 *                               "importance_score":.., "recency_score":..,
 *                               "reinforcement_score":.., "policy_notes":[..]} | null
 *              }, ...]}
 * Returns 0 on success, nonzero on failure.
 */
int vivvy_store_recall(const VivvyStore *store, const char *options_json,
                        const float *query_vector, size_t dim,
                        const char *query_text, size_t top_k,
                        char **out_results_json);

/*
 * Runs the same hybrid recall as vivvy_store_recall and formats the results
 * into a token-budgeted string suitable for direct LLM prompt injection.
 *
 * format_json (optional, NULL uses library defaults):
 *   {"max_tokens": <uint>, "template": <string>, "header": <string>, "footer": <string>}
 *
 * On success, *out_context receives the formatted, NUL-terminated string
 * owned by the caller (free with vivvy_free_string).
 * Returns 0 on success, nonzero on failure.
 */
int vivvy_store_format_context(const VivvyStore *store,
                                const char *options_json,
                                const float *query_vector, size_t dim,
                                const char *query_text, size_t top_k,
                                const char *format_json, char **out_context);

/*
 * Creates a zero-downtime, crash-consistent backup of the store at
 * `target_path` (created if it does not exist). Returns 0 on success,
 * nonzero on failure.
 */
int vivvy_store_backup(const VivvyStore *store, const char *target_path);

/*
 * Physically purges up to `batch_size` tombstoned/expired records and
 * rebuilds the vector accelerator if anything was purged. On success,
 * *out_purged receives the number of records removed. Returns 0 on success,
 * nonzero on failure.
 */
int vivvy_store_vacuum(const VivvyStore *store, size_t batch_size,
                        size_t *out_purged);

/*
 * Frees a string previously returned through an out_* parameter of any
 * vivvy_store_* function above. `s` may be NULL (no-op). Never call this on
 * the pointer returned by vivvy_last_error_message() — that one is borrowed.
 */
void vivvy_free_string(char *s);

/*
 * Returns the calling thread's most recent error message, or an empty
 * string if the last call succeeded. The pointer is borrowed and valid only
 * until the next vivvy_* call on this thread; do not free it.
 */
const char *vivvy_last_error_message(void);

#ifdef __cplusplus
}
#endif

#endif /* VIVVY_H */
