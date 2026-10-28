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
 *   15  VIVVY_ERR_ENCRYPTION_FAILED
 *   16  VIVVY_ERR_DECRYPTION_FAILED
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
#include <stdint.h>

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
#define VIVVY_ERR_ENCRYPTION_FAILED 15
#define VIVVY_ERR_DECRYPTION_FAILED 16

/* Opaque handle to an open store. Only ever accessed through a pointer. */
typedef struct VivvyStore VivvyStore;
typedef struct VivvyIndex VivvyIndex;

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
 * Atomically inserts `count` records. records_json is an array of the same
 * objects accepted by vivvy_store_insert; vectors_flat contains `count`
 * consecutive vectors, each of width dim. On success, out_ids_json receives
 * an owned JSON array of generated IDs.
 */
int vivvy_store_insert_batch(const VivvyStore *store, const char *records_json,
                              const float *vectors_flat, size_t dim, size_t count,
                              char **out_ids_json);

/*
 * Looks up one active record in a scope.
 *
 * scope_json: {"tenant_id": <string>, "namespace": <string>,
 *              "agent_id": <string, optional>, "user_id": <string, optional>}
 *
 * On success, out_record_json receives an owned JSON record, or the JSON
 * literal null if no active record with that ID exists.
 */
int vivvy_store_get(const VivvyStore *store, const char *scope_json,
                    const char *id, char **out_record_json);

/*
 * Atomically updates a record with optimistic concurrency.
 *
 * update_json: {"tenant_id":..., "namespace":..., "id":...,
 *               "expected_revision": <uint>, "content": <string, optional>,
 *               "kind": <string, optional>, "importance": <float, optional>,
 *               "expires_at_ms": <int|null, optional>,
 *               "metadata_patch": <object, optional>, "operation_id": <string, optional>}
 *
 * Pass vector=NULL and dim=0 to retain the current embedding. Otherwise it
 * replaces the embedding. An omitted expires_at_ms retains the old expiry;
 * a JSON null clears it. Returns VIVVY_ERR_REVISION_CONFLICT on stale writes.
 */
int vivvy_store_update(const VivvyStore *store, const char *update_json,
                       const float *vector, size_t dim);

/* Soft-delete one record, immediately hiding it from future reads. */
int vivvy_store_forget(const VivvyStore *store, const char *scope_json,
                       const char *id);

/* Soft-delete a JSON array of IDs in one scope as a single transaction. */
int vivvy_store_forget_batch(const VivvyStore *store, const char *scope_json,
                             const char *ids_json);

/* Returns an owned, privacy-safe JSON StoreHealth snapshot. */
int vivvy_store_health(const VivvyStore *store, char **out_health_json);

/* Rebuilds the in-memory vector accelerator from durable active records. */
int vivvy_store_rebuild_index(const VivvyStore *store);

/*
 * Low-level vector index. config_json: {"dimensions": <uint>,
 * "metric": "l2"|"cosine"|"dot", "wal_path": <string, optional>,
 * "data_dir": <string, optional>}. The optional paths enable durable WAL
 * replay and sealed-segment compaction; omit both for an in-memory index.
 */
int vivvy_index_open(const char *config_json, VivvyIndex **out_index);
void vivvy_index_close(VivvyIndex *index);

/* metadata_json is optional and, when supplied, is an object of string fields
 * and values used by filtered search. out_id receives an auto-generated ID. */
int vivvy_index_insert(const VivvyIndex *index, const float *vector, size_t dim,
                       const char *metadata_json, uint64_t *out_id);
int vivvy_index_insert_with_id(const VivvyIndex *index, uint64_t id,
                               const float *vector, size_t dim);
/* Inserts count consecutive vectors from vectors_flat (each with dim floats)
 * and returns an owned JSON array of their generated uint64 IDs. */
int vivvy_index_insert_batch(const VivvyIndex *index, const float *vectors_flat,
                             size_t dim, size_t count, char **out_ids_json);
int vivvy_index_delete(const VivvyIndex *index, uint64_t id,
                       int *out_deleted);

/*
 * filter_json is optional. Its recursive grammar is one of:
 * {"eq":{"field":"...","value":"..."}},
 * {"in":{"field":"...","values":["..."]}},
 * {"and":[<filter>,...]}, {"or":[<filter>,...]}, or {"not":<filter>}.
 * On success out_results_json receives an owned JSON array of
 * {"id": <uint64>, "distance": <float>} objects.
 */
int vivvy_index_search(const VivvyIndex *index, const float *query, size_t dim,
                       size_t k, const char *filter_json, char **out_results_json);
int vivvy_index_compact(const VivvyIndex *index);

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
 *                "mmr_lambda": <float 0..1, optional, enables MMR diversity reranking>,
 *                "scoring": <{"weight_similarity":.., "weight_importance":.., "weight_recency":.., "weight_reinforcement":.., "recency_half_life_days":..}, optional>}
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
