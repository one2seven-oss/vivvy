//! `extern "C"` entry points exposed to Go (and any other C-ABI host).
//!
//! Every function follows the same shape: validate raw arguments, run the real work behind
//! [`run_guarded`] so a Rust panic can never unwind across the FFI boundary, and report
//! failures as a `c_int` code plus a thread-local message retrievable via
//! [`vivvy_last_error_message`]. Strings returned through `out_*` pointers are heap-allocated
//! by Rust and must be released with [`vivvy_free_string`]; anything else (in particular the
//! pointer from `vivvy_last_error_message`) is borrowed and must not be freed.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::AssertUnwindSafe;

use libc::c_int;

use vivvy_memory::{
    ContextFormatOptions, MemoryConfig, MemoryFilter, MemoryScope, MemoryStore, RecallRequest,
    RememberRequest,
};

use crate::error::{run_guarded, FfiError};
use crate::types::{
    parse_kind, ExplanationJson, FormatOptionsJson, InsertRecordJson, OpenConfigJson,
    RecallOptionsJson, RecallResultsJson, ScoredMemoryJson,
};

/// Opaque handle owning a `MemoryStore`. Never constructed or read from C/Go;
/// only ever passed back through the pointer `vivvy_store_open` returned.
pub struct CVivvyStore {
    inner: MemoryStore,
}

// -- argument marshaling helpers ------------------------------------------------------------

unsafe fn cstr_to_str<'a>(ptr: *const c_char, field: &'static str) -> Result<&'a str, FfiError> {
    if ptr.is_null() {
        return Err(FfiError::NullPointer(field));
    }
    // SAFETY: `ptr` is non-null (checked above); the caller's `# Safety` contract requires it
    // to point to a valid NUL-terminated C string for the duration of this call.
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|_| FfiError::InvalidUtf8(field))
}

unsafe fn opt_cstr_to_str<'a>(
    ptr: *const c_char,
    field: &'static str,
) -> Result<Option<&'a str>, FfiError> {
    if ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: same as `cstr_to_str` above; NULL (meaning "absent") is handled before this point.
    CStr::from_ptr(ptr)
        .to_str()
        .map(Some)
        .map_err(|_| FfiError::InvalidUtf8(field))
}

unsafe fn slice_from_raw<'a>(
    ptr: *const f32,
    len: usize,
    field: &'static str,
) -> Result<&'a [f32], FfiError> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(FfiError::NullPointer(field));
    }
    // SAFETY: `len != 0` and `ptr` is non-null (checked above); the caller's `# Safety`
    // contract requires `ptr` to point to at least `len` contiguous, initialized `f32`s.
    Ok(std::slice::from_raw_parts(ptr, len))
}

unsafe fn store_ref<'a>(ptr: *const CVivvyStore) -> Result<&'a MemoryStore, FfiError> {
    if ptr.is_null() {
        return Err(FfiError::NullPointer("store"));
    }
    // SAFETY: `ptr` is non-null (checked above); the caller's `# Safety` contract requires it
    // to be a live pointer from `vivvy_store_open` not yet passed to `vivvy_store_close`.
    Ok(&(*ptr).inner)
}

/// Hands ownership of a new C string to the caller. Interior NULs (which
/// cannot occur in valid JSON/UUIDs but could in principle appear in raw
/// `content`) are stripped rather than panicking or rejecting the call.
fn to_c_string(s: String) -> *mut c_char {
    let sanitized = if s.as_bytes().contains(&0) {
        s.replace('\0', "")
    } else {
        s
    };
    CString::new(sanitized)
        .expect("NUL bytes stripped above")
        .into_raw()
}

fn build_scope(
    tenant_id: &str,
    namespace: &str,
    agent_id: Option<&str>,
    user_id: Option<&str>,
) -> Result<MemoryScope, FfiError> {
    let mut scope = MemoryScope::new(tenant_id, namespace)?;
    if let Some(a) = agent_id {
        scope = scope.with_agent(a)?;
    }
    if let Some(u) = user_id {
        scope = scope.with_user(u)?;
    }
    Ok(scope)
}

fn build_filters(opts: &RecallOptionsJson) -> MemoryFilter {
    MemoryFilter {
        kinds: opts
            .kinds
            .as_ref()
            .map(|ks| ks.iter().map(|s| parse_kind(s)).collect()),
        min_importance: opts.min_importance,
        metadata_eq: opts.metadata_eq.clone(),
        created_after_ms: opts.created_after_ms,
        created_before_ms: opts.created_before_ms,
    }
}

fn scored_memory_json_list(items: &[vivvy_memory::RecallItem]) -> Vec<ScoredMemoryJson> {
    items
        .iter()
        .map(|item| {
            let mut json = ScoredMemoryJson::from(&item.memory);
            json.score = item.score;
            json.explanation = item.explanation.as_ref().map(|e| ExplanationJson {
                total_score: e.total_score,
                similarity_score: e.similarity_score,
                importance_score: e.importance_score,
                recency_score: e.recency_score,
                reinforcement_score: e.reinforcement_score,
                policy_notes: e.policy_notes.clone(),
            });
            json
        })
        .collect()
}

// -- extern "C" API ------------------------------------------------------------------------

/// Opens (or creates) a durable memory store at `path`.
///
/// `config_json` must decode to `{"dimensions": <uint>, "embedding_model": <string>,
/// "max_recall_limit": <uint, optional, default 100>}`. On success, `*out_store` receives an
/// opaque handle that must eventually be released with `vivvy_store_close`.
///
/// Returns 0 on success, or a nonzero code documented in `vivvy.h` on failure — call
/// `vivvy_last_error_message()` for details.
///
/// # Safety
/// `path` and `config_json` must each be NULL or a valid pointer to a NUL-terminated C
/// string. `out_store` must be NULL or a valid, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn vivvy_store_open(
    path: *const c_char,
    config_json: *const c_char,
    out_store: *mut *mut CVivvyStore,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(
        || -> Result<*mut CVivvyStore, FfiError> {
            if out_store.is_null() {
                return Err(FfiError::NullPointer("out_store"));
            }
            // SAFETY: `path`/`config_json` satisfy this function's `# Safety` contract.
            let path_str = unsafe { cstr_to_str(path, "path")? };
            let config_str = unsafe { cstr_to_str(config_json, "config_json")? };
            let cfg: OpenConfigJson =
                serde_json::from_str(config_str).map_err(|source| FfiError::InvalidJson {
                    field: "config_json",
                    source,
                })?;

            let config = MemoryConfig::builder(path_str)
                .dimensions(cfg.dimensions)
                .embedding_model(cfg.embedding_model)
                .max_recall_limit(cfg.max_recall_limit)
                .build()?;

            let store = MemoryStore::open(config)?;
            Ok(Box::into_raw(Box::new(CVivvyStore { inner: store })))
        },
    ));

    match outcome {
        Ok(ptr) => {
            // SAFETY: `out_store` was checked non-null above; the caller's `# Safety`
            // contract requires it to be a valid, writable `*mut *mut CVivvyStore`.
            unsafe { *out_store = ptr };
            0
        }
        Err(code) => code,
    }
}

/// Closes a store opened with `vivvy_store_open`, releasing all resources.
/// `store` may be NULL (no-op). After this call the pointer is invalid and
/// must not be passed to any other `vivvy_*` function.
///
/// # Safety
/// `store` must be NULL or a pointer previously returned by `vivvy_store_open` that has not
/// already been passed to `vivvy_store_close`, and must not be in concurrent use by another
/// thread when this call is made.
#[no_mangle]
pub unsafe extern "C" fn vivvy_store_close(store: *mut CVivvyStore) {
    if store.is_null() {
        return;
    }
    let _ = run_guarded(AssertUnwindSafe(|| -> Result<(), FfiError> {
        // SAFETY: `store` is non-null (checked above) and, per this function's `# Safety`
        // contract, a pointer `vivvy_store_open` produced via `Box::into_raw` that has not
        // already been freed or handed to another concurrent call.
        unsafe { drop(Box::from_raw(store)) };
        Ok(())
    }));
}

/// Inserts a new memory record. `record_json` carries everything except the
/// embedding (scope, content, kind, importance, metadata, ...); `vector`/`dim`
/// carry the embedding. On success, `*out_id` receives the newly assigned
/// record ID as an owned string (free with `vivvy_free_string`).
///
/// # Safety
/// `store` must be a valid pointer from `vivvy_store_open`. `record_json` must be NULL or a
/// valid NUL-terminated C string. `vector` must point to at least `dim` contiguous `f32`s (or
/// be NULL if `dim` is 0). `out_id` must be NULL or a valid, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn vivvy_store_insert(
    store: *const CVivvyStore,
    record_json: *const c_char,
    vector: *const f32,
    dim: usize,
    out_id: *mut *mut c_char,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<String, FfiError> {
        if out_id.is_null() {
            return Err(FfiError::NullPointer("out_id"));
        }
        // SAFETY: `store`, `record_json`, and `vector`/`dim` satisfy this function's
        // `# Safety` contract.
        let store_ref = unsafe { store_ref(store)? };
        let record_str = unsafe { cstr_to_str(record_json, "record_json")? };
        let rec: InsertRecordJson =
            serde_json::from_str(record_str).map_err(|source| FfiError::InvalidJson {
                field: "record_json",
                source,
            })?;
        let embedding = unsafe { slice_from_raw(vector, dim, "vector")? }.to_vec();

        let scope = build_scope(
            &rec.tenant_id,
            &rec.namespace,
            rec.agent_id.as_deref(),
            rec.user_id.as_deref(),
        )?;

        let req = RememberRequest {
            operation_id: rec.operation_id,
            scope,
            content: rec.content,
            embedding,
            kind: parse_kind(&rec.kind),
            importance: rec.importance,
            expires_at_ms: rec.expires_at_ms,
            metadata: rec.metadata,
            source: rec.source,
        };

        Ok(store_ref.remember(req)?)
    }));

    match outcome {
        Ok(id) => {
            // SAFETY: `out_id` was checked non-null above (mirrors `out_store` in
            // `vivvy_store_open`).
            unsafe { *out_id = to_c_string(id) };
            0
        }
        Err(code) => code,
    }
}

/// Hybrid (vector + optional FTS) recall. `options_json` carries scope,
/// filters, and scoring knobs — see `vivvy.h` for its shape. `query_text` may
/// be NULL to search by vector only. On success, `*out_results_json` receives
/// a `{"total_candidates": ..., "items": [...]}` document (free with
/// `vivvy_free_string`).
///
/// # Safety
/// `store` must be a valid pointer from `vivvy_store_open`. `options_json` must be NULL or a
/// valid NUL-terminated C string. `query_vector` must point to at least `dim` contiguous
/// `f32`s (or be NULL if `dim` is 0). `query_text` must be NULL or a valid NUL-terminated C
/// string. `out_results_json` must be NULL or a valid, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn vivvy_store_recall(
    store: *const CVivvyStore,
    options_json: *const c_char,
    query_vector: *const f32,
    dim: usize,
    query_text: *const c_char,
    top_k: usize,
    out_results_json: *mut *mut c_char,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<String, FfiError> {
        if out_results_json.is_null() {
            return Err(FfiError::NullPointer("out_results_json"));
        }
        // SAFETY: `store`, `options_json`, `query_vector`/`dim`, and `query_text` satisfy
        // this function's `# Safety` contract.
        let store_ref = unsafe { store_ref(store)? };
        let options_str = unsafe { cstr_to_str(options_json, "options_json")? };
        let opts: RecallOptionsJson =
            serde_json::from_str(options_str).map_err(|source| FfiError::InvalidJson {
                field: "options_json",
                source,
            })?;
        let query_embedding =
            unsafe { slice_from_raw(query_vector, dim, "query_vector")? }.to_vec();
        let query_text_opt =
            unsafe { opt_cstr_to_str(query_text, "query_text")? }.map(String::from);

        let scope = build_scope(
            &opts.tenant_id,
            &opts.namespace,
            opts.agent_id.as_deref(),
            opts.user_id.as_deref(),
        )?;
        let filters = build_filters(&opts);

        let req = RecallRequest {
            scope,
            query_embedding,
            query_text: query_text_opt,
            limit: top_k,
            filters,
            include_explanations: opts.include_explanations,
            mmr_lambda: opts.mmr_lambda,
        };

        let resp = store_ref.recall(req)?;
        let out = RecallResultsJson {
            total_candidates: resp.total_candidates,
            items: scored_memory_json_list(&resp.items),
        };
        serde_json::to_string(&out).map_err(|source| FfiError::InvalidJson {
            field: "out_results_json",
            source,
        })
    }));

    match outcome {
        Ok(json) => {
            // SAFETY: `out_results_json` was checked non-null above.
            unsafe { *out_results_json = to_c_string(json) };
            0
        }
        Err(code) => code,
    }
}

/// Runs the same hybrid recall as `vivvy_store_recall` and formats the
/// results into a token-budgeted string suitable for direct LLM prompt
/// injection. `format_json` is optional (NULL uses library defaults) and may
/// carry `{"max_tokens":.., "template":.., "header":.., "footer":..}`.
///
/// # Safety
/// `store` must be a valid pointer from `vivvy_store_open`. `options_json` must be NULL or a
/// valid NUL-terminated C string. `query_vector` must point to at least `dim` contiguous
/// `f32`s (or be NULL if `dim` is 0). `query_text` and `format_json` must each be NULL or a
/// valid NUL-terminated C string. `out_context` must be NULL or a valid, writable pointer.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn vivvy_store_format_context(
    store: *const CVivvyStore,
    options_json: *const c_char,
    query_vector: *const f32,
    dim: usize,
    query_text: *const c_char,
    top_k: usize,
    format_json: *const c_char,
    out_context: *mut *mut c_char,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<String, FfiError> {
        if out_context.is_null() {
            return Err(FfiError::NullPointer("out_context"));
        }
        // SAFETY: `store`, `options_json`, `query_vector`/`dim`, `query_text`, and
        // `format_json` satisfy this function's `# Safety` contract.
        let store_ref = unsafe { store_ref(store)? };
        let options_str = unsafe { cstr_to_str(options_json, "options_json")? };
        let opts: RecallOptionsJson =
            serde_json::from_str(options_str).map_err(|source| FfiError::InvalidJson {
                field: "options_json",
                source,
            })?;
        let query_embedding =
            unsafe { slice_from_raw(query_vector, dim, "query_vector")? }.to_vec();
        let query_text_opt =
            unsafe { opt_cstr_to_str(query_text, "query_text")? }.map(String::from);
        let format_str = unsafe { opt_cstr_to_str(format_json, "format_json")? };

        let scope = build_scope(
            &opts.tenant_id,
            &opts.namespace,
            opts.agent_id.as_deref(),
            opts.user_id.as_deref(),
        )?;
        let filters = build_filters(&opts);

        let req = RecallRequest {
            scope,
            query_embedding,
            query_text: query_text_opt,
            limit: top_k,
            filters,
            include_explanations: false,
            mmr_lambda: opts.mmr_lambda,
        };

        let format_opts = match format_str {
            Some(s) => {
                let parsed: FormatOptionsJson =
                    serde_json::from_str(s).map_err(|source| FfiError::InvalidJson {
                        field: "format_json",
                        source,
                    })?;
                let default = ContextFormatOptions::default();
                ContextFormatOptions {
                    max_tokens: parsed.max_tokens.unwrap_or(default.max_tokens),
                    template: parsed.template.unwrap_or(default.template),
                    header: parsed.header.or(default.header),
                    footer: parsed.footer.or(default.footer),
                }
            }
            None => ContextFormatOptions::default(),
        };

        Ok(store_ref.format_context(req, format_opts)?)
    }));

    match outcome {
        Ok(text) => {
            // SAFETY: `out_context` was checked non-null above.
            unsafe { *out_context = to_c_string(text) };
            0
        }
        Err(code) => code,
    }
}

/// Creates a zero-downtime, crash-consistent backup of the store at
/// `target_path` (created if it does not exist).
///
/// # Safety
/// `store` must be a valid pointer from `vivvy_store_open`. `target_path` must be NULL or a
/// valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn vivvy_store_backup(
    store: *const CVivvyStore,
    target_path: *const c_char,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<(), FfiError> {
        // SAFETY: `store` and `target_path` satisfy this function's `# Safety` contract.
        let store_ref = unsafe { store_ref(store)? };
        let path = unsafe { cstr_to_str(target_path, "target_path")? };
        store_ref.backup(path)?;
        Ok(())
    }));

    match outcome {
        Ok(()) => 0,
        Err(code) => code,
    }
}

/// Physically purges up to `batch_size` tombstoned/expired records and
/// rebuilds the vector accelerator if anything was purged. On success,
/// `*out_purged` receives the number of records removed.
///
/// # Safety
/// `store` must be a valid pointer from `vivvy_store_open`. `out_purged` must be NULL or a
/// valid, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn vivvy_store_vacuum(
    store: *const CVivvyStore,
    batch_size: usize,
    out_purged: *mut usize,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<usize, FfiError> {
        if out_purged.is_null() {
            return Err(FfiError::NullPointer("out_purged"));
        }
        // SAFETY: `store` satisfies this function's `# Safety` contract.
        let store_ref = unsafe { store_ref(store)? };
        Ok(store_ref.vacuum_tombstones(batch_size)?)
    }));

    match outcome {
        Ok(purged) => {
            // SAFETY: `out_purged` was checked non-null above.
            unsafe { *out_purged = purged };
            0
        }
        Err(code) => code,
    }
}

/// Frees a string previously returned through an `out_*` parameter by any
/// `vivvy_store_*` function. `s` may be NULL (no-op). Never call this on the
/// pointer returned by `vivvy_last_error_message` — that one is borrowed.
///
/// # Safety
/// `s` must be NULL or a pointer previously returned through an `out_*` parameter of a
/// `vivvy_store_*` function, not already freed.
#[no_mangle]
pub unsafe extern "C" fn vivvy_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: `s` is non-null (checked above) and, per this function's `# Safety` contract,
    // a pointer this crate produced via `CString::into_raw` that has not already been freed.
    unsafe { drop(CString::from_raw(s)) };
}

/// Returns the calling thread's most recent error message, or an empty
/// string if the last call succeeded. The pointer is borrowed and valid only
/// until the next `vivvy_*` call on this thread; do not free it.
#[no_mangle]
pub extern "C" fn vivvy_last_error_message() -> *const c_char {
    crate::error::last_error_ptr()
}
