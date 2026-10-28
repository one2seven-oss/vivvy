//! C ABI for the low-level `vivvy-core` vector index.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::AssertUnwindSafe;

use libc::c_int;
use serde::{Deserialize, Serialize};
use vivvy_core::concurrent::VivvyIndex;
use vivvy_core::distance::Metric;
use vivvy_core::filter::FilterExpr;

use crate::error::{run_guarded, FfiError};

/// Opaque owner of a low-level vector index.
pub struct CVivvyIndex {
    inner: VivvyIndex,
}

#[derive(Deserialize)]
struct IndexConfigJson {
    dimensions: usize,
    metric: String,
    #[serde(default)]
    wal_path: Option<String>,
    #[serde(default)]
    data_dir: Option<String>,
}

#[derive(Deserialize)]
struct FilterJson {
    #[serde(default)]
    eq: Option<FilterFieldJson>,
    #[serde(default)]
    r#in: Option<FilterValuesJson>,
    #[serde(default)]
    and: Option<Vec<FilterJson>>,
    #[serde(default)]
    or: Option<Vec<FilterJson>>,
    #[serde(default)]
    not: Option<Box<FilterJson>>,
}

#[derive(Deserialize)]
struct FilterFieldJson {
    field: String,
    value: String,
}

#[derive(Deserialize)]
struct FilterValuesJson {
    field: String,
    values: Vec<String>,
}

#[derive(Serialize)]
struct SearchResultJson {
    id: u64,
    distance: f32,
}

unsafe fn cstr_to_str<'a>(ptr: *const c_char, field: &'static str) -> Result<&'a str, FfiError> {
    if ptr.is_null() {
        return Err(FfiError::NullPointer(field));
    }
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
    Ok(Some(cstr_to_str(ptr, field)?))
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
    Ok(std::slice::from_raw_parts(ptr, len))
}

unsafe fn index_ref<'a>(ptr: *const CVivvyIndex) -> Result<&'a VivvyIndex, FfiError> {
    if ptr.is_null() {
        return Err(FfiError::NullPointer("index"));
    }
    Ok(&(*ptr).inner)
}

fn to_c_string(s: String) -> *mut c_char {
    CString::new(s.replace('\0', ""))
        .expect("NUL bytes stripped above")
        .into_raw()
}

fn invalid_input(message: &str) -> FfiError {
    FfiError::Memory(vivvy_memory::MemoryError::invalid_input(message))
}

fn parse_metric(metric: &str) -> Result<Metric, FfiError> {
    match metric.to_ascii_lowercase().as_str() {
        "l2" => Ok(Metric::L2),
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        _ => Err(invalid_input("metric must be one of: l2, cosine, dot")),
    }
}

fn parse_filter(filter: FilterJson) -> Result<FilterExpr, FfiError> {
    let mut exprs = 0;
    exprs += usize::from(filter.eq.is_some());
    exprs += usize::from(filter.r#in.is_some());
    exprs += usize::from(filter.and.is_some());
    exprs += usize::from(filter.or.is_some());
    exprs += usize::from(filter.not.is_some());
    if exprs != 1 {
        return Err(invalid_input(
            "a filter object must contain exactly one operator",
        ));
    }
    if let Some(eq) = filter.eq {
        return Ok(FilterExpr::Equals {
            field: eq.field,
            value: eq.value,
        });
    }
    if let Some(values) = filter.r#in {
        return Ok(FilterExpr::In {
            field: values.field,
            values: values.values,
        });
    }
    if let Some(and) = filter.and {
        return and
            .into_iter()
            .map(parse_filter)
            .collect::<Result<Vec<_>, _>>()
            .map(FilterExpr::And);
    }
    if let Some(or) = filter.or {
        return or
            .into_iter()
            .map(parse_filter)
            .collect::<Result<Vec<_>, _>>()
            .map(FilterExpr::Or);
    }
    Ok(FilterExpr::Not(Box::new(parse_filter(
        *filter.not.expect("operator count checked"),
    )?)))
}

/// # Safety
/// `config_json` must be NUL-terminated and `out_index` must be writable.
#[no_mangle]
pub unsafe extern "C" fn vivvy_index_open(
    config_json: *const c_char,
    out_index: *mut *mut CVivvyIndex,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(
        || -> Result<*mut CVivvyIndex, FfiError> {
            if out_index.is_null() {
                return Err(FfiError::NullPointer("out_index"));
            }
            let config_str = unsafe { cstr_to_str(config_json, "config_json")? };
            let config: IndexConfigJson =
                serde_json::from_str(config_str).map_err(|source| FfiError::InvalidJson {
                    field: "config_json",
                    source,
                })?;
            if config.dimensions == 0 {
                return Err(invalid_input("dimensions must be greater than zero"));
            }
            let index = VivvyIndex::new(
                config.dimensions,
                parse_metric(&config.metric)?,
                config.wal_path.as_deref(),
                config.data_dir.as_deref(),
            )
            .map_err(vivvy_core::error::VivvyError::Wal)?;
            Ok(Box::into_raw(Box::new(CVivvyIndex { inner: index })))
        },
    ));
    match outcome {
        Ok(index) => {
            unsafe { *out_index = index };
            0
        }
        Err(code) => code,
    }
}

/// Closes a raw index handle.
///
/// # Safety
/// `index` must be null or a live handle returned by `vivvy_index_open`, not
/// concurrently used by another call.
#[no_mangle]
pub unsafe extern "C" fn vivvy_index_close(index: *mut CVivvyIndex) {
    if index.is_null() {
        return;
    }
    let _ = run_guarded(AssertUnwindSafe(|| -> Result<(), FfiError> {
        unsafe { drop(Box::from_raw(index)) };
        Ok(())
    }));
}

/// # Safety
/// `index` must be live; the vector buffer must contain `dim` initialized
/// floats; optional metadata must be NUL-terminated; and `out_id` writable.
#[no_mangle]
pub unsafe extern "C" fn vivvy_index_insert(
    index: *const CVivvyIndex,
    vector: *const f32,
    dim: usize,
    metadata_json: *const c_char,
    out_id: *mut u64,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<u64, FfiError> {
        if out_id.is_null() {
            return Err(FfiError::NullPointer("out_id"));
        }
        let index = unsafe { index_ref(index)? };
        let vector = unsafe { slice_from_raw(vector, dim, "vector")? }.to_vec();
        let metadata = match unsafe { opt_cstr_to_str(metadata_json, "metadata_json")? } {
            Some(json) => {
                serde_json::from_str::<HashMap<String, String>>(json).map_err(|source| {
                    FfiError::InvalidJson {
                        field: "metadata_json",
                        source,
                    }
                })?
            }
            None => HashMap::new(),
        };
        if metadata.is_empty() {
            Ok(index.insert(vector)?)
        } else {
            Ok(index.insert_with_metadata(vector, metadata.into_iter().collect())?)
        }
    }));
    match outcome {
        Ok(id) => {
            unsafe { *out_id = id };
            0
        }
        Err(code) => code,
    }
}

/// # Safety
/// `index` must be live and `vector` must contain `dim` initialized floats.
#[no_mangle]
pub unsafe extern "C" fn vivvy_index_insert_with_id(
    index: *const CVivvyIndex,
    id: u64,
    vector: *const f32,
    dim: usize,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<(), FfiError> {
        unsafe { index_ref(index)? }.insert_with_id(
            id,
            unsafe { slice_from_raw(vector, dim, "vector")? }.to_vec(),
        )?;
        Ok(())
    }));
    match outcome {
        Ok(()) => 0,
        Err(code) => code,
    }
}

/// Inserts `count` consecutive vectors from `vectors_flat` atomically. Each
/// vector has `dim` elements; returned auto-generated IDs are a JSON array.
///
/// # Safety
/// `index` must be live; `vectors_flat` must contain `count * dim` initialized
/// floats; and `out_ids_json` must be writable.
#[no_mangle]
pub unsafe extern "C" fn vivvy_index_insert_batch(
    index: *const CVivvyIndex,
    vectors_flat: *const f32,
    dim: usize,
    count: usize,
    out_ids_json: *mut *mut c_char,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<String, FfiError> {
        if out_ids_json.is_null() {
            return Err(FfiError::NullPointer("out_ids_json"));
        }
        let total = count
            .checked_mul(dim)
            .ok_or_else(|| invalid_input("count * dim overflows usize"))?;
        let flat = unsafe { slice_from_raw(vectors_flat, total, "vectors_flat")? };
        let vectors = (0..count)
            .map(|offset| flat[offset * dim..(offset + 1) * dim].to_vec())
            .collect();
        let ids = unsafe { index_ref(index)? }.insert_batch(vectors)?;
        serde_json::to_string(&ids).map_err(|source| FfiError::InvalidJson {
            field: "out_ids_json",
            source,
        })
    }));
    match outcome {
        Ok(json) => {
            unsafe { *out_ids_json = to_c_string(json) };
            0
        }
        Err(code) => code,
    }
}

/// # Safety
/// `index` must be live and `out_deleted` must be writable.
#[no_mangle]
pub unsafe extern "C" fn vivvy_index_delete(
    index: *const CVivvyIndex,
    id: u64,
    out_deleted: *mut c_int,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<bool, FfiError> {
        if out_deleted.is_null() {
            return Err(FfiError::NullPointer("out_deleted"));
        }
        Ok(unsafe { index_ref(index)? }.delete(id)?)
    }));
    match outcome {
        Ok(deleted) => {
            unsafe { *out_deleted = c_int::from(deleted) };
            0
        }
        Err(code) => code,
    }
}

/// # Safety
/// `index` must be live; `query` must contain `dim` initialized floats;
/// optional filter JSON must be NUL-terminated; and the output pointer writable.
#[no_mangle]
pub unsafe extern "C" fn vivvy_index_search(
    index: *const CVivvyIndex,
    query: *const f32,
    dim: usize,
    k: usize,
    filter_json: *const c_char,
    out_results_json: *mut *mut c_char,
) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<String, FfiError> {
        if out_results_json.is_null() {
            return Err(FfiError::NullPointer("out_results_json"));
        }
        let filter = match unsafe { opt_cstr_to_str(filter_json, "filter_json")? } {
            Some(json) => Some(parse_filter(serde_json::from_str(json).map_err(
                |source| FfiError::InvalidJson {
                    field: "filter_json",
                    source,
                },
            )?)?),
            None => None,
        };
        let hits = unsafe { index_ref(index)? }.search_filtered(
            unsafe { slice_from_raw(query, dim, "query")? },
            k,
            filter.as_ref(),
        )?;
        serde_json::to_string(
            &hits
                .into_iter()
                .map(|(id, distance)| SearchResultJson { id, distance })
                .collect::<Vec<_>>(),
        )
        .map_err(|source| FfiError::InvalidJson {
            field: "out_results_json",
            source,
        })
    }));
    match outcome {
        Ok(json) => {
            unsafe { *out_results_json = to_c_string(json) };
            0
        }
        Err(code) => code,
    }
}

/// # Safety
/// `index` must be a live handle returned by `vivvy_index_open`.
#[no_mangle]
pub unsafe extern "C" fn vivvy_index_compact(index: *const CVivvyIndex) -> c_int {
    let outcome = run_guarded(AssertUnwindSafe(|| -> Result<(), FfiError> {
        unsafe { index_ref(index)? }.compact_now();
        Ok(())
    }));
    match outcome {
        Ok(()) => 0,
        Err(code) => code,
    }
}
