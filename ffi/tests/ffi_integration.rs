//! Exercises the `unsafe extern "C"` boundary directly, the same way a C or
//! Go caller would, without going through cgo. This catches boundary bugs
//! (wrong error codes, bad null handling, leaked/double-freed strings)
//! closer to the source than the Go test suite in `go/vivvy_test.go` can,
//! and runs as part of the ordinary `cargo test --workspace`.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use vivvy_ffi::index::{
    vivvy_index_close, vivvy_index_delete, vivvy_index_insert, vivvy_index_insert_batch,
    vivvy_index_open, vivvy_index_search, CVivvyIndex,
};
use vivvy_ffi::store::{
    vivvy_free_string, vivvy_last_error_message, vivvy_store_close, vivvy_store_forget_batch,
    vivvy_store_get, vivvy_store_health, vivvy_store_insert, vivvy_store_insert_batch,
    vivvy_store_open, vivvy_store_rebuild_index, vivvy_store_recall, vivvy_store_update,
    vivvy_store_vacuum, CVivvyStore,
};

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

unsafe fn last_error() -> String {
    CStr::from_ptr(vivvy_last_error_message())
        .to_string_lossy()
        .into_owned()
}

/// Opens a store in a fresh temp directory. Returns the handle and the
/// `TempDir` guard, which must be kept alive for the duration of the test.
unsafe fn open_test_store(dims: usize) -> (*mut CVivvyStore, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = cstr(dir.path().to_str().unwrap());
    let config = cstr(&format!(
        r#"{{"dimensions":{dims},"embedding_model":"test-model"}}"#
    ));

    let mut handle: *mut CVivvyStore = ptr::null_mut();
    let code = vivvy_store_open(path.as_ptr(), config.as_ptr(), &mut handle);
    assert_eq!(code, 0, "vivvy_store_open failed: {}", last_error());
    assert!(!handle.is_null());
    (handle, dir)
}

#[test]
fn open_insert_recall_close_roundtrip() {
    unsafe {
        let (store, _dir) = open_test_store(3);

        let record = cstr(r#"{"tenant_id":"acme","namespace":"default","content":"hello world"}"#);
        let vector: [f32; 3] = [1.0, 0.0, 0.0];
        let mut out_id: *mut c_char = ptr::null_mut();

        let code = vivvy_store_insert(
            store,
            record.as_ptr(),
            vector.as_ptr(),
            vector.len(),
            &mut out_id,
        );
        assert_eq!(code, 0, "insert failed: {}", last_error());
        assert!(!out_id.is_null());
        let id = CStr::from_ptr(out_id).to_string_lossy().into_owned();
        assert!(!id.is_empty());
        vivvy_free_string(out_id);

        let options = cstr(r#"{"tenant_id":"acme","namespace":"default"}"#);
        let mut out_results: *mut c_char = ptr::null_mut();
        let code = vivvy_store_recall(
            store,
            options.as_ptr(),
            vector.as_ptr(),
            vector.len(),
            ptr::null(), // no query_text: vector-only search
            5,
            &mut out_results,
        );
        assert_eq!(code, 0, "recall failed: {}", last_error());
        assert!(!out_results.is_null());
        let results = CStr::from_ptr(out_results).to_string_lossy().into_owned();
        assert!(
            results.contains("hello world"),
            "unexpected results: {results}"
        );
        assert!(results.contains(&id), "unexpected results: {results}");
        vivvy_free_string(out_results);

        vivvy_store_close(store);
    }
}

#[test]
fn dimension_mismatch_reports_the_documented_positive_code() {
    const ERR_DIMENSION_MISMATCH: i32 = 2; // must match vivvy.h / ffi::error

    unsafe {
        let (store, _dir) = open_test_store(3);

        let record = cstr(r#"{"tenant_id":"acme","namespace":"default","content":"x"}"#);
        let wrong_dim_vector: [f32; 2] = [1.0, 2.0];
        let mut out_id: *mut c_char = ptr::null_mut();

        let code = vivvy_store_insert(
            store,
            record.as_ptr(),
            wrong_dim_vector.as_ptr(),
            wrong_dim_vector.len(),
            &mut out_id,
        );
        assert_eq!(code, ERR_DIMENSION_MISMATCH);
        assert!(out_id.is_null(), "out_id must not be written on failure");
        assert!(!last_error().is_empty());

        vivvy_store_close(store);
    }
}

#[test]
fn null_required_pointer_reports_null_pointer_error() {
    const ERR_NULL_POINTER: i32 = -1;

    unsafe {
        let mut handle: *mut CVivvyStore = ptr::null_mut();
        let config = cstr(r#"{"dimensions":3,"embedding_model":"test-model"}"#);

        // path is required and NULL.
        let code = vivvy_store_open(ptr::null(), config.as_ptr(), &mut handle);
        assert_eq!(code, ERR_NULL_POINTER);
        assert!(handle.is_null());
    }
}

#[test]
fn invalid_json_reports_invalid_json_error() {
    const ERR_INVALID_JSON: i32 = -3;

    unsafe {
        let (store, _dir) = open_test_store(3);

        let malformed = cstr("{not valid json");
        let vector: [f32; 3] = [1.0, 0.0, 0.0];
        let mut out_id: *mut c_char = ptr::null_mut();

        let code = vivvy_store_insert(
            store,
            malformed.as_ptr(),
            vector.as_ptr(),
            vector.len(),
            &mut out_id,
        );
        assert_eq!(code, ERR_INVALID_JSON);
        assert!(out_id.is_null());

        vivvy_store_close(store);
    }
}

#[test]
fn vacuum_on_a_fresh_store_purges_nothing() {
    unsafe {
        let (store, _dir) = open_test_store(2);
        let mut purged: usize = 999;
        let code = vivvy_store_vacuum(store, 100, &mut purged);
        assert_eq!(code, 0, "vacuum failed: {}", last_error());
        assert_eq!(purged, 0);
        vivvy_store_close(store);
    }
}

#[test]
fn memory_store_crud_batch_health_and_rebuild_roundtrip() {
    unsafe {
        let (store, _dir) = open_test_store(2);
        let records = cstr(
            r#"[
                {"tenant_id":"acme","namespace":"default","content":"first"},
                {"tenant_id":"acme","namespace":"default","content":"second"}
            ]"#,
        );
        let vectors: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        let mut out_ids: *mut c_char = ptr::null_mut();
        let code = vivvy_store_insert_batch(
            store,
            records.as_ptr(),
            vectors.as_ptr(),
            2,
            2,
            &mut out_ids,
        );
        assert_eq!(code, 0, "batch insert failed: {}", last_error());
        let ids: Vec<String> = serde_json::from_str(&CStr::from_ptr(out_ids).to_string_lossy())
            .expect("batch result must be a JSON ID array");
        vivvy_free_string(out_ids);
        assert_eq!(ids.len(), 2);

        let scope = cstr(r#"{"tenant_id":"acme","namespace":"default"}"#);
        let id = cstr(&ids[0]);
        let mut out_record: *mut c_char = ptr::null_mut();
        let code = vivvy_store_get(store, scope.as_ptr(), id.as_ptr(), &mut out_record);
        assert_eq!(code, 0, "get failed: {}", last_error());
        let initial: serde_json::Value =
            serde_json::from_str(&CStr::from_ptr(out_record).to_string_lossy()).unwrap();
        vivvy_free_string(out_record);
        assert_eq!(initial["content"], "first");
        assert_eq!(initial["revision"], 1);

        let update = cstr(&format!(
            r#"{{"tenant_id":"acme","namespace":"default","id":"{}","expected_revision":1,"content":"updated"}}"#,
            ids[0]
        ));
        let code = vivvy_store_update(store, update.as_ptr(), ptr::null(), 0);
        assert_eq!(code, 0, "update failed: {}", last_error());

        let mut out_updated: *mut c_char = ptr::null_mut();
        let code = vivvy_store_get(store, scope.as_ptr(), id.as_ptr(), &mut out_updated);
        assert_eq!(code, 0, "get after update failed: {}", last_error());
        let updated: serde_json::Value =
            serde_json::from_str(&CStr::from_ptr(out_updated).to_string_lossy()).unwrap();
        vivvy_free_string(out_updated);
        assert_eq!(updated["content"], "updated");
        assert_eq!(updated["revision"], 2);

        let mut out_health: *mut c_char = ptr::null_mut();
        let code = vivvy_store_health(store, &mut out_health);
        assert_eq!(code, 0, "health failed: {}", last_error());
        let health: serde_json::Value =
            serde_json::from_str(&CStr::from_ptr(out_health).to_string_lossy()).unwrap();
        vivvy_free_string(out_health);
        assert_eq!(health["total_active_records"], 2);

        let code = vivvy_store_rebuild_index(store);
        assert_eq!(code, 0, "rebuild failed: {}", last_error());

        let ids_json = cstr(&serde_json::to_string(&ids).unwrap());
        let code = vivvy_store_forget_batch(store, scope.as_ptr(), ids_json.as_ptr());
        assert_eq!(code, 0, "batch forget failed: {}", last_error());

        let mut out_missing: *mut c_char = ptr::null_mut();
        let code = vivvy_store_get(store, scope.as_ptr(), id.as_ptr(), &mut out_missing);
        assert_eq!(code, 0, "get after forget failed: {}", last_error());
        assert_eq!(CStr::from_ptr(out_missing).to_bytes(), b"null");
        vivvy_free_string(out_missing);

        vivvy_store_close(store);
    }
}

#[test]
fn vector_index_insert_filter_search_and_delete_roundtrip() {
    unsafe {
        let config = cstr(r#"{"dimensions":2,"metric":"l2"}"#);
        let mut index: *mut CVivvyIndex = ptr::null_mut();
        assert_eq!(
            vivvy_index_open(config.as_ptr(), &mut index),
            0,
            "open: {}",
            last_error()
        );

        let red = cstr(r#"{"color":"red"}"#);
        let red_vector = [1.0f32, 0.0];
        let mut red_id = 0u64;
        assert_eq!(
            vivvy_index_insert(index, red_vector.as_ptr(), 2, red.as_ptr(), &mut red_id),
            0,
            "insert: {}",
            last_error()
        );

        let blue = cstr(r#"{"color":"blue"}"#);
        let blue_vector = [0.0f32, 1.0];
        let mut blue_id = 0u64;
        assert_eq!(
            vivvy_index_insert(index, blue_vector.as_ptr(), 2, blue.as_ptr(), &mut blue_id),
            0,
            "insert: {}",
            last_error()
        );

        let batch_vector = [0.5f32, 0.5];
        let mut out_batch_ids: *mut c_char = ptr::null_mut();
        assert_eq!(
            vivvy_index_insert_batch(index, batch_vector.as_ptr(), 2, 1, &mut out_batch_ids),
            0,
            "batch insert: {}",
            last_error()
        );
        let batch_ids: Vec<u64> =
            serde_json::from_str(&CStr::from_ptr(out_batch_ids).to_string_lossy()).unwrap();
        vivvy_free_string(out_batch_ids);
        assert_eq!(batch_ids.len(), 1);

        let filter = cstr(r#"{"eq":{"field":"color","value":"red"}}"#);
        let mut out_results: *mut c_char = ptr::null_mut();
        assert_eq!(
            vivvy_index_search(
                index,
                red_vector.as_ptr(),
                2,
                5,
                filter.as_ptr(),
                &mut out_results
            ),
            0,
            "search: {}",
            last_error()
        );
        let results: serde_json::Value =
            serde_json::from_str(&CStr::from_ptr(out_results).to_string_lossy()).unwrap();
        vivvy_free_string(out_results);
        assert_eq!(results.as_array().unwrap()[0]["id"], red_id);

        let mut deleted = 0;
        assert_eq!(
            vivvy_index_delete(index, red_id, &mut deleted),
            0,
            "delete: {}",
            last_error()
        );
        assert_eq!(deleted, 1);
        vivvy_index_close(index);
    }
}

#[test]
fn close_and_free_string_accept_null_as_a_noop() {
    unsafe {
        // Must not crash: both functions document NULL as a no-op.
        vivvy_store_close(ptr::null_mut());
        vivvy_free_string(ptr::null_mut());
    }
}

#[test]
fn recall_with_and_without_custom_scoring() {
    unsafe {
        let (store, _dir) = open_test_store(3);

        let record = cstr(
            r#"{"tenant_id":"acme","namespace":"scoring","content":"scored fact","importance":0.8}"#,
        );
        let vector: [f32; 3] = [1.0, 0.0, 0.0];
        let mut out_id: *mut c_char = ptr::null_mut();

        let code = vivvy_store_insert(
            store,
            record.as_ptr(),
            vector.as_ptr(),
            vector.len(),
            &mut out_id,
        );
        assert_eq!(code, 0, "insert failed: {}", last_error());
        assert!(!out_id.is_null());
        vivvy_free_string(out_id);

        // 1. Without scoring field in options_json
        let options_default = cstr(r#"{"tenant_id":"acme","namespace":"scoring"}"#);
        let mut out_results: *mut c_char = ptr::null_mut();
        let code = vivvy_store_recall(
            store,
            options_default.as_ptr(),
            vector.as_ptr(),
            vector.len(),
            ptr::null(),
            5,
            &mut out_results,
        );
        assert_eq!(code, 0, "default recall failed: {}", last_error());
        assert!(!out_results.is_null());
        let results_str = CStr::from_ptr(out_results).to_string_lossy().into_owned();
        assert!(results_str.contains("default_v1_scoring"));
        vivvy_free_string(out_results);

        // 2. With scoring field in options_json
        let options_custom = cstr(
            r#"{"tenant_id":"acme","namespace":"scoring","scoring":{"weight_similarity":0.9,"weight_importance":0.1,"weight_recency":0.0,"weight_reinforcement":0.0,"recency_half_life_days":14.0}}"#,
        );
        let mut out_custom_results: *mut c_char = ptr::null_mut();
        let code = vivvy_store_recall(
            store,
            options_custom.as_ptr(),
            vector.as_ptr(),
            vector.len(),
            ptr::null(),
            5,
            &mut out_custom_results,
        );
        assert_eq!(code, 0, "custom recall failed: {}", last_error());
        assert!(!out_custom_results.is_null());
        let custom_str = CStr::from_ptr(out_custom_results)
            .to_string_lossy()
            .into_owned();
        assert!(custom_str.contains("custom_scoring"));
        vivvy_free_string(out_custom_results);

        vivvy_store_close(store);
    }
}
