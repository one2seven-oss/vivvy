//! Exercises the `unsafe extern "C"` boundary directly, the same way a C or
//! Go caller would, without going through cgo. This catches boundary bugs
//! (wrong error codes, bad null handling, leaked/double-freed strings)
//! closer to the source than the Go test suite in `go/vivvy_test.go` can,
//! and runs as part of the ordinary `cargo test --workspace`.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use vivvy_ffi::store::{
    vivvy_free_string, vivvy_last_error_message, vivvy_store_close, vivvy_store_insert,
    vivvy_store_open, vivvy_store_recall, vivvy_store_vacuum, CVivvyStore,
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
fn close_and_free_string_accept_null_as_a_noop() {
    unsafe {
        // Must not crash: both functions document NULL as a no-op.
        vivvy_store_close(ptr::null_mut());
        vivvy_free_string(ptr::null_mut());
    }
}
