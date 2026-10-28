#include <node_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "vivvy.h"

typedef struct {
    VivvyStore *store;
} StoreWrapper;

typedef struct {
    VivvyIndex *index;
} IndexWrapper;

static void finalize_store(napi_env env, void *finalize_data, void *finalize_hint) {
    StoreWrapper *wrapper = (StoreWrapper *)finalize_data;
    if (wrapper) {
        if (wrapper->store) {
            vivvy_store_close(wrapper->store);
            wrapper->store = NULL;
        }
        free(wrapper);
    }
}

static void finalize_index(napi_env env, void *finalize_data, void *finalize_hint) {
    IndexWrapper *wrapper = (IndexWrapper *)finalize_data;
    if (wrapper) {
        if (wrapper->index) vivvy_index_close(wrapper->index);
        free(wrapper);
    }
}

static napi_value create_vivvy_error(napi_env env, int code, const char *msg) {
    char buf[2048];
    snprintf(buf, sizeof(buf), "vivvy: [%d] %s", code, (msg && msg[0]) ? msg : "unknown error");
    napi_value js_msg, js_err, js_code;
    napi_create_string_utf8(env, buf, NAPI_AUTO_LENGTH, &js_msg);
    napi_create_error(env, NULL, js_msg, &js_err);
    napi_create_int32(env, code, &js_code);
    napi_set_named_property(env, js_err, "code", js_code);
    return js_err;
}

static char *get_string_utf8(napi_env env, napi_value val) {
    if (!val) return NULL;
    napi_valuetype type;
    napi_typeof(env, val, &type);
    if (type != napi_string) return NULL;
    size_t len = 0;
    napi_status st = napi_get_value_string_utf8(env, val, NULL, 0, &len);
    if (st != napi_ok) return NULL;
    char *buf = (char *)malloc(len + 1);
    if (!buf) return NULL;
    napi_get_value_string_utf8(env, val, buf, len + 1, &len);
    return buf;
}

static float *get_float32_array(napi_env env, napi_value val, size_t *out_len) {
    *out_len = 0;
    if (!val) return NULL;
    bool is_typedarray = false;
    napi_is_typedarray(env, val, &is_typedarray);
    if (!is_typedarray) return NULL;
    napi_typedarray_type type;
    size_t length = 0;
    void *data = NULL;
    napi_get_typedarray_info(env, val, &type, &length, &data, NULL, NULL);
    if (type != napi_float32_array) return NULL;
    *out_len = length;
    if (length == 0 || !data) return NULL;
    float *copy = (float *)malloc(length * sizeof(float));
    if (copy) {
        memcpy(copy, data, length * sizeof(float));
    }
    return copy;
}

static StoreWrapper *get_store_wrapper(napi_env env, napi_value val) {
    if (!val) return NULL;
    void *data = NULL;
    napi_status st = napi_unwrap(env, val, &data);
    if (st != napi_ok || !data) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "invalid store instance"));
        return NULL;
    }
    StoreWrapper *wrapper = (StoreWrapper *)data;
    if (!wrapper->store) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_STORE_IO_ERROR, "store is closed"));
        return NULL;
    }
    return wrapper;
}

static IndexWrapper *get_index_wrapper(napi_env env, napi_value val) {
    void *data = NULL;
    if (!val || napi_unwrap(env, val, &data) != napi_ok || !data) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "invalid index instance"));
        return NULL;
    }
    IndexWrapper *wrapper = (IndexWrapper *)data;
    if (!wrapper->index) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_STORE_IO_ERROR, "index is closed"));
        return NULL;
    }
    return wrapper;
}

static napi_value throw_last_error(napi_env env, int code) {
    napi_throw(env, create_vivvy_error(env, code, vivvy_last_error_message()));
    return NULL;
}

/* -------------------------------------------------------------------------
 * vivvy_store_open(path, config_json)
 * ------------------------------------------------------------------------- */
static napi_value Node_StoreOpen(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 2) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (path, config_json)"));
        return NULL;
    }

    char *path = get_string_utf8(env, argv[0]);
    char *config_json = get_string_utf8(env, argv[1]);
    if (!path || !config_json) {
        free(path);
        free(config_json);
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "arguments must be strings"));
        return NULL;
    }

    VivvyStore *store = NULL;
    int rc = vivvy_store_open(path, config_json, &store);
    free(path);
    free(config_json);

    if (rc != 0) {
        const char *msg = vivvy_last_error_message();
        napi_throw(env, create_vivvy_error(env, rc, msg));
        return NULL;
    }

    StoreWrapper *wrapper = (StoreWrapper *)malloc(sizeof(StoreWrapper));
    wrapper->store = store;

    napi_value js_obj;
    napi_create_object(env, &js_obj);
    napi_wrap(env, js_obj, wrapper, finalize_store, NULL, NULL);
    return js_obj;
}

/* -------------------------------------------------------------------------
 * vivvy_store_close(store)
 * ------------------------------------------------------------------------- */
static napi_value Node_StoreClose(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 1) return NULL;

    void *data = NULL;
    napi_status st = napi_unwrap(env, argv[0], &data);
    if (st == napi_ok && data) {
        StoreWrapper *wrapper = (StoreWrapper *)data;
        if (wrapper->store) {
            vivvy_store_close(wrapper->store);
            wrapper->store = NULL;
        }
    }
    return NULL;
}

/* -------------------------------------------------------------------------
 * Async Insert Operation
 * ------------------------------------------------------------------------- */
typedef struct {
    napi_async_work work;
    napi_deferred deferred;
    VivvyStore *store;
    char *record_json;
    float *vector;
    size_t dim;
    int status_code;
    char *out_id;
    char *err_msg;
} AsyncInsertData;

static void AsyncInsert_Execute(napi_env env, void *data) {
    AsyncInsertData *d = (AsyncInsertData *)data;
    d->status_code = vivvy_store_insert(d->store, d->record_json, d->vector, d->dim, &d->out_id);
    if (d->status_code != 0) {
        const char *msg = vivvy_last_error_message();
        d->err_msg = msg ? strdup(msg) : NULL;
    }
}

static void AsyncInsert_Complete(napi_env env, napi_status status, void *data) {
    AsyncInsertData *d = (AsyncInsertData *)data;
    if (d->status_code == 0 && d->out_id) {
        napi_value js_res;
        napi_create_string_utf8(env, d->out_id, NAPI_AUTO_LENGTH, &js_res);
        vivvy_free_string(d->out_id);
        napi_resolve_deferred(env, d->deferred, js_res);
    } else {
        napi_value js_err = create_vivvy_error(env, d->status_code, d->err_msg);
        napi_reject_deferred(env, d->deferred, js_err);
    }

    if (d->err_msg) free(d->err_msg);
    free(d->record_json);
    free(d->vector);
    napi_delete_async_work(env, d->work);
    free(d);
}

static napi_value Node_StoreInsert(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 3) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, record_json, vector)"));
        return NULL;
    }

    StoreWrapper *wrapper = get_store_wrapper(env, argv[0]);
    if (!wrapper) return NULL;

    char *record_json = get_string_utf8(env, argv[1]);
    if (!record_json) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "record_json must be string"));
        return NULL;
    }

    size_t dim = 0;
    float *vector = get_float32_array(env, argv[2], &dim);
    if (!vector) {
        free(record_json);
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "vector must be non-empty Float32Array"));
        return NULL;
    }

    napi_value promise;
    napi_deferred deferred;
    napi_create_promise(env, &deferred, &promise);

    AsyncInsertData *d = (AsyncInsertData *)malloc(sizeof(AsyncInsertData));
    memset(d, 0, sizeof(AsyncInsertData));
    d->deferred = deferred;
    d->store = wrapper->store;
    d->record_json = record_json;
    d->vector = vector;
    d->dim = dim;

    napi_value resource_name;
    napi_create_string_utf8(env, "vivvy_store_insert", NAPI_AUTO_LENGTH, &resource_name);
    napi_create_async_work(env, NULL, resource_name, AsyncInsert_Execute, AsyncInsert_Complete, d, &d->work);
    napi_queue_async_work(env, d->work);

    return promise;
}

static napi_value Node_StoreInsertSync(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 3) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, record_json, vector)"));
        return NULL;
    }

    StoreWrapper *wrapper = get_store_wrapper(env, argv[0]);
    if (!wrapper) return NULL;

    char *record_json = get_string_utf8(env, argv[1]);
    if (!record_json) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "record_json must be string"));
        return NULL;
    }

    size_t dim = 0;
    float *vector = get_float32_array(env, argv[2], &dim);
    if (!vector) {
        free(record_json);
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "vector must be non-empty Float32Array"));
        return NULL;
    }

    char *out_id = NULL;
    int rc = vivvy_store_insert(wrapper->store, record_json, vector, dim, &out_id);
    free(record_json);
    free(vector);

    if (rc != 0) {
        const char *msg = vivvy_last_error_message();
        napi_throw(env, create_vivvy_error(env, rc, msg));
        return NULL;
    }

    napi_value js_res;
    napi_create_string_utf8(env, out_id, NAPI_AUTO_LENGTH, &js_res);
    vivvy_free_string(out_id);
    return js_res;
}

/* -------------------------------------------------------------------------
 * Async Recall Operation
 * ------------------------------------------------------------------------- */
typedef struct {
    napi_async_work work;
    napi_deferred deferred;
    VivvyStore *store;
    char *options_json;
    float *query_vector;
    size_t dim;
    char *query_text;
    size_t top_k;
    int status_code;
    char *out_results_json;
    char *err_msg;
} AsyncRecallData;

static void AsyncRecall_Execute(napi_env env, void *data) {
    AsyncRecallData *d = (AsyncRecallData *)data;
    d->status_code = vivvy_store_recall(d->store, d->options_json, d->query_vector, d->dim, d->query_text, d->top_k, &d->out_results_json);
    if (d->status_code != 0) {
        const char *msg = vivvy_last_error_message();
        d->err_msg = msg ? strdup(msg) : NULL;
    }
}

static void AsyncRecall_Complete(napi_env env, napi_status status, void *data) {
    AsyncRecallData *d = (AsyncRecallData *)data;
    if (d->status_code == 0 && d->out_results_json) {
        napi_value js_res;
        napi_create_string_utf8(env, d->out_results_json, NAPI_AUTO_LENGTH, &js_res);
        vivvy_free_string(d->out_results_json);
        napi_resolve_deferred(env, d->deferred, js_res);
    } else {
        napi_value js_err = create_vivvy_error(env, d->status_code, d->err_msg);
        napi_reject_deferred(env, d->deferred, js_err);
    }

    if (d->err_msg) free(d->err_msg);
    free(d->options_json);
    free(d->query_vector);
    free(d->query_text);
    napi_delete_async_work(env, d->work);
    free(d);
}

static napi_value Node_StoreRecall(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value argv[5];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 5) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, options_json, vector, query_text, top_k)"));
        return NULL;
    }

    StoreWrapper *wrapper = get_store_wrapper(env, argv[0]);
    if (!wrapper) return NULL;

    char *options_json = get_string_utf8(env, argv[1]);
    if (!options_json) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "options_json must be string"));
        return NULL;
    }

    size_t dim = 0;
    float *query_vector = get_float32_array(env, argv[2], &dim);
    char *query_text = get_string_utf8(env, argv[3]);

    int64_t top_k_val = 10;
    napi_get_value_int64(env, argv[4], &top_k_val);
    if (top_k_val <= 0) top_k_val = 10;

    napi_value promise;
    napi_deferred deferred;
    napi_create_promise(env, &deferred, &promise);

    AsyncRecallData *d = (AsyncRecallData *)malloc(sizeof(AsyncRecallData));
    memset(d, 0, sizeof(AsyncRecallData));
    d->deferred = deferred;
    d->store = wrapper->store;
    d->options_json = options_json;
    d->query_vector = query_vector;
    d->dim = dim;
    d->query_text = query_text;
    d->top_k = (size_t)top_k_val;

    napi_value resource_name;
    napi_create_string_utf8(env, "vivvy_store_recall", NAPI_AUTO_LENGTH, &resource_name);
    napi_create_async_work(env, NULL, resource_name, AsyncRecall_Execute, AsyncRecall_Complete, d, &d->work);
    napi_queue_async_work(env, d->work);

    return promise;
}

static napi_value Node_StoreRecallSync(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value argv[5];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 5) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, options_json, vector, query_text, top_k)"));
        return NULL;
    }

    StoreWrapper *wrapper = get_store_wrapper(env, argv[0]);
    if (!wrapper) return NULL;

    char *options_json = get_string_utf8(env, argv[1]);
    if (!options_json) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "options_json must be string"));
        return NULL;
    }

    size_t dim = 0;
    float *query_vector = get_float32_array(env, argv[2], &dim);
    char *query_text = get_string_utf8(env, argv[3]);

    int64_t top_k_val = 10;
    napi_get_value_int64(env, argv[4], &top_k_val);
    if (top_k_val <= 0) top_k_val = 10;

    char *out_results_json = NULL;
    int rc = vivvy_store_recall(wrapper->store, options_json, query_vector, dim, query_text, (size_t)top_k_val, &out_results_json);
    free(options_json);
    free(query_vector);
    free(query_text);

    if (rc != 0) {
        const char *msg = vivvy_last_error_message();
        napi_throw(env, create_vivvy_error(env, rc, msg));
        return NULL;
    }

    napi_value js_res;
    napi_create_string_utf8(env, out_results_json, NAPI_AUTO_LENGTH, &js_res);
    vivvy_free_string(out_results_json);
    return js_res;
}

/* -------------------------------------------------------------------------
 * Async Format Context Operation
 * ------------------------------------------------------------------------- */
typedef struct {
    napi_async_work work;
    napi_deferred deferred;
    VivvyStore *store;
    char *options_json;
    float *query_vector;
    size_t dim;
    char *query_text;
    size_t top_k;
    char *format_json;
    int status_code;
    char *out_context;
    char *err_msg;
} AsyncFormatContextData;

static void AsyncFormatContext_Execute(napi_env env, void *data) {
    AsyncFormatContextData *d = (AsyncFormatContextData *)data;
    d->status_code = vivvy_store_format_context(d->store, d->options_json, d->query_vector, d->dim, d->query_text, d->top_k, d->format_json, &d->out_context);
    if (d->status_code != 0) {
        const char *msg = vivvy_last_error_message();
        d->err_msg = msg ? strdup(msg) : NULL;
    }
}

static void AsyncFormatContext_Complete(napi_env env, napi_status status, void *data) {
    AsyncFormatContextData *d = (AsyncFormatContextData *)data;
    if (d->status_code == 0 && d->out_context) {
        napi_value js_res;
        napi_create_string_utf8(env, d->out_context, NAPI_AUTO_LENGTH, &js_res);
        vivvy_free_string(d->out_context);
        napi_resolve_deferred(env, d->deferred, js_res);
    } else {
        napi_value js_err = create_vivvy_error(env, d->status_code, d->err_msg);
        napi_reject_deferred(env, d->deferred, js_err);
    }

    if (d->err_msg) free(d->err_msg);
    free(d->options_json);
    free(d->query_vector);
    free(d->query_text);
    free(d->format_json);
    napi_delete_async_work(env, d->work);
    free(d);
}

static napi_value Node_StoreFormatContext(napi_env env, napi_callback_info info) {
    size_t argc = 6;
    napi_value argv[6];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 6) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, options_json, vector, query_text, top_k, format_json)"));
        return NULL;
    }

    StoreWrapper *wrapper = get_store_wrapper(env, argv[0]);
    if (!wrapper) return NULL;

    char *options_json = get_string_utf8(env, argv[1]);
    size_t dim = 0;
    float *query_vector = get_float32_array(env, argv[2], &dim);
    char *query_text = get_string_utf8(env, argv[3]);

    int64_t top_k_val = 10;
    napi_get_value_int64(env, argv[4], &top_k_val);
    if (top_k_val <= 0) top_k_val = 10;

    char *format_json = get_string_utf8(env, argv[5]);

    napi_value promise;
    napi_deferred deferred;
    napi_create_promise(env, &deferred, &promise);

    AsyncFormatContextData *d = (AsyncFormatContextData *)malloc(sizeof(AsyncFormatContextData));
    memset(d, 0, sizeof(AsyncFormatContextData));
    d->deferred = deferred;
    d->store = wrapper->store;
    d->options_json = options_json;
    d->query_vector = query_vector;
    d->dim = dim;
    d->query_text = query_text;
    d->top_k = (size_t)top_k_val;
    d->format_json = format_json;

    napi_value resource_name;
    napi_create_string_utf8(env, "vivvy_store_format_context", NAPI_AUTO_LENGTH, &resource_name);
    napi_create_async_work(env, NULL, resource_name, AsyncFormatContext_Execute, AsyncFormatContext_Complete, d, &d->work);
    napi_queue_async_work(env, d->work);

    return promise;
}

static napi_value Node_StoreFormatContextSync(napi_env env, napi_callback_info info) {
    size_t argc = 6;
    napi_value argv[6];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 6) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, options_json, vector, query_text, top_k, format_json)"));
        return NULL;
    }

    StoreWrapper *wrapper = get_store_wrapper(env, argv[0]);
    if (!wrapper) return NULL;

    char *options_json = get_string_utf8(env, argv[1]);
    size_t dim = 0;
    float *query_vector = get_float32_array(env, argv[2], &dim);
    char *query_text = get_string_utf8(env, argv[3]);

    int64_t top_k_val = 10;
    napi_get_value_int64(env, argv[4], &top_k_val);
    if (top_k_val <= 0) top_k_val = 10;

    char *format_json = get_string_utf8(env, argv[5]);

    char *out_context = NULL;
    int rc = vivvy_store_format_context(wrapper->store, options_json, query_vector, dim, query_text, (size_t)top_k_val, format_json, &out_context);
    free(options_json);
    free(query_vector);
    free(query_text);
    free(format_json);

    if (rc != 0) {
        const char *msg = vivvy_last_error_message();
        napi_throw(env, create_vivvy_error(env, rc, msg));
        return NULL;
    }

    napi_value js_res;
    napi_create_string_utf8(env, out_context, NAPI_AUTO_LENGTH, &js_res);
    vivvy_free_string(out_context);
    return js_res;
}

/* -------------------------------------------------------------------------
 * Backup & Vacuum
 * ------------------------------------------------------------------------- */
static napi_value Node_StoreBackupSync(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 2) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, target_path)"));
        return NULL;
    }
    StoreWrapper *wrapper = get_store_wrapper(env, argv[0]);
    if (!wrapper) return NULL;
    char *target = get_string_utf8(env, argv[1]);
    if (!target) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "target_path must be string"));
        return NULL;
    }
    int rc = vivvy_store_backup(wrapper->store, target);
    free(target);
    if (rc != 0) {
        const char *msg = vivvy_last_error_message();
        napi_throw(env, create_vivvy_error(env, rc, msg));
        return NULL;
    }
    return NULL;
}

static napi_value Node_StoreVacuumSync(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 2) {
        napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, batch_size)"));
        return NULL;
    }
    StoreWrapper *wrapper = get_store_wrapper(env, argv[0]);
    if (!wrapper) return NULL;
    int64_t batch = 100;
    napi_get_value_int64(env, argv[1], &batch);
    if (batch <= 0) batch = 100;
    size_t purged = 0;
    int rc = vivvy_store_vacuum(wrapper->store, (size_t)batch, &purged);
    if (rc != 0) {
        const char *msg = vivvy_last_error_message();
        napi_throw(env, create_vivvy_error(env, rc, msg));
        return NULL;
    }
    napi_value js_res;
    napi_create_int64(env, (int64_t)purged, &js_res);
    return js_res;
}

/* -------------------------------------------------------------------------
 * Remaining synchronous MemoryStore operations
 * ------------------------------------------------------------------------- */
static napi_value Node_StoreInsertBatchSync(napi_env env, napi_callback_info info) {
    size_t argc = 5; napi_value argv[5]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 5) { napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, records_json, vectors, dim, count)")); return NULL; }
    StoreWrapper *w = get_store_wrapper(env, argv[0]); if (!w) return NULL;
    char *records = get_string_utf8(env, argv[1]); size_t flat_len = 0; float *flat = get_float32_array(env, argv[2], &flat_len);
    int64_t dim = 0, count = 0; napi_get_value_int64(env, argv[3], &dim); napi_get_value_int64(env, argv[4], &count);
    if (!records || dim <= 0 || count < 0 || (count > 0 && !flat) || (size_t)(dim * count) != flat_len) { free(records); free(flat); napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "invalid batch arguments")); return NULL; }
    char *out = NULL; int rc = vivvy_store_insert_batch(w->store, records, flat, (size_t)dim, (size_t)count, &out); free(records); free(flat);
    if (rc != 0) return throw_last_error(env, rc);
    napi_value result; napi_create_string_utf8(env, out, NAPI_AUTO_LENGTH, &result); vivvy_free_string(out); return result;
}

static napi_value Node_StoreGetSync(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value argv[3]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 3) { napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, scope_json, id)")); return NULL; }
    StoreWrapper *w = get_store_wrapper(env, argv[0]); if (!w) return NULL;
    char *scope = get_string_utf8(env, argv[1]); char *id = get_string_utf8(env, argv[2]);
    if (!scope || !id) { free(scope); free(id); napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "scope_json and id must be strings")); return NULL; }
    char *out = NULL; int rc = vivvy_store_get(w->store, scope, id, &out); free(scope); free(id);
    if (rc != 0) return throw_last_error(env, rc);
    napi_value result; napi_create_string_utf8(env, out, NAPI_AUTO_LENGTH, &result); vivvy_free_string(out); return result;
}

static napi_value Node_StoreUpdateSync(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value argv[3]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 3) { napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, update_json, vector_or_null)")); return NULL; }
    StoreWrapper *w = get_store_wrapper(env, argv[0]); if (!w) return NULL;
    char *update = get_string_utf8(env, argv[1]); size_t dim = 0; float *vector = get_float32_array(env, argv[2], &dim);
    if (!update) { free(vector); napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "update_json must be string")); return NULL; }
    int rc = vivvy_store_update(w->store, update, vector, dim); free(update); free(vector);
    if (rc != 0) return throw_last_error(env, rc); return NULL;
}

static napi_value Node_StoreForgetSync(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value argv[3]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 3) { napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, scope_json, id)")); return NULL; }
    StoreWrapper *w = get_store_wrapper(env, argv[0]); if (!w) return NULL;
    char *scope = get_string_utf8(env, argv[1]); char *id = get_string_utf8(env, argv[2]);
    if (!scope || !id) { free(scope); free(id); napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "scope_json and id must be strings")); return NULL; }
    int rc = vivvy_store_forget(w->store, scope, id); free(scope); free(id); if (rc != 0) return throw_last_error(env, rc); return NULL;
}

static napi_value Node_StoreForgetBatchSync(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value argv[3]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 3) { napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (store, scope_json, ids_json)")); return NULL; }
    StoreWrapper *w = get_store_wrapper(env, argv[0]); if (!w) return NULL;
    char *scope = get_string_utf8(env, argv[1]); char *ids = get_string_utf8(env, argv[2]);
    if (!scope || !ids) { free(scope); free(ids); napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "scope_json and ids_json must be strings")); return NULL; }
    int rc = vivvy_store_forget_batch(w->store, scope, ids); free(scope); free(ids); if (rc != 0) return throw_last_error(env, rc); return NULL;
}

static napi_value Node_StoreHealthSync(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value argv[1]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL); if (argc < 1) return NULL;
    StoreWrapper *w = get_store_wrapper(env, argv[0]); if (!w) return NULL; char *out = NULL;
    int rc = vivvy_store_health(w->store, &out); if (rc != 0) return throw_last_error(env, rc);
    napi_value result; napi_create_string_utf8(env, out, NAPI_AUTO_LENGTH, &result); vivvy_free_string(out); return result;
}

static napi_value Node_StoreRebuildIndexSync(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value argv[1]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL); if (argc < 1) return NULL;
    StoreWrapper *w = get_store_wrapper(env, argv[0]); if (!w) return NULL; int rc = vivvy_store_rebuild_index(w->store);
    if (rc != 0) return throw_last_error(env, rc); return NULL;
}

/* -------------------------------------------------------------------------
 * Low-level Index operations
 * ------------------------------------------------------------------------- */
static napi_value Node_IndexOpen(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value argv[1]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    char *config = argc ? get_string_utf8(env, argv[0]) : NULL;
    if (!config) { napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "config_json must be string")); return NULL; }
    VivvyIndex *index = NULL; int rc = vivvy_index_open(config, &index); free(config); if (rc != 0) return throw_last_error(env, rc);
    IndexWrapper *wrapper = malloc(sizeof(IndexWrapper)); if (!wrapper) { vivvy_index_close(index); napi_throw_error(env, NULL, "out of memory"); return NULL; }
    wrapper->index = index; napi_value result; napi_create_object(env, &result); napi_wrap(env, result, wrapper, finalize_index, NULL, NULL); return result;
}

static napi_value Node_IndexClose(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value argv[1]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL); if (!argc) return NULL;
    void *data = NULL; if (napi_unwrap(env, argv[0], &data) == napi_ok && data) { IndexWrapper *w = data; if (w->index) { vivvy_index_close(w->index); w->index = NULL; } } return NULL;
}

static napi_value Node_IndexInsert(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value argv[3]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL); if (argc < 3) { napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (index, vector, metadata_json_or_null)")); return NULL; }
    IndexWrapper *w = get_index_wrapper(env, argv[0]); if (!w) return NULL; size_t dim = 0; float *vector = get_float32_array(env, argv[1], &dim); char *metadata = get_string_utf8(env, argv[2]);
    if (!vector) { free(metadata); napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "vector must be non-empty Float32Array")); return NULL; }
    uint64_t id = 0; int rc = vivvy_index_insert(w->index, vector, dim, metadata, &id); free(vector); free(metadata); if (rc != 0) return throw_last_error(env, rc);
    napi_value result; napi_create_bigint_uint64(env, id, &result); return result;
}

static napi_value Node_IndexInsertWithID(napi_env env, napi_callback_info info) {
    size_t argc = 3; napi_value argv[3]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL); if (argc < 3) return NULL;
    IndexWrapper *w = get_index_wrapper(env, argv[0]); if (!w) return NULL; bool lossless = false; uint64_t id = 0;
    napi_get_value_bigint_uint64(env, argv[1], &id, &lossless); size_t dim = 0; float *vector = get_float32_array(env, argv[2], &dim);
    if (!lossless || !vector) { free(vector); napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "id must be uint64 BigInt and vector Float32Array")); return NULL; }
    int rc = vivvy_index_insert_with_id(w->index, id, vector, dim); free(vector); if (rc != 0) return throw_last_error(env, rc); return NULL;
}

static napi_value Node_IndexInsertBatch(napi_env env, napi_callback_info info) {
    size_t argc = 4; napi_value argv[4]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL); if (argc < 4) return NULL;
    IndexWrapper *w = get_index_wrapper(env, argv[0]); if (!w) return NULL; size_t len = 0; float *flat = get_float32_array(env, argv[1], &len); int64_t dim = 0, count = 0;
    napi_get_value_int64(env, argv[2], &dim); napi_get_value_int64(env, argv[3], &count);
    if (!flat || dim <= 0 || count < 0 || (size_t)(dim * count) != len) { free(flat); napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "invalid batch vectors")); return NULL; }
    char *out = NULL; int rc = vivvy_index_insert_batch(w->index, flat, (size_t)dim, (size_t)count, &out); free(flat); if (rc != 0) return throw_last_error(env, rc);
    napi_value result; napi_create_string_utf8(env, out, NAPI_AUTO_LENGTH, &result); vivvy_free_string(out); return result;
}

static napi_value Node_IndexSearch(napi_env env, napi_callback_info info) {
    size_t argc = 4; napi_value argv[4]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL); if (argc < 4) { napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "expected (index, query, k, filter_json_or_null)")); return NULL; }
    IndexWrapper *w = get_index_wrapper(env, argv[0]); if (!w) return NULL; size_t dim = 0; float *query = get_float32_array(env, argv[1], &dim); int64_t k = 0; napi_get_value_int64(env, argv[2], &k); char *filter = get_string_utf8(env, argv[3]);
    if (!query || k <= 0) { free(query); free(filter); napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "query must be Float32Array and k positive")); return NULL; }
    char *out = NULL; int rc = vivvy_index_search(w->index, query, dim, (size_t)k, filter, &out); free(query); free(filter); if (rc != 0) return throw_last_error(env, rc);
    napi_value result; napi_create_string_utf8(env, out, NAPI_AUTO_LENGTH, &result); vivvy_free_string(out); return result;
}

static napi_value Node_IndexDelete(napi_env env, napi_callback_info info) {
    size_t argc = 2; napi_value argv[2]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL); if (argc < 2) return NULL; IndexWrapper *w = get_index_wrapper(env, argv[0]); if (!w) return NULL;
    bool lossless = false; uint64_t id = 0; napi_get_value_bigint_uint64(env, argv[1], &id, &lossless); if (!lossless) { napi_throw(env, create_vivvy_error(env, VIVVY_ERR_INVALID_INPUT, "id must be uint64 BigInt")); return NULL; }
    int deleted = 0; int rc = vivvy_index_delete(w->index, id, &deleted); if (rc != 0) return throw_last_error(env, rc); napi_value result; napi_get_boolean(env, deleted != 0, &result); return result;
}

static napi_value Node_IndexCompact(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value argv[1]; napi_get_cb_info(env, info, &argc, argv, NULL, NULL); if (!argc) return NULL; IndexWrapper *w = get_index_wrapper(env, argv[0]); if (!w) return NULL; int rc = vivvy_index_compact(w->index); if (rc != 0) return throw_last_error(env, rc); return NULL;
}

static napi_value Node_LastError(napi_env env, napi_callback_info info) {
    const char *msg = vivvy_last_error_message();
    napi_value js_res;
    napi_create_string_utf8(env, msg ? msg : "", NAPI_AUTO_LENGTH, &js_res);
    return js_res;
}

/* -------------------------------------------------------------------------
 * Module Initialization
 * ------------------------------------------------------------------------- */
#define EXPORT_FN(name, fn) do { \
    napi_value nfn; \
    napi_create_function(env, name, NAPI_AUTO_LENGTH, fn, NULL, &nfn); \
    napi_set_named_property(env, exports, name, nfn); \
} while (0)

NAPI_MODULE_INIT() {
    EXPORT_FN("storeOpen", Node_StoreOpen);
    EXPORT_FN("storeClose", Node_StoreClose);
    EXPORT_FN("storeInsert", Node_StoreInsert);
    EXPORT_FN("storeInsertSync", Node_StoreInsertSync);
    EXPORT_FN("storeRecall", Node_StoreRecall);
    EXPORT_FN("storeRecallSync", Node_StoreRecallSync);
    EXPORT_FN("storeFormatContext", Node_StoreFormatContext);
    EXPORT_FN("storeFormatContextSync", Node_StoreFormatContextSync);
    EXPORT_FN("storeBackupSync", Node_StoreBackupSync);
    EXPORT_FN("storeVacuumSync", Node_StoreVacuumSync);
    EXPORT_FN("storeInsertBatchSync", Node_StoreInsertBatchSync);
    EXPORT_FN("storeGetSync", Node_StoreGetSync);
    EXPORT_FN("storeUpdateSync", Node_StoreUpdateSync);
    EXPORT_FN("storeForgetSync", Node_StoreForgetSync);
    EXPORT_FN("storeForgetBatchSync", Node_StoreForgetBatchSync);
    EXPORT_FN("storeHealthSync", Node_StoreHealthSync);
    EXPORT_FN("storeRebuildIndexSync", Node_StoreRebuildIndexSync);
    EXPORT_FN("indexOpen", Node_IndexOpen);
    EXPORT_FN("indexClose", Node_IndexClose);
    EXPORT_FN("indexInsert", Node_IndexInsert);
    EXPORT_FN("indexInsertWithId", Node_IndexInsertWithID);
    EXPORT_FN("indexInsertBatch", Node_IndexInsertBatch);
    EXPORT_FN("indexSearch", Node_IndexSearch);
    EXPORT_FN("indexDelete", Node_IndexDelete);
    EXPORT_FN("indexCompact", Node_IndexCompact);
    EXPORT_FN("lastError", Node_LastError);
    return exports;
}
