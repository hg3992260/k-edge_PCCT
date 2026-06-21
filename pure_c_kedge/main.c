#include "kedge.h"
#include "cuda_backend.h"
#include <windows.h>
static double get_time_s_main() {
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / (double)f.QuadPart;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} PathList;

typedef struct {
    char *path;
    char *series_uid;
    double order_key;
    size_t fallback_index;
    int direct_child;
    int rows;
    int cols;
} SliceCandidate;

typedef struct {
    SliceCandidate *items;
    size_t count;
    size_t capacity;
} SliceCandidateList;

static char *dup_text(const char *text) {
    size_t len = strlen(text) + 1;
    char *copy = (char *)malloc(len);
    if (copy != NULL) {
        memcpy(copy, text, len);
    }
    return copy;
}

static int path_list_push(PathList *list, const char *path) {
    char **new_items;
    if (list->count == list->capacity) {
        size_t new_capacity = (list->capacity == 0) ? 16 : list->capacity * 2;
        new_items = (char **)realloc(list->items, new_capacity * sizeof(char *));
        if (new_items == NULL) {
            return 0;
        }
        list->items = new_items;
        list->capacity = new_capacity;
    }
    list->items[list->count] = dup_text(path);
    if (list->items[list->count] == NULL) {
        return 0;
    }
    list->count++;
    return 1;
}

static void path_list_free(PathList *list) {
    size_t i;
    for (i = 0; i < list->count; ++i) {
        free(list->items[i]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int slice_candidate_list_push(
    SliceCandidateList *list,
    const char *path,
    const char *series_uid,
    double order_key,
    size_t fallback_index,
    int direct_child,
    int rows,
    int cols
) {
    SliceCandidate *new_items;
    SliceCandidate *item;
    if (list->count == list->capacity) {
        size_t new_capacity = (list->capacity == 0) ? 16 : list->capacity * 2;
        new_items = (SliceCandidate *)realloc(list->items, new_capacity * sizeof(SliceCandidate));
        if (new_items == NULL) {
            return 0;
        }
        list->items = new_items;
        list->capacity = new_capacity;
    }
    item = &list->items[list->count];
    memset(item, 0, sizeof(*item));
    item->path = dup_text(path);
    item->series_uid = dup_text(series_uid);
    if (item->path == NULL || item->series_uid == NULL) {
        free(item->path);
        free(item->series_uid);
        memset(item, 0, sizeof(*item));
        return 0;
    }
    item->order_key = order_key;
    item->fallback_index = fallback_index;
    item->direct_child = direct_child;
    item->rows = rows;
    item->cols = cols;
    list->count++;
    return 1;
}

static void slice_candidate_list_free(SliceCandidateList *list) {
    size_t i;
    for (i = 0; i < list->count; ++i) {
        free(list->items[i].path);
        free(list->items[i].series_uid);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int is_directory_path(const char *path) {
    DWORD attrs = GetFileAttributesA(path);
    return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY);
}

static int path_exists(const char *path) {
    DWORD attrs = GetFileAttributesA(path);
    return attrs != INVALID_FILE_ATTRIBUTES;
}

static void trim_line_inplace(char *text) {
    size_t len;
    if (text == NULL) {
        return;
    }
    if ((unsigned char)text[0] == 0xEF && (unsigned char)text[1] == 0xBB && (unsigned char)text[2] == 0xBF) {
        memmove(text, text + 3, strlen(text + 3) + 1);
    }
    len = strlen(text);
    while (len > 0 && (text[len - 1] == '\n' || text[len - 1] == '\r' || text[len - 1] == ' ' || text[len - 1] == '\t')) {
        text[--len] = '\0';
    }
    while (*text == ' ' || *text == '\t') {
        memmove(text, text + 1, strlen(text));
    }
}

static int prompt_console_value(const char *label, const char *default_value, char *buffer, size_t buffer_size, int allow_empty) {
    if (default_value != NULL && default_value[0] != '\0') {
        printf("%s [%s]: ", label, default_value);
    } else {
        printf("%s: ", label);
    }
    if (fgets(buffer, (int)buffer_size, stdin) == NULL) {
        return 0;
    }
    trim_line_inplace(buffer);
    if (buffer[0] == '\0') {
        if (default_value != NULL && default_value[0] != '\0') {
            snprintf(buffer, buffer_size, "%s", default_value);
            return 1;
        }
        return allow_empty;
    }
    return 1;
}

static int collect_paths_recursive(const char *root, PathList *list) {
    WIN32_FIND_DATAA entry;
    HANDLE handle;
    char pattern[KEDGE_PATH_CAP];
    char child_path[KEDGE_PATH_CAP];
    snprintf(pattern, sizeof(pattern), "%s\\*", root);
    handle = FindFirstFileA(pattern, &entry);
    if (handle == INVALID_HANDLE_VALUE) {
        return 0;
    }
    do {
        if (strcmp(entry.cFileName, ".") == 0 || strcmp(entry.cFileName, "..") == 0) {
            continue;
        }
        snprintf(child_path, sizeof(child_path), "%s\\%s", root, entry.cFileName);
        if (entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            collect_paths_recursive(child_path, list);
        } else {
            path_list_push(list, child_path);
        }
    } while (FindNextFileA(handle, &entry));
    FindClose(handle);
    return 1;
}

static const char *base_name_of(const char *path) {
    const char *slash = strrchr(path, '\\');
    const char *slash2 = strrchr(path, '/');
    const char *best = slash;
    if (slash2 != NULL && (best == NULL || slash2 > best)) {
        best = slash2;
    }
    return (best != NULL) ? (best + 1) : path;
}

static int parent_matches_root(const char *path, const char *root) {
    char buffer[KEDGE_PATH_CAP];
    char *slash = NULL;
    snprintf(buffer, sizeof(buffer), "%s", path);
    slash = strrchr(buffer, '\\');
    if (slash == NULL) {
        slash = strrchr(buffer, '/');
    }
    if (slash == NULL) {
        return 0;
    }
    *slash = '\0';
    return _stricmp(buffer, root) == 0;
}

static int compare_selected_candidate(const void *lhs, const void *rhs) {
    const SliceCandidate *a = (const SliceCandidate *)lhs;
    const SliceCandidate *b = (const SliceCandidate *)rhs;
    if (a->order_key < b->order_key) {
        return -1;
    }
    if (a->order_key > b->order_key) {
        return 1;
    }
    return _stricmp(base_name_of(a->path), base_name_of(b->path));
}

static int collect_ordered_dicom_files(const char *root, PathList *selected_files) {
    PathList all_files;
    SliceCandidateList candidates;
    size_t fallback_counter = 0;
    size_t i;
    char *best_series_uid = NULL;
    size_t best_series_count = 0;
    size_t best_shape_count = 0;
    int best_direct_child_count = -1;
    int best_rows = 0;
    int best_cols = 0;
    size_t skipped_shape_mismatch = 0;
    memset(&all_files, 0, sizeof(all_files));
    memset(&candidates, 0, sizeof(candidates));
    if (!collect_paths_recursive(root, &all_files)) {
        return 0;
    }
    for (i = 0; i < all_files.count; ++i) {
        KEdgeStatus status;
        KEdgeDicomDataset dataset;
        const char *series_uid = NULL;
        double order_key = 0.0;
        kedge_status_reset(&status);
        if (!kedge_dicom_read_file(all_files.items[i], &dataset, &status)) {
            continue;
        }
        series_uid = (dataset.series_instance_uid[0] != '\0') ? dataset.series_instance_uid : "__NO_SERIES__";
        if (dataset.have_image_position_patient) {
            order_key = dataset.image_position_patient[2];
        } else if (dataset.have_slice_location) {
            order_key = dataset.slice_location;
        } else if (dataset.instance_number != 0) {
            order_key = (double)dataset.instance_number;
        } else {
            order_key = (double)fallback_counter;
            fallback_counter++;
        }
        if (!slice_candidate_list_push(
                &candidates,
                all_files.items[i],
                series_uid,
                order_key,
                fallback_counter,
                parent_matches_root(all_files.items[i], root),
                dataset.rows,
                dataset.cols)) {
            kedge_dicom_free(&dataset);
            path_list_free(&all_files);
            slice_candidate_list_free(&candidates);
            return 0;
        }
        kedge_dicom_free(&dataset);
    }
    path_list_free(&all_files);
    for (i = 0; i < candidates.count; ++i) {
        size_t j;
        size_t series_count = 0;
        int direct_child_count = 0;
        if (best_series_uid != NULL && _stricmp(candidates.items[i].series_uid, best_series_uid) == 0) {
            continue;
        }
        for (j = 0; j < candidates.count; ++j) {
            if (_stricmp(candidates.items[j].series_uid, candidates.items[i].series_uid) == 0) {
                series_count++;
                direct_child_count += candidates.items[j].direct_child ? 1 : 0;
            }
        }
        if (best_series_uid == NULL ||
            series_count > best_series_count ||
            (series_count == best_series_count && direct_child_count > best_direct_child_count) ||
            (series_count == best_series_count && direct_child_count == best_direct_child_count &&
             strcmp(candidates.items[i].series_uid, best_series_uid) > 0)) {
            best_series_uid = candidates.items[i].series_uid;
            best_series_count = series_count;
            best_direct_child_count = direct_child_count;
        }
    }
    if (best_series_uid == NULL) {
        slice_candidate_list_free(&candidates);
        return 1;
    }
    for (i = 0; i < candidates.count; ++i) {
        size_t j;
        size_t shape_count = 0;
        if (_stricmp(candidates.items[i].series_uid, best_series_uid) != 0) {
            continue;
        }
        for (j = 0; j < candidates.count; ++j) {
            if (_stricmp(candidates.items[j].series_uid, best_series_uid) == 0 &&
                candidates.items[j].rows == candidates.items[i].rows &&
                candidates.items[j].cols == candidates.items[i].cols) {
                shape_count++;
            }
        }
        if (best_shape_count == 0 || shape_count > best_shape_count) {
            best_shape_count = shape_count;
            best_rows = candidates.items[i].rows;
            best_cols = candidates.items[i].cols;
        }
    }
    qsort(candidates.items, candidates.count, sizeof(SliceCandidate), compare_selected_candidate);
    for (i = 0; i < candidates.count; ++i) {
        if (_stricmp(candidates.items[i].series_uid, best_series_uid) == 0) {
            if (best_shape_count > 0 &&
                (candidates.items[i].rows != best_rows || candidates.items[i].cols != best_cols)) {
                skipped_shape_mismatch++;
                continue;
            }
            if (!path_list_push(selected_files, candidates.items[i].path)) {
                slice_candidate_list_free(&candidates);
                return 0;
            }
        }
    }
    printf(
        "Selected series UID: %s | slice count: %zu | dominant shape: %dx%d | skipped shape mismatches: %zu\n",
        best_series_uid,
        selected_files->count,
        best_rows,
        best_cols,
        skipped_shape_mismatch
    );
    slice_candidate_list_free(&candidates);
    return 1;
}

static int ensure_output_dir(const char *root, const char *name, char *out_path, size_t out_size) {
    snprintf(out_path, out_size, "%s\\%s", root, name);
    return kedge_make_directory(out_path);
}

static void json_escape(const char *src, char *dst, size_t dst_size) {
    size_t in_idx = 0;
    size_t out_idx = 0;
    while (src[in_idx] != '\0' && out_idx + 2 < dst_size) {
        char ch = src[in_idx++];
        if (ch == '\\' || ch == '"') {
            dst[out_idx++] = '\\';
        }
        dst[out_idx++] = ch;
    }
    dst[out_idx] = '\0';
}

static void write_slice_support_files(const char *out_dir, const char *input_path, KEdgeComputeBackend backend) {
    char meta_path[KEDGE_PATH_CAP];
    char report_json_path[KEDGE_PATH_CAP];
    char report_html_path[KEDGE_PATH_CAP];
    char assets_dir[KEDGE_PATH_CAP];
    char iodine_assets_dir[KEDGE_PATH_CAP];
    char json[2048];
    char html[4096];
    char input_path_json[KEDGE_PATH_CAP * 2];
    const char *backend_label = "Pure C / OpenJPEG";
    if (backend == KEDGE_BACKEND_CUDA) {
        backend_label = "Pure C / OpenJPEG + CUDA";
    } else if (backend == KEDGE_BACKEND_CUDA_FULL) {
        backend_label = "Pure C / nvJPEG2000 + CUDA";
    }
    KEdgeStatus status;
    json_escape(input_path, input_path_json, sizeof(input_path_json));
    snprintf(assets_dir, sizeof(assets_dir), "%s\\native_assets", out_dir);
    snprintf(iodine_assets_dir, sizeof(iodine_assets_dir), "%s\\iodine_report", assets_dir);
    kedge_make_directory(assets_dir);
    kedge_make_directory(iodine_assets_dir);
    snprintf(meta_path, sizeof(meta_path), "%s\\report_generation_meta.json", out_dir);
    snprintf(report_json_path, sizeof(report_json_path), "%s\\iodine_kedge_report_native.json", out_dir);
    snprintf(report_html_path, sizeof(report_html_path), "%s\\iodine_kedge_report.html", out_dir);
    snprintf(
        json,
        sizeof(json),
        "{\n"
        "  \"source_path\": \"%s\",\n"
        "  \"backend_name\": \"%s\",\n"
        "  \"recon_mode\": \"spectral_prior\",\n"
        "  \"report_interaction_version\": 2,\n"
        "  \"report_content_version\": 8,\n"
        "  \"kedge_model_version\": 1,\n"
        "  \"dcm_exports_ready\": true\n"
        "}\n",
        input_path_json,
        backend_label
    );
    kedge_status_reset(&status);
    kedge_write_text_file(meta_path, json, &status);
    snprintf(
        json,
        sizeof(json),
        "{\n"
        "  \"source_path\": \"%s\",\n"
        "  \"backend_name\": \"%s\",\n"
        "  \"recon_mode\": \"spectral_prior\",\n"
        "  \"exports\": [\n"
        "    \"water_weight_map.dcm\",\n"
        "    \"iodine_weight_map.dcm\",\n"
        "    \"calcium_weight_map.dcm\",\n"
        "    \"gadolinium_weight_map.dcm\",\n"
        "    \"kedge_weight_map.dcm\",\n"
        "    \"gadolinium_kedge_weight_map.dcm\"\n"
        "  ]\n"
        "}\n",
        input_path_json,
        backend_label
    );
    kedge_write_text_file(report_json_path, json, &status);
    snprintf(
        html,
        sizeof(html),
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\"><title>Iodine K-edge Report</title></head>"
        "<body><h1>Iodine K-edge Report</h1><p>Source: %s</p><ul>"
        "<li>water_weight_map.dcm</li>"
        "<li>iodine_weight_map.dcm</li>"
        "<li>calcium_weight_map.dcm</li>"
        "<li>gadolinium_weight_map.dcm</li>"
        "<li>kedge_weight_map.dcm</li>"
        "<li>gadolinium_kedge_weight_map.dcm</li>"
        "</ul><p>Backend: %s</p></body></html>\n",
        input_path,
        backend_label
    );
    kedge_write_text_file(report_html_path, html, &status);
}

static int write_band_stack_file(const char *out_dir, const KEdgeDecompositionResult *result, KEdgeStatus *status) {
    char band_path[KEDGE_PATH_CAP];
    FILE *fp = NULL;
    uint32_t version = 1;
    uint32_t band_count = 8;
    uint32_t rows;
    uint32_t cols;
    size_t band_idx;
    size_t pixel_count;
    float *buffer = NULL;
    snprintf(band_path, sizeof(band_path), "%s\\band_stack_float32.bin", out_dir);
    rows = (uint32_t)result->band_stack[0].height;
    cols = (uint32_t)result->band_stack[0].width;
    if (rows == 0 || cols == 0) {
        kedge_status_set(status, "band_stack is empty, cannot export 8-bin data.");
        return 0;
    }
    pixel_count = (size_t)rows * (size_t)cols;
    buffer = (float *)malloc(pixel_count * sizeof(float));
    if (buffer == NULL) {
        kedge_status_set(status, "Out of memory while exporting band_stack.");
        return 0;
    }
    fp = fopen(band_path, "wb");
    if (fp == NULL) {
        free(buffer);
        kedge_status_set(status, "Failed to create band_stack export: %s", band_path);
        return 0;
    }
    fwrite("KBND", 1, 4, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&band_count, sizeof(band_count), 1, fp);
    fwrite(&rows, sizeof(rows), 1, fp);
    fwrite(&cols, sizeof(cols), 1, fp);
    for (band_idx = 0; band_idx < band_count; ++band_idx) {
        size_t i;
        if (result->band_stack[band_idx].height != rows || result->band_stack[band_idx].width != cols) {
            fclose(fp);
            free(buffer);
            kedge_status_set(status, "band_stack[%zu] has inconsistent shape.", band_idx);
            return 0;
        }
        for (i = 0; i < pixel_count; ++i) {
            buffer[i] = (float)result->band_stack[band_idx].data[i];
        }
        fwrite(buffer, sizeof(float), pixel_count, fp);
    }
    fclose(fp);
    free(buffer);
    return 1;
}

static void write_batch_summary(const char *output_root, const PathList *files, int success_count) {
    char summary_txt[KEDGE_PATH_CAP];
    char summary_json[KEDGE_PATH_CAP];
    char *text = NULL;
    char *json = NULL;
    size_t cap = 65536;
    size_t len = 0;
    size_t i;
    KEdgeStatus status;
    snprintf(summary_txt, sizeof(summary_txt), "%s\\summary.txt", output_root);
    snprintf(summary_json, sizeof(summary_json), "%s\\summary.json", output_root);
    text = (char *)malloc(cap);
    json = (char *)malloc(cap);
    if (text == NULL || json == NULL) {
        free(text);
        free(json);
        return;
    }
    len += snprintf(text + len, cap - len, "Pure C K-edge Batch Summary\n===========================\n");
    len += snprintf(text + len, cap - len, "Output root: %s\nProcessed: %d / %zu\n\n", output_root, success_count, files->count);
    for (i = 0; i < files->count && len + 2048 < cap; ++i) {
        len += snprintf(text + len, cap - len, "[%zu] %s\n", i + 1, files->items[i]);
    }
    {
        size_t jlen = 0;
        jlen += snprintf(json + jlen, cap - jlen, "{\n  \"output_root\": \"");
        json_escape(output_root, json + jlen, cap - jlen);
        jlen = strlen(json);
        jlen += snprintf(json + jlen, cap - jlen, "\",\n  \"processed\": %d,\n  \"total\": %zu,\n  \"files\": [\n", success_count, files->count);
        for (i = 0; i < files->count && jlen + 2048 < cap; ++i) {
            char escaped[KEDGE_PATH_CAP * 2];
            json_escape(files->items[i], escaped, sizeof(escaped));
            jlen += snprintf(json + jlen, cap - jlen, "    \"%s\"%s\n", escaped, (i + 1 < files->count) ? "," : "");
        }
        jlen += snprintf(json + jlen, cap - jlen, "  ]\n}\n");
    }
    kedge_status_reset(&status);
    kedge_write_text_file(summary_txt, text, &status);
    kedge_write_text_file(summary_json, json, &status);
    free(text);
    free(json);
}

static const char *backend_name_text(KEdgeComputeBackend backend) {
    switch (backend) {
        case KEDGE_BACKEND_CPU:
            return "CPU";
        case KEDGE_BACKEND_CUDA:
            return "CUDA";
        case KEDGE_BACKEND_CUDA_FULL:
            return "CUDA_FULL";
        case KEDGE_BACKEND_AUTO:
            return "AUTO";
        default:
            return "UNKNOWN";
    }
}

static int parse_backend_preference(const char *text, KEdgeComputeBackend *backend_out) {
    if (text == NULL || backend_out == NULL) {
        return 0;
    }
    if (_stricmp(text, "cpu") == 0) {
        *backend_out = KEDGE_BACKEND_CPU;
        return 1;
    }
    if (_stricmp(text, "cuda-full") == 0 || _stricmp(text, "cuda_full") == 0) {
        *backend_out = KEDGE_BACKEND_CUDA_FULL;
        return 1;
    }
    if (_stricmp(text, "cuda") == 0) {
        *backend_out = KEDGE_BACKEND_CUDA;
        return 1;
    }
    if (_stricmp(text, "auto") == 0) {
        *backend_out = KEDGE_BACKEND_AUTO;
        return 1;
    }
    return 0;
}

static int process_single_file(const char *input_path, const char *output_root, const char *temp_dir, KEdgeComputeBackend backend) {
    KEdgeStatus status;
    KEdgeDicomDataset dataset;
    KEdgeDecompositionResult result;
    int ok = 0;
    int map_idx;
    char out_dir[KEDGE_PATH_CAP];
    char map_path[KEDGE_PATH_CAP];
    char series_desc[128];
    char derivation_desc[192];
    double t0, t1, t2;

    t0 = get_time_s_main();
    kedge_status_reset(&status);
    if (!kedge_dicom_read_file(input_path, &dataset, &status)) {
        fprintf(stderr, "Skip %s: %s\n", input_path, status.error);
        return 0;
    }
    fprintf(stderr, "[DEBUG] Read DICOM took %.4f s\n", get_time_s_main() - t0);

    if (!ensure_output_dir(output_root, base_name_of(input_path), out_dir, sizeof(out_dir))) {
        fprintf(stderr, "Failed to create output directory: %s\n", out_dir);
        kedge_dicom_free(&dataset);
        return 0;
    }

    t1 = get_time_s_main();
    if (!kedge_compute_decomposition_with_backend(&dataset, temp_dir, backend, &result, &status)) {
        fprintf(stderr, "Processing failed %s: %s\n", input_path, status.error);
        kedge_dicom_free(&dataset);
        return 0;
    }
    fprintf(stderr, "[DEBUG] Compute took %.4f s\n", get_time_s_main() - t1);

    t2 = get_time_s_main();
    if (result.band_stack_cached_on_device) {
        if (!kedge_cuda_sync_band_stack_from_cache(result.band_stack, &status)) {
            fprintf(stderr, "Band sync failed %s: %s\n", input_path, status.error);
            goto cleanup;
        }
    }
    if (!write_band_stack_file(out_dir, &result, &status)) {
        fprintf(stderr, "Band export failed %s: %s\n", input_path, status.error);
        goto cleanup;
    }
    for (map_idx = 0; map_idx < 6; ++map_idx) {
        KEdgeImageI16 image_i16;
        double slope;
        memset(&image_i16, 0, sizeof(image_i16));
        if (!kedge_convert_map_to_int16(&result.full_maps[map_idx], &image_i16, &slope, &status)) {
            fprintf(stderr, "Quantization failed %s: %s\n", input_path, status.error);
            goto cleanup;
        }
        snprintf(map_path, sizeof(map_path), "%s\\%s_weight_map.dcm", out_dir, kedge_output_keys[map_idx]);
        snprintf(series_desc, sizeof(series_desc), "Derived %s Weight Map", kedge_output_keys[map_idx]);
        snprintf(derivation_desc, sizeof(derivation_desc), "Pure C derived %s weight map from source DICOM.", kedge_output_keys[map_idx]);
        if (!kedge_export_derived_dicom(&dataset, &image_i16, map_path, series_desc, derivation_desc, slope, &status)) {
            fprintf(stderr, "Export failed %s: %s\n", map_path, status.error);
            kedge_free_image_i16(&image_i16);
            goto cleanup;
        }
        kedge_free_image_i16(&image_i16);
    }
    write_slice_support_files(out_dir, input_path, backend);
    fprintf(stderr, "[DEBUG] Export took %.4f s\n", get_time_s_main() - t2);

    printf("Done [%s]: %s -> %s\n", backend_name_text(backend), input_path, out_dir);
    ok = 1;

cleanup:
    fprintf(stderr, "[DEBUG] Before kedge_free_decomposition\n");
    kedge_free_decomposition(&result);
    fprintf(stderr, "[DEBUG] Before kedge_dicom_free\n");
    kedge_dicom_free(&dataset);
    fprintf(stderr, "[DEBUG] Done\n");
    return ok;
}

int main(int argc, char **argv) {
    const char *input_path = NULL;
    const char *output_root = "d:\\xl\\00020006\\pure_c_output";
    const char *temp_dir = "d:\\xl\\00020006\\pure_c_temp";
    KEdgeComputeBackend backend = KEDGE_BACKEND_AUTO;
    char input_buffer[KEDGE_PATH_CAP];
    char output_buffer[KEDGE_PATH_CAP];
    char effective_output_root[KEDGE_PATH_CAP];
    PathList files;
    size_t i;
    int success_count = 0;
    memset(&files, 0, sizeof(files));
    memset(input_buffer, 0, sizeof(input_buffer));
    memset(output_buffer, 0, sizeof(output_buffer));
    effective_output_root[0] = '\0';

    if (argc < 2) {
        printf("Pure C K-edge Console\n");
        printf("=====================\n");
        if (!prompt_console_value("Input DICOM file or directory path", NULL, input_buffer, sizeof(input_buffer), 0)) {
            fprintf(stderr, "No valid input path provided.\n");
            return 1;
        }
        if (!prompt_console_value("Input output directory, press Enter to use default", output_root, output_buffer, sizeof(output_buffer), 1)) {
            fprintf(stderr, "Failed to read output directory.\n");
            return 1;
        }
        input_path = input_buffer;
        output_root = output_buffer;
    } else {
        input_path = argv[1];
        if (argc >= 3) {
            output_root = argv[2];
        }
        if (argc >= 5 && strcmp(argv[3], "--backend") == 0) {
            if (!parse_backend_preference(argv[4], &backend)) {
                fprintf(stderr, "Invalid backend: %s\n", argv[4]);
                return 1;
            }
        }
    }
    if (!path_exists(input_path)) {
        fprintf(stderr, "Input path does not exist: %s\n", input_path);
        return 1;
    }
    kedge_make_directory(output_root);
    kedge_make_directory(temp_dir);

    if (is_directory_path(input_path)) {
        snprintf(effective_output_root, sizeof(effective_output_root), "%s\\%s", output_root, base_name_of(input_path));
        kedge_make_directory(effective_output_root);
        if (!collect_ordered_dicom_files(input_path, &files)) {
            fprintf(stderr, "Directory scan failed: %s\n", input_path);
            return 1;
        }
        printf("Selected %zu slices after series filtering.\n", files.count);
    } else {
        snprintf(effective_output_root, sizeof(effective_output_root), "%s", output_root);
        path_list_push(&files, input_path);
    }
    if (files.count == 0) {
        fprintf(stderr, "No DICOM files matched the current policy.\n");
        path_list_free(&files);
        return 1;
    }
    printf("Backend preference: %s\n", backend_name_text(backend));
    for (i = 0; i < files.count; ++i) {
        success_count += process_single_file(files.items[i], effective_output_root, temp_dir, backend);
    }
    write_batch_summary(effective_output_root, &files, success_count);
    printf("Successfully processed %d file(s).\n", success_count);
    path_list_free(&files);
    return (success_count > 0) ? 0 : 1;
}
