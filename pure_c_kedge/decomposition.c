#include "kedge.h"
#include "cuda_backend.h"

#include <windows.h>
static double get_time_s() {
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / (double)f.QuadPart;
}

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int kedge_empty_tiles[4] = {0, 7, 56, 63};
static const int kedge_bin_ranges[KEDGE_BIN_COUNT][2] = {
    {0, 28}, {28, 33}, {33, 38}, {38, 48}, {48, 62}, {62, 80}, {80, 105}, {105, 200}
};
static const int kedge_curve_offsets[KEDGE_MATERIAL_COUNT] = {0x1C0, 0x850, 0xB70, 0xED4};
static const int kedge_curve_shift_candidates[5] = {0x00, 0x40, 0x80, 0xC0, 0x100};
static const double kedge_block_suppress_strength = 0.70;
static const double kedge_projection_block_suppress_strength = 0.45;

static void upsample_rows_linear(const double *src, int rows, int cols, int factor, double *dst);
static int smooth_1d_cuda_aware(
    int enable_cuda,
    const double *src,
    int count,
    int window,
    double *dst,
    KEdgeStatus *status
);
static int normalize_in_mask_cuda_aware(
    int enable_cuda,
    const double *src,
    const uint8_t *mask,
    int width,
    int height,
    double *dst,
    KEdgeStatus *status
);
static int masked_median_profiles_cuda_aware(
    int enable_cuda,
    const double *image,
    const uint8_t *mask,
    int width,
    int height,
    double *row_profile,
    double *col_profile,
    KEdgeStatus *status
);
static int robust_normalize_1d_cuda_aware(
    int enable_cuda,
    double *values,
    int count,
    KEdgeStatus *status
);
static int suppress_lowres_block_artifacts_cuda_aware(
    int enable_cuda,
    const double *src,
    int height,
    int width,
    double strength,
    const uint8_t *mask,
    int row_step,
    int col_step,
    double *dst,
    KEdgeStatus *status
);
static int build_spectral_prior_lowres_cuda_aware(
    int enable_cuda,
    const double *direct_low,
    const double *pixel_data,
    const uint8_t *lowres_body,
    int full_rows,
    int full_cols,
    int low_rows,
    int low_cols,
    double p50,
    double p995,
    double p985,
    const double *density_norm,
    const double *structure_norm,
    double *coarse_out,
    KEdgeStatus *status
);
static int fuse_structure_map_cuda_aware(
    int enable_cuda,
    const double *coarse_low,
    const double *pixel_data,
    const uint8_t *body,
    const double *base_norm,
    int full_rows,
    int full_cols,
    int low_rows,
    int low_cols,
    double p50,
    double p995,
    double p985,
    double *full_out,
    KEdgeStatus *status
);
static void build_body_mask_full(const double *pixel_data, int rows, int cols, double mid, uint8_t *mask);

/* #region debug-point marker-scan */
static size_t find_first_pattern(const unsigned char *blob, size_t blob_size, const unsigned char *pattern, size_t pattern_size) {
    size_t pos;
    if (blob == NULL || pattern == NULL || pattern_size == 0 || blob_size < pattern_size) {
        return (size_t)-1;
    }
    for (pos = 0; pos + pattern_size <= blob_size; ++pos) {
        if (memcmp(blob + pos, pattern, pattern_size) == 0) {
            return pos;
        }
    }
    return (size_t)-1;
}

static size_t count_pattern_occurrences(const unsigned char *blob, size_t blob_size, const unsigned char *pattern, size_t pattern_size) {
    size_t pos;
    size_t count = 0;
    if (blob == NULL || pattern == NULL || pattern_size == 0 || blob_size < pattern_size) {
        return 0;
    }
    for (pos = 0; pos + pattern_size <= blob_size; ++pos) {
        if (memcmp(blob + pos, pattern, pattern_size) == 0) {
            count++;
        }
    }
    return count;
}
/* #endregion */

static const char *debug_base_name(const char *path) {
    const char *slash = strrchr(path, '\\');
    const char *slash2 = strrchr(path, '/');
    const char *best = slash;
    if (slash2 != NULL && (best == NULL || slash2 > best)) {
        best = slash2;
    }
    return (best != NULL) ? (best + 1) : path;
}

static int build_stage_dump_dir(
    const char *source_path,
    char *dir_out,
    size_t dir_out_size
) {
    const char *root = getenv("KEDGE_STAGE_DUMP_ROOT");
    const char *label = getenv("KEDGE_STAGE_DUMP_LABEL");
    const char *slice_name;
    char label_dir[KEDGE_PATH_CAP];
    if (dir_out == NULL || dir_out_size == 0 || root == NULL || root[0] == '\0' || source_path == NULL) {
        return 0;
    }
    if (label == NULL || label[0] == '\0') {
        label = "default";
    }
    slice_name = debug_base_name(source_path);
    if (!kedge_make_directory(root)) {
        return 0;
    }
    snprintf(label_dir, sizeof(label_dir), "%s\\%s", root, label);
    if (!kedge_make_directory(label_dir)) {
        return 0;
    }
    snprintf(dir_out, dir_out_size, "%s\\%s", label_dir, slice_name);
    if (!kedge_make_directory(dir_out)) {
        return 0;
    }
    return 1;
}

static int stage_dump_enabled(void) {
    const char *root = getenv("KEDGE_STAGE_DUMP_ROOT");
    return (root != NULL && root[0] != '\0');
}

static void dump_stage_tensor_f64(
    const char *source_path,
    const char *stage_name,
    const double *data,
    uint32_t planes,
    uint32_t rows,
    uint32_t cols
) {
    char dump_dir[KEDGE_PATH_CAP];
    char dump_path[KEDGE_PATH_CAP];
    FILE *fp;
    uint32_t version = 1;
    if (stage_name == NULL || data == NULL || planes == 0 || rows == 0 || cols == 0) {
        return;
    }
    if (!build_stage_dump_dir(source_path, dump_dir, sizeof(dump_dir))) {
        return;
    }
    snprintf(dump_path, sizeof(dump_path), "%s\\%s.bin", dump_dir, stage_name);
    fp = fopen(dump_path, "wb");
    if (fp == NULL) {
        return;
    }
    fwrite("KDBG", 1, 4, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&planes, sizeof(planes), 1, fp);
    fwrite(&rows, sizeof(rows), 1, fp);
    fwrite(&cols, sizeof(cols), 1, fp);
    fwrite(data, sizeof(double), (size_t)planes * (size_t)rows * (size_t)cols, fp);
    fclose(fp);
}

static void dump_stage_image_f64(
    const char *source_path,
    const char *stage_name,
    const KEdgeImageF64 *image
) {
    if (!stage_dump_enabled()) return;
    if (image == NULL || image->data == NULL) {
        return;
    }
    dump_stage_tensor_f64(
        source_path,
        stage_name,
        image->data,
        1U,
        (uint32_t)image->height,
        (uint32_t)image->width
    );
}

static void dump_stage_image_array_f64(
    const char *source_path,
    const char *stage_name,
    const KEdgeImageF64 *images,
    uint32_t image_count
) {
    uint32_t idx;
    uint32_t rows;
    uint32_t cols;
    size_t plane_size;
    double *buffer;
    if (!stage_dump_enabled()) return;
    if (images == NULL || image_count == 0 || images[0].data == NULL) {
        return;
    }
    rows = (uint32_t)images[0].height;
    cols = (uint32_t)images[0].width;
    plane_size = (size_t)rows * (size_t)cols;
    buffer = (double *)malloc((size_t)image_count * plane_size * sizeof(double));
    if (buffer == NULL) {
        return;
    }
    for (idx = 0; idx < image_count; ++idx) {
        if (images[idx].data == NULL || images[idx].height != rows || images[idx].width != cols) {
            free(buffer);
            return;
        }
        memcpy(buffer + (size_t)idx * plane_size, images[idx].data, plane_size * sizeof(double));
    }
    dump_stage_tensor_f64(source_path, stage_name, buffer, image_count, rows, cols);
    free(buffer);
}

static void dump_stage_mask_u8(
    const char *source_path,
    const char *stage_name,
    const uint8_t *mask,
    uint32_t rows,
    uint32_t cols
) {
    size_t count = (size_t)rows * (size_t)cols;
    double *buffer;
    size_t i;
    if (!stage_dump_enabled()) return;
    if (mask == NULL || rows == 0 || cols == 0) {
        return;
    }
    buffer = (double *)malloc(count * sizeof(double));
    if (buffer == NULL) {
        return;
    }
    for (i = 0; i < count; ++i) {
        buffer[i] = mask[i] ? 1.0 : 0.0;
    }
    dump_stage_tensor_f64(source_path, stage_name, buffer, 1U, rows, cols);
    free(buffer);
}

static int cuda_fast_numeric_mode_enabled(void) {
    const char *mode = getenv("KEDGE_CUDA_NUMERIC_MODE");
    return (mode != NULL && mode[0] != '\0' && _stricmp(mode, "fast") == 0);
}

static void free_image_f64(KEdgeImageF64 *image) {
    if (image != NULL) {
        free(image->data);
        image->data = NULL;
        image->width = 0;
        image->height = 0;
    }
}

static int alloc_image_f64(KEdgeImageF64 *image, size_t width, size_t height, KEdgeStatus *status) {
    size_t pixel_count = width * height;
    image->width = width;
    image->height = height;
    image->data = (double *)calloc(pixel_count, sizeof(double));
    if (image->data == NULL) {
        image->width = 0;
        image->height = 0;
        kedge_status_set(status, "Out of memory while allocating image buffer %zux%zu.", width, height);
        return 0;
    }
    return 1;
}

static int is_empty_tile_position(int tile_pos) {
    size_t i;
    for (i = 0; i < 4; ++i) {
        if (kedge_empty_tiles[i] == tile_pos) {
            return 1;
        }
    }
    return 0;
}

static float read_f32_le(const unsigned char *ptr) {
    float value = 0.0f;
    memcpy(&value, ptr, sizeof(float));
    return value;
}

static int parse_curves_at_shift(const unsigned char *blob, size_t blob_size, int shift, KEdgeCurves *curves) {
    int material_idx;
    int point_idx;
    for (material_idx = 0; material_idx < KEDGE_MATERIAL_COUNT; ++material_idx) {
        size_t base = (size_t)(kedge_curve_offsets[material_idx] + shift);
        size_t end = base + KEDGE_CURVE_POINT_COUNT * sizeof(float);
        if (end > blob_size) {
            return 0;
        }
        for (point_idx = 0; point_idx < KEDGE_CURVE_POINT_COUNT; ++point_idx) {
            curves->materials[material_idx][point_idx] = (double)read_f32_le(blob + base + point_idx * sizeof(float));
        }
    }
    curves->selected_shift = shift;
    return 1;
}

static void compute_bin_curve_matrix(const KEdgeCurves *curves, double out_matrix[KEDGE_BIN_COUNT][KEDGE_MATERIAL_COUNT]) {
    int bin_idx;
    int material_idx;
    for (bin_idx = 0; bin_idx < KEDGE_BIN_COUNT; ++bin_idx) {
        int lo = kedge_bin_ranges[bin_idx][0];
        int hi = kedge_bin_ranges[bin_idx][1];
        for (material_idx = 0; material_idx < KEDGE_MATERIAL_COUNT; ++material_idx) {
            double sum = 0.0;
            int i;
            for (i = lo; i < hi; ++i) {
                sum += curves->materials[material_idx][i];
            }
            out_matrix[bin_idx][material_idx] = sum / (double)(hi - lo);
        }
    }
}

static int curves_are_valid(const KEdgeCurves *curves) {
    int material_idx;
    double matrix[KEDGE_BIN_COUNT][KEDGE_MATERIAL_COUNT];
    for (material_idx = 0; material_idx < KEDGE_MATERIAL_COUNT; ++material_idx) {
        int i;
        int positive_head = 0;
        double max_abs = 0.0;
        for (i = 0; i < KEDGE_CURVE_POINT_COUNT; ++i) {
            double v = curves->materials[material_idx][i];
            if (!isfinite(v)) {
                return 0;
            }
            if (fabs(v) > max_abs) {
                max_abs = fabs(v);
            }
            if (i < 16 && v > 0.0) {
                positive_head++;
            }
        }
        if (max_abs > 1e6 || positive_head < 12) {
            return 0;
        }
    }
    compute_bin_curve_matrix(curves, matrix);
    for (material_idx = 0; material_idx < KEDGE_MATERIAL_COUNT; ++material_idx) {
        int bin_idx;
        for (bin_idx = 0; bin_idx < KEDGE_BIN_COUNT; ++bin_idx) {
            if (!isfinite(matrix[bin_idx][material_idx])) {
                return 0;
            }
        }
    }
    return 1;
}

static int choose_curve_shift(const unsigned char *blob, size_t blob_size, KEdgeCurves *curves) {
    size_t i;
    for (i = 0; i < sizeof(kedge_curve_shift_candidates) / sizeof(kedge_curve_shift_candidates[0]); ++i) {
        if (parse_curves_at_shift(blob, blob_size, kedge_curve_shift_candidates[i], curves) && curves_are_valid(curves)) {
            return 1;
        }
    }
    return parse_curves_at_shift(blob, blob_size, 0, curves);
}

static void zscore_8(const double in_values[8], double out_values[8]) {
    int i;
    double mean = 0.0;
    double variance = 0.0;
    for (i = 0; i < 8; ++i) {
        mean += in_values[i];
    }
    mean /= 8.0;
    for (i = 0; i < 8; ++i) {
        double d = in_values[i] - mean;
        variance += d * d;
    }
    variance /= 8.0;
    variance = sqrt(variance);
    if (variance < 1e-9) {
        variance = 1e-9;
    }
    for (i = 0; i < 8; ++i) {
        out_values[i] = (in_values[i] - mean) / variance;
    }
}

static int solve_linear_system(double *matrix, double *rhs, int n) {
    int i;
    int j;
    int k;
    for (i = 0; i < n; ++i) {
        int pivot = i;
        double pivot_abs = fabs(matrix[i * n + i]);
        for (j = i + 1; j < n; ++j) {
            double candidate = fabs(matrix[j * n + i]);
            if (candidate > pivot_abs) {
                pivot = j;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs < 1e-12) {
            return 0;
        }
        if (pivot != i) {
            for (k = i; k < n; ++k) {
                double temp = matrix[i * n + k];
                matrix[i * n + k] = matrix[pivot * n + k];
                matrix[pivot * n + k] = temp;
            }
            {
                double temp = rhs[i];
                rhs[i] = rhs[pivot];
                rhs[pivot] = temp;
            }
        }
        for (j = i + 1; j < n; ++j) {
            double factor = matrix[j * n + i] / matrix[i * n + i];
            rhs[j] -= factor * rhs[i];
            for (k = i; k < n; ++k) {
                matrix[j * n + k] -= factor * matrix[i * n + k];
            }
        }
    }
    for (i = n - 1; i >= 0; --i) {
        double sum = rhs[i];
        for (j = i + 1; j < n; ++j) {
            sum -= matrix[i * n + j] * rhs[j];
        }
        rhs[i] = sum / matrix[i * n + i];
    }
    return 1;
}

static void project_out(const double signal[8], const double *nuisance_columns, int nuisance_count, double out_signal[8]) {
    double gram[25];
    double rhs[5];
    double coef[5];
    int row;
    int col;
    memset(gram, 0, sizeof(gram));
    memset(rhs, 0, sizeof(rhs));
    memset(coef, 0, sizeof(coef));
    for (row = 0; row < nuisance_count; ++row) {
        int k;
        for (k = 0; k < 8; ++k) {
            rhs[row] += nuisance_columns[k * nuisance_count + row] * signal[k];
        }
        for (col = 0; col < nuisance_count; ++col) {
            for (k = 0; k < 8; ++k) {
                gram[row * nuisance_count + col] += nuisance_columns[k * nuisance_count + row] * nuisance_columns[k * nuisance_count + col];
            }
            if (row == col) {
                gram[row * nuisance_count + col] += 1e-8;
            }
        }
    }
    memcpy(coef, rhs, nuisance_count * sizeof(double));
    if (!solve_linear_system(gram, coef, nuisance_count)) {
        memcpy(out_signal, signal, 8 * sizeof(double));
        return;
    }
    for (row = 0; row < 8; ++row) {
        double projected = signal[row];
        for (col = 0; col < nuisance_count; ++col) {
            projected -= nuisance_columns[row * nuisance_count + col] * coef[col];
        }
        out_signal[row] = projected;
    }
}

static void normalize_l1_8(double values[8]) {
    int i;
    double sum_abs = 0.0;
    for (i = 0; i < 8; ++i) {
        sum_abs += fabs(values[i]);
    }
    if (sum_abs < 1e-9) {
        sum_abs = 1e-9;
    }
    for (i = 0; i < 8; ++i) {
        values[i] /= sum_abs;
    }
}

static void build_material_vectors(
    const double matrix[KEDGE_BIN_COUNT][KEDGE_MATERIAL_COUNT],
    double material_vectors[KEDGE_MATERIAL_COUNT][KEDGE_BIN_COUNT],
    double iodine_kedge[8],
    double gadolinium_kedge[8]
) {
    double targets[KEDGE_MATERIAL_COUNT][8];
    int target_idx;
    for (target_idx = 0; target_idx < KEDGE_MATERIAL_COUNT; ++target_idx) {
        double column[8];
        int bin_idx;
        for (bin_idx = 0; bin_idx < 8; ++bin_idx) {
            column[bin_idx] = matrix[bin_idx][target_idx];
        }
        zscore_8(column, targets[target_idx]);
    }
    for (target_idx = 0; target_idx < KEDGE_MATERIAL_COUNT; ++target_idx) {
        double nuisance[8 * 4];
        int nuisance_col = 0;
        int material_idx;
        for (material_idx = 0; material_idx < KEDGE_MATERIAL_COUNT; ++material_idx) {
            int bin_idx;
            if (material_idx == target_idx) {
                continue;
            }
            for (bin_idx = 0; bin_idx < 8; ++bin_idx) {
                nuisance[bin_idx * 4 + nuisance_col] = targets[material_idx][bin_idx];
            }
            nuisance_col++;
        }
        for (material_idx = 0; material_idx < 8; ++material_idx) {
            nuisance[material_idx * 4 + nuisance_col] = 1.0;
        }
        project_out(targets[target_idx], nuisance, 4, material_vectors[target_idx]);
        normalize_l1_8(material_vectors[target_idx]);
    }
    {
        double nuisance[8 * 4];
        double target[8] = {0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int bin_idx;
        for (bin_idx = 0; bin_idx < 8; ++bin_idx) {
            nuisance[bin_idx * 4 + 0] = targets[KEDGE_MATERIAL_WATER][bin_idx];
            nuisance[bin_idx * 4 + 1] = targets[KEDGE_MATERIAL_CALCIUM][bin_idx];
            nuisance[bin_idx * 4 + 2] = targets[KEDGE_MATERIAL_GADOLINIUM][bin_idx];
            nuisance[bin_idx * 4 + 3] = 1.0;
        }
        project_out(target, nuisance, 4, iodine_kedge);
        normalize_l1_8(iodine_kedge);
    }
    {
        double nuisance[8 * 4];
        double target[8] = {0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0};
        int bin_idx;
        for (bin_idx = 0; bin_idx < 8; ++bin_idx) {
            nuisance[bin_idx * 4 + 0] = targets[KEDGE_MATERIAL_WATER][bin_idx];
            nuisance[bin_idx * 4 + 1] = targets[KEDGE_MATERIAL_IODINE][bin_idx];
            nuisance[bin_idx * 4 + 2] = targets[KEDGE_MATERIAL_CALCIUM][bin_idx];
            nuisance[bin_idx * 4 + 3] = 1.0;
        }
        project_out(target, nuisance, 4, gadolinium_kedge);
        normalize_l1_8(gadolinium_kedge);
    }
}

static void assemble_interleaved(const KEdgeImageF64 *tiles, KEdgeImageF64 *full) {
    int tile_pos;
    int idx = 0;
    for (tile_pos = 0; tile_pos < 64; ++tile_pos) {
        int tile_row;
        int tile_col;
        int band_idx;
        if (is_empty_tile_position(tile_pos)) {
            continue;
        }
        tile_row = tile_pos / 8;
        tile_col = tile_pos % 8;
        for (band_idx = 0; band_idx < 8; ++band_idx) {
            int row0 = (tile_row * 8 + band_idx) * KEDGE_TILE_BAND_HEIGHT;
            int col0 = tile_col * KEDGE_TILE_SIDE;
            int y;
            for (y = 0; y < KEDGE_TILE_BAND_HEIGHT; ++y) {
                memcpy(
                    full->data + (size_t)(row0 + y) * full->width + col0,
                    tiles[idx].data + (size_t)(band_idx * KEDGE_TILE_BAND_HEIGHT + y) * tiles[idx].width,
                    KEDGE_TILE_SIDE * sizeof(double)
                );
            }
        }
        idx++;
    }
}

static void assemble_band_stack(const KEdgeImageF64 *tiles, KEdgeImageF64 band_stack[8]) {
    int tile_pos;
    int idx = 0;
    for (tile_pos = 0; tile_pos < 64; ++tile_pos) {
        int tile_row;
        int tile_col;
        int band_idx;
        if (is_empty_tile_position(tile_pos)) {
            continue;
        }
        tile_row = tile_pos / 8;
        tile_col = tile_pos % 8;
        for (band_idx = 0; band_idx < 8; ++band_idx) {
            int row0 = tile_row * KEDGE_TILE_BAND_HEIGHT;
            int col0 = tile_col * KEDGE_TILE_SIDE;
            int y;
            for (y = 0; y < KEDGE_TILE_BAND_HEIGHT; ++y) {
                memcpy(
                    band_stack[band_idx].data + (size_t)(row0 + y) * band_stack[band_idx].width + col0,
                    tiles[idx].data + (size_t)(band_idx * KEDGE_TILE_BAND_HEIGHT + y) * tiles[idx].width,
                    KEDGE_TILE_SIDE * sizeof(double)
                );
            }
        }
        idx++;
    }
}

static void adaptive_lowres_block_shape(int height, int width, int *row_step, int *col_step) {
    *row_step = (height > 0) ? (height / 8) : 1;
    *col_step = (width > 0) ? (width / 8) : 1;
    if (*row_step < 1) {
        *row_step = 1;
    }
    if (*col_step < 1) {
        *col_step = 1;
    }
}

static double median_finite(double *values, size_t count) {
    double *tmp;
    double out;
    if (count == 0) {
        return 0.0;
    }
    tmp = (double *)malloc(count * sizeof(double));
    if (tmp == NULL) {
        return 0.0;
    }
    memcpy(tmp, values, count * sizeof(double));
    out = kedge_percentile(tmp, count, 50.0);
    free(tmp);
    return out;
}

static void block_reduce_grid(
    const double *image,
    int height,
    int width,
    int row_step,
    int col_step,
    const uint8_t *mask,
    double *grid
) {
    int grid_h = height / row_step;
    int grid_w = width / col_step;
    size_t patch_cap = (size_t)row_step * col_step;
    double *tmp_all = (double *)malloc(patch_cap * sizeof(double));
    double *tmp_masked = (double *)malloc(patch_cap * sizeof(double));
    int iy;
    int ix;
    if (tmp_all == NULL || tmp_masked == NULL) {
        free(tmp_all);
        free(tmp_masked);
        memset(grid, 0, (size_t)grid_h * grid_w * sizeof(double));
        return;
    }
    for (iy = 0; iy < grid_h; ++iy) {
        for (ix = 0; ix < grid_w; ++ix) {
            int y0 = iy * row_step;
            int x0 = ix * col_step;
            size_t count = 0;
            size_t count_masked = 0;
            int y;
            int x;
            for (y = 0; y < row_step; ++y) {
                for (x = 0; x < col_step; ++x) {
                    size_t idx = (size_t)(y0 + y) * width + (x0 + x);
                    double v = image[idx];
                    if (!isfinite(v)) {
                        continue;
                    }
                    tmp_all[count++] = v;
                    if (mask != NULL && mask[idx]) {
                        tmp_masked[count_masked++] = v;
                    }
                }
            }
            if (mask != NULL && count_masked >= 8) {
                grid[(size_t)iy * grid_w + ix] = median_finite(tmp_masked, count_masked);
            } else {
                grid[(size_t)iy * grid_w + ix] = median_finite(tmp_all, count);
            }
        }
    }
    free(tmp_all);
    free(tmp_masked);
}

static void expand_block_grid(const double *grid, int grid_h, int grid_w, int row_step, int col_step, int height, int width, double *expanded) {
    int y;
    for (y = 0; y < height; ++y) {
        int gy = y / row_step;
        int x;
        if (gy >= grid_h) {
            gy = grid_h - 1;
        }
        for (x = 0; x < width; ++x) {
            int gx = x / col_step;
            if (gx >= grid_w) {
                gx = grid_w - 1;
            }
            expanded[(size_t)y * width + x] = grid[(size_t)gy * grid_w + gx];
        }
    }
}

static int suppress_lowres_block_artifacts(
    const double *src,
    int height,
    int width,
    double strength,
    const uint8_t *mask,
    int row_step,
    int col_step,
    double *dst
) {
    int grid_h;
    int grid_w;
    double *cell_grid;
    double *expanded;
    double *row_trend;
    double *col_trend;
    double global_level;
    int iy;
    if (strength <= 0.0) {
        memcpy(dst, src, (size_t)height * width * sizeof(double));
        return 1;
    }
    grid_h = height / row_step;
    grid_w = width / col_step;
    if (grid_h <= 0 || grid_w <= 0) {
        memcpy(dst, src, (size_t)height * width * sizeof(double));
        return 1;
    }
    cell_grid = (double *)malloc((size_t)grid_h * grid_w * sizeof(double));
    expanded = (double *)malloc((size_t)height * width * sizeof(double));
    row_trend = (double *)malloc((size_t)grid_h * sizeof(double));
    col_trend = (double *)malloc((size_t)grid_w * sizeof(double));
    if (cell_grid == NULL || expanded == NULL || row_trend == NULL || col_trend == NULL) {
        free(cell_grid);
        free(expanded);
        free(row_trend);
        free(col_trend);
        memcpy(dst, src, (size_t)height * width * sizeof(double));
        return 0;
    }
    block_reduce_grid(src, height, width, row_step, col_step, mask, cell_grid);
    for (iy = 0; iy < grid_h; ++iy) {
        row_trend[iy] = median_finite(cell_grid + (size_t)iy * grid_w, (size_t)grid_w);
    }
    {
        int ix;
        double *tmp = (double *)malloc((size_t)grid_h * sizeof(double));
        for (ix = 0; ix < grid_w; ++ix) {
            int y;
            for (y = 0; y < grid_h; ++y) {
                tmp[y] = cell_grid[(size_t)y * grid_w + ix];
            }
            col_trend[ix] = median_finite(tmp, (size_t)grid_h);
        }
        global_level = median_finite(cell_grid, (size_t)grid_h * grid_w);
        for (iy = 0; iy < grid_h; ++iy) {
            for (ix = 0; ix < grid_w; ++ix) {
                cell_grid[(size_t)iy * grid_w + ix] -= (row_trend[iy] + col_trend[ix] - global_level);
            }
        }
        free(tmp);
    }
    expand_block_grid(cell_grid, grid_h, grid_w, row_step, col_step, height, width, expanded);
    {
        size_t i;
        size_t count = (size_t)height * width;
        for (i = 0; i < count; ++i) {
            dst[i] = src[i] - strength * expanded[i];
        }
    }
    free(cell_grid);
    free(expanded);
    free(row_trend);
    free(col_trend);
    return 1;
}

static void suppress_band_stack_block_artifacts(int enable_cuda, KEdgeImageF64 band_stack[8], const uint8_t *lowres_body, KEdgeStatus *status) {
    int row_step;
    int col_step;
    int band_idx;
    adaptive_lowres_block_shape((int)band_stack[0].height, (int)band_stack[0].width, &row_step, &col_step);
    for (band_idx = 0; band_idx < 8; ++band_idx) {
        size_t plane_size = band_stack[band_idx].width * band_stack[band_idx].height;
        double *tmp = (double *)malloc(plane_size * sizeof(double));
        if (tmp == NULL) {
            continue;
        }
        if (!suppress_lowres_block_artifacts_cuda_aware(
            enable_cuda,
            band_stack[band_idx].data,
            (int)band_stack[band_idx].height,
            (int)band_stack[band_idx].width,
            kedge_block_suppress_strength,
            lowres_body,
            row_step,
            col_step,
            tmp,
            status
        )) {
            free(tmp);
            continue;
        }
        memcpy(band_stack[band_idx].data, tmp, plane_size * sizeof(double));
        free(tmp);
    }
}

static int make_double_pixel_data(const KEdgeDicomDataset *dataset, double **pixel_out, KEdgeStatus *status) {
    size_t pixel_count = (size_t)dataset->rows * (size_t)dataset->cols;
    size_t i;
    double *buffer = (double *)malloc(pixel_count * sizeof(double));
    if (buffer == NULL) {
        kedge_status_set(status, "Out of memory while converting PixelData.");
        return 0;
    }
    for (i = 0; i < pixel_count; ++i) {
        buffer[i] = (double)dataset->pixel_data_i16[i] * dataset->rescale_slope + dataset->rescale_intercept;
    }
    *pixel_out = buffer;
    return 1;
}

static int build_lowres_body(const double *pixel_data, int rows, int cols, double mid, uint8_t *lowres_mask) {
    int y;
    for (y = 0; y < rows / 8; ++y) {
        int x;
        for (x = 0; x < cols; ++x) {
            int k;
            int hits = 0;
            for (k = 0; k < 8; ++k) {
                if (pixel_data[(size_t)(y * 8 + k) * cols + x] > mid) {
                    hits++;
                }
            }
            lowres_mask[(size_t)y * cols + x] = (((double)hits / 8.0) > 0.2) ? 1U : 0U;
        }
    }
    return 1;
}

static void compute_direct_low(const KEdgeImageF64 band_stack[8], const double weights[8], KEdgeImageF64 *out_map) {
    size_t idx;
    size_t pixel_count = out_map->width * out_map->height;
    for (idx = 0; idx < pixel_count; ++idx) {
        int band_idx;
        double acc = 0.0;
        for (band_idx = 0; band_idx < 8; ++band_idx) {
            acc += weights[band_idx] * band_stack[band_idx].data[idx];
        }
        out_map->data[idx] = acc;
    }
}

static double masked_median_axis1(const double *image, const uint8_t *mask, int width, int row) {
    double *tmp = (double *)malloc((size_t)width * sizeof(double));
    int count = 0;
    int x;
    double result = 0.0;
    for (x = 0; x < width; ++x) {
        if (mask == NULL || mask[(size_t)row * width + x]) {
            tmp[count++] = image[(size_t)row * width + x];
        }
    }
    if (count > 0) {
        result = kedge_percentile(tmp, (size_t)count, 50.0);
    }
    free(tmp);
    return result;
}

static double masked_median_axis0(const double *image, const uint8_t *mask, int width, int height, int col) {
    double *tmp = (double *)malloc((size_t)height * sizeof(double));
    int count = 0;
    int y;
    double result = 0.0;
    for (y = 0; y < height; ++y) {
        if (mask == NULL || mask[(size_t)y * width + col]) {
            tmp[count++] = image[(size_t)y * width + col];
        }
    }
    if (count > 0) {
        result = kedge_percentile(tmp, (size_t)count, 50.0);
    }
    free(tmp);
    return result;
}

static void smooth_1d(const double *src, int count, int window, double *dst) {
    int radius;
    int i;
    if (window <= 1) {
        memcpy(dst, src, (size_t)count * sizeof(double));
        return;
    }
    if ((window % 2) == 0) {
        window += 1;
    }
    radius = window / 2;
    for (i = 0; i < count; ++i) {
        int begin = i - radius;
        int end = i + radius;
        int j;
        double sum = 0.0;
        int samples = 0;
        if (begin < 0) {
            begin = 0;
        }
        if (end >= count) {
            end = count - 1;
        }
        for (j = begin; j <= end; ++j) {
            sum += src[j];
            samples++;
        }
        dst[i] = (samples > 0) ? (sum / (double)samples) : 0.0;
    }
}

static void robust_normalize_1d(double *values, int count) {
    double *tmp = (double *)malloc((size_t)count * sizeof(double));
    double lo;
    double hi;
    int i;
    memcpy(tmp, values, (size_t)count * sizeof(double));
    lo = kedge_percentile(tmp, (size_t)count, 1.0);
    memcpy(tmp, values, (size_t)count * sizeof(double));
    hi = kedge_percentile(tmp, (size_t)count, 99.0);
    free(tmp);
    if (hi - lo < 1e-9) {
        memset(values, 0, (size_t)count * sizeof(double));
        return;
    }
    for (i = 0; i < count; ++i) {
        double v = (values[i] - lo) / (hi - lo);
        if (v < 0.0) {
            v = 0.0;
        }
        if (v > 1.0) {
            v = 1.0;
        }
        values[i] = v;
    }
}

static double percentile_scalar_masked(const double *src, const uint8_t *mask, size_t count, double pct) {
    double *tmp = (double *)malloc(count * sizeof(double));
    size_t used = 0;
    size_t i;
    double out = 0.0;
    if (tmp == NULL) {
        return 0.0;
    }
    for (i = 0; i < count; ++i) {
        if ((mask == NULL || mask[i]) && isfinite(src[i])) {
            tmp[used++] = src[i];
        }
    }
    if (used > 0) {
        out = kedge_percentile(tmp, used, pct);
    }
    free(tmp);
    return out;
}

static void normalize_in_mask(const double *src, const uint8_t *mask, int width, int height, double *dst) {
    size_t pixel_count = (size_t)width * (size_t)height;
    double *tmp = (double *)malloc(pixel_count * sizeof(double));
    size_t count = 0;
    size_t i;
    if (tmp == NULL) {
        memset(dst, 0, pixel_count * sizeof(double));
        return;
    }
    for (i = 0; i < pixel_count; ++i) {
        if (mask == NULL || mask[i]) {
            tmp[count++] = src[i];
        }
    }
    if (count == 0) {
        memset(dst, 0, pixel_count * sizeof(double));
        free(tmp);
        return;
    }
    {
        double lo = kedge_percentile(tmp, count, 1.0);
        double hi;
        count = 0;
        for (i = 0; i < pixel_count; ++i) {
            if (mask == NULL || mask[i]) {
                tmp[count++] = src[i];
            }
        }
        hi = kedge_percentile(tmp, count, 99.0);
        if (hi - lo < 1e-9) {
            memset(dst, 0, pixel_count * sizeof(double));
        } else {
            for (i = 0; i < pixel_count; ++i) {
                double v = (src[i] - lo) / (hi - lo);
                if (v < 0.0) {
                    v = 0.0;
                }
                if (v > 1.0) {
                    v = 1.0;
                }
                dst[i] = (mask == NULL || mask[i]) ? v : 0.0;
            }
        }
    }
    free(tmp);
}

static int smooth_1d_cuda_aware(
    int enable_cuda,
    const double *src,
    int count,
    int window,
    double *dst,
    KEdgeStatus *status
) {
    if (!enable_cuda || !cuda_fast_numeric_mode_enabled()) {
        smooth_1d(src, count, window, dst);
        return 1;
    }
    return kedge_cuda_smooth_1d(src, count, window, dst, status);
}

static int normalize_in_mask_cuda_aware(
    int enable_cuda,
    const double *src,
    const uint8_t *mask,
    int width,
    int height,
    double *dst,
    KEdgeStatus *status
) {
    if (!enable_cuda || !cuda_fast_numeric_mode_enabled()) {
        normalize_in_mask(src, mask, width, height, dst);
        return 1;
    }
    return kedge_cuda_normalize_in_mask(src, mask, width, height, dst, status);
}

static int suppress_lowres_block_artifacts_cuda_aware(
    int enable_cuda,
    const double *src,
    int height,
    int width,
    double strength,
    const uint8_t *mask,
    int row_step,
    int col_step,
    double *dst,
    KEdgeStatus *status
) {
    if (!enable_cuda || !cuda_fast_numeric_mode_enabled()) {
        return suppress_lowres_block_artifacts(src, height, width, strength, mask, row_step, col_step, dst);
    }
    return kedge_cuda_suppress_lowres_block_artifacts(src, height, width, strength, mask, row_step, col_step, dst, status);
}

static int masked_median_profiles_cuda_aware(
    int enable_cuda,
    const double *image,
    const uint8_t *mask,
    int width,
    int height,
    double *row_profile,
    double *col_profile,
    KEdgeStatus *status
) {
    int y;
    int x;
    if (!enable_cuda || !cuda_fast_numeric_mode_enabled()) {
        for (y = 0; y < height; ++y) {
            row_profile[y] = masked_median_axis1(image, mask, width, y);
        }
        for (x = 0; x < width; ++x) {
            col_profile[x] = masked_median_axis0(image, mask, width, height, x);
        }
        return 1;
    }
    return kedge_cuda_masked_median_profiles(image, mask, width, height, row_profile, col_profile, status);
}

static int robust_normalize_1d_cuda_aware(
    int enable_cuda,
    double *values,
    int count,
    KEdgeStatus *status
) {
    if (!enable_cuda || !cuda_fast_numeric_mode_enabled()) {
        robust_normalize_1d(values, count);
        return 1;
    }
    return kedge_cuda_robust_normalize_1d(values, count, status);
}

static int compare_double_ascending(const void *lhs, const void *rhs) {
    double a = *(const double *)lhs;
    double b = *(const double *)rhs;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static void orient_full_map_positive(const double *src, const double *pixel_data, int rows, int cols, double p50, double p995, double p985, double *dst) {
    size_t count = (size_t)rows * cols;
    size_t i;
    double body_sum = 0.0, high995_sum = 0.0, high985_sum = 0.0;
    size_t body_n = 0, high995_n = 0, high985_n = 0;

    for (i = 0; i < count; ++i) {
        double p = pixel_data[i];
        if (p > p50 || p > p985) {
            double v = src[i];
            if (p > p50) { body_sum += v; body_n++; }
            if (p > p995) { high995_sum += v; high995_n++; }
            if (p > p985) { high985_sum += v; high985_n++; }
        }
    }

    double body_mean = (body_n > 0) ? (body_sum / (double)body_n) : 0.0;
    double high_mean = 0.0;
    if (high995_n >= 100) {
        high_mean = (high995_n > 0) ? (high995_sum / (double)high995_n) : 0.0;
    } else {
        high_mean = (high985_n > 0) ? (high985_sum / (double)high985_n) : 0.0;
    }

    int flip = (high_mean - body_mean < 0.0);
    for (i = 0; i < count; ++i) {
        dst[i] = flip ? -src[i] : src[i];
    }
}

static void orient_map_positive(const double *map_low, int low_rows, int low_cols, const double *pixel_data, int full_rows, int full_cols, double p50, double p995, double p985, double *dst) {
    double body_sum = 0.0, high995_sum = 0.0, high985_sum = 0.0;
    size_t body_n = 0, high995_n = 0, high985_n = 0;
    int y, x;

    for (y = 0; y < full_rows; ++y) {
        double src_y = (double)y / 8.0;
        int y0 = (int)src_y;
        int y1 = y0 + 1;
        double t = src_y - (double)y0;
        if (y0 >= low_rows) y0 = low_rows - 1;
        if (y1 >= low_rows) y1 = low_rows - 1;

        for (x = 0; x < full_cols; ++x) {
            double p = pixel_data[(size_t)y * full_cols + x];
            if (p > p50 || p > p985) {
                double a = map_low[(size_t)y0 * low_cols + x];
                double b = map_low[(size_t)y1 * low_cols + x];
                double v = a * (1.0 - t) + b * t;

                if (p > p50) { body_sum += v; body_n++; }
                if (p > p995) { high995_sum += v; high995_n++; }
                if (p > p985) { high985_sum += v; high985_n++; }
            }
        }
    }

    double body_mean = (body_n > 0) ? (body_sum / (double)body_n) : 0.0;
    double high_mean = 0.0;
    if (high995_n >= 100) {
        high_mean = (high995_n > 0) ? (high995_sum / (double)high995_n) : 0.0;
    } else {
        high_mean = (high985_n > 0) ? (high985_sum / (double)high985_n) : 0.0;
    }

    int flip = (high_mean - body_mean < 0.0);
    size_t low_count = (size_t)low_rows * low_cols;
    size_t i;
    for (i = 0; i < low_count; ++i) {
        dst[i] = flip ? -map_low[i] : map_low[i];
    }
}

static double parse_nominal_energy_kev(const KEdgeDicomDataset *dataset) {
    const char *text = dataset != NULL ? dataset->series_description : NULL;
    double value = 0.0;
    if (text != NULL) {
        if (sscanf(text, "ME %lf keV", &value) == 1 || sscanf(text, "%lf keV", &value) == 1) {
            return value;
        }
    }
    return 50.0;
}

static void downsample_rows_mean(const double *src, int rows, int cols, double *dst) {
    int y;
    for (y = 0; y < rows / 8; ++y) {
        int x;
        for (x = 0; x < cols; ++x) {
            int k;
            double sum = 0.0;
            for (k = 0; k < 8; ++k) {
                sum += src[(size_t)(y * 8 + k) * cols + x];
            }
            dst[(size_t)y * cols + x] = sum / 8.0;
        }
    }
}

static void upsample_rows_linear(const double *src, int rows, int cols, int factor, double *dst) {
    int y;
    for (y = 0; y < rows * factor; ++y) {
        double src_y = (double)y / (double)factor;
        int y0 = (int)floor(src_y);
        int y1 = y0 + 1;
        double t = src_y - (double)y0;
        int x;
        if (y0 < 0) {
            y0 = 0;
        }
        if (y1 >= rows) {
            y1 = rows - 1;
        }
        for (x = 0; x < cols; ++x) {
            double a = src[(size_t)y0 * cols + x];
            double b = src[(size_t)y1 * cols + x];
            dst[(size_t)y * cols + x] = a * (1.0 - t) + b * t;
        }
    }
}

static void compute_gradient_magnitude_lowres(const double *src, int rows, int cols, double *dst) {
    int y;
    for (y = 0; y < rows; ++y) {
        int x;
        for (x = 0; x < cols; ++x) {
            int y0 = (y > 0) ? (y - 1) : y;
            int y1 = (y + 1 < rows) ? (y + 1) : y;
            int x0 = (x > 0) ? (x - 1) : x;
            int x1 = (x + 1 < cols) ? (x + 1) : x;
            double gy = 0.5 * (src[(size_t)y1 * cols + x] - src[(size_t)y0 * cols + x]);
            double gx = 0.5 * (src[(size_t)y * cols + x1] - src[(size_t)y * cols + x0]);
            dst[(size_t)y * cols + x] = sqrt(gx * gx + gy * gy);
        }
    }
}

static int build_pixel_energy_surrogate_stack(
    const KEdgeDicomDataset *dataset,
    const double *pixel_data,
    const uint8_t *lowres_body,
    const double bin_curve_matrix[KEDGE_BIN_COUNT][KEDGE_MATERIAL_COUNT],
    KEdgeImageF64 *structure_full,
    KEdgeImageF64 band_stack[8],
    KEdgeStatus *status
) {
    static const double bin_centers[8] = {14.0, 30.5, 35.5, 43.0, 55.0, 71.0, 92.5, 152.5};
    static const double kedge_profile[8] = {0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int full_rows = (int)structure_full->height;
    int full_cols = (int)structure_full->width;
    int low_rows = (int)band_stack[0].height;
    int low_cols = (int)band_stack[0].width;
    size_t plane_size = (size_t)low_rows * low_cols;
    size_t full_size = (size_t)full_rows * full_cols;
    double *pixel_low = (double *)malloc(plane_size * sizeof(double));
    double *density_low = (double *)malloc(plane_size * sizeof(double));
    double *enhancement_src = (double *)malloc(plane_size * sizeof(double));
    double *enhancement_low = (double *)malloc(plane_size * sizeof(double));
    double *hot_src = (double *)malloc(plane_size * sizeof(double));
    double *hot_low = (double *)malloc(plane_size * sizeof(double));
    double *edge_src = (double *)malloc(plane_size * sizeof(double));
    double *edge_low = (double *)malloc(plane_size * sizeof(double));
    double water_profile[8];
    double iodine_delta[8];
    double calcium_delta[8];
    double iodine_profile[8];
    double calcium_profile[8];
    double energy_gate[8];
    double nominal_energy = parse_nominal_energy_kev(dataset);
    double p50;
    double p75;
    double p92;
    int band_idx;
    size_t idx;
    if (
        pixel_low == NULL || density_low == NULL || enhancement_src == NULL || enhancement_low == NULL ||
        hot_src == NULL || hot_low == NULL || edge_src == NULL || edge_low == NULL
    ) {
        free(pixel_low);
        free(density_low);
        free(enhancement_src);
        free(enhancement_low);
        free(hot_src);
        free(hot_low);
        free(edge_src);
        free(edge_low);
        kedge_status_set(status, "Out of memory while building energy pixel fallback.");
        return 0;
    }
    if (full_rows <= 0 || full_cols <= 0 || low_rows <= 0 || low_cols <= 0 || full_rows != low_rows * 8 || full_cols != low_cols) {
        free(pixel_low);
        free(density_low);
        free(enhancement_src);
        free(enhancement_low);
        free(hot_src);
        free(hot_low);
        free(edge_src);
        free(edge_low);
        kedge_status_set(
            status,
            "Invalid fallback dimensions full=%dx%d low=%dx%d.",
            full_rows,
            full_cols,
            low_rows,
            low_cols
        );
        return 0;
    }

    downsample_rows_mean(pixel_data, full_rows, full_cols, pixel_low);
    p50 = percentile_scalar_masked(pixel_low, lowres_body, plane_size, 50.0);
    p75 = percentile_scalar_masked(pixel_low, lowres_body, plane_size, 75.0);
    p92 = percentile_scalar_masked(pixel_low, lowres_body, plane_size, 92.0);
    normalize_in_mask(pixel_low, lowres_body, low_cols, low_rows, density_low);
    for (idx = 0; idx < plane_size; ++idx) {
        enhancement_src[idx] = pixel_low[idx] > p75 ? (pixel_low[idx] - p75) : 0.0;
        hot_src[idx] = pixel_low[idx] > p92 ? (pixel_low[idx] - p92) : 0.0;
    }
    normalize_in_mask(enhancement_src, lowres_body, low_cols, low_rows, enhancement_low);
    normalize_in_mask(hot_src, lowres_body, low_cols, low_rows, hot_low);
    compute_gradient_magnitude_lowres(pixel_low, low_rows, low_cols, edge_src);
    normalize_in_mask(edge_src, lowres_body, low_cols, low_rows, edge_low);

    for (band_idx = 0; band_idx < 8; ++band_idx) {
        water_profile[band_idx] = bin_curve_matrix[band_idx][KEDGE_MATERIAL_WATER];
        iodine_delta[band_idx] = bin_curve_matrix[band_idx][KEDGE_MATERIAL_IODINE] - bin_curve_matrix[band_idx][KEDGE_MATERIAL_WATER];
        calcium_delta[band_idx] = bin_curve_matrix[band_idx][KEDGE_MATERIAL_CALCIUM] - bin_curve_matrix[band_idx][KEDGE_MATERIAL_WATER];
    }
    zscore_8(water_profile, water_profile);
    zscore_8(iodine_delta, iodine_profile);
    zscore_8(calcium_delta, calcium_profile);
    for (band_idx = 0; band_idx < 8; ++band_idx) {
        double gate = exp(-0.5 * pow((bin_centers[band_idx] - nominal_energy) / 18.0, 2.0));
        energy_gate[band_idx] = gate;
    }
    {
        double gate_max = 0.0;
        for (band_idx = 0; band_idx < 8; ++band_idx) {
            if (energy_gate[band_idx] > gate_max) {
                gate_max = energy_gate[band_idx];
            }
        }
        if (gate_max < 1e-9) {
            gate_max = 1.0;
        }
        for (band_idx = 0; band_idx < 8; ++band_idx) {
            energy_gate[band_idx] /= gate_max;
        }
    }

    for (idx = 0; idx < full_size; ++idx) {
        structure_full->data[idx] = pixel_data[idx] - p50;
    }
    for (band_idx = 0; band_idx < 8; ++band_idx) {
        for (idx = 0; idx < plane_size; ++idx) {
            double band_value =
                0.35 * water_profile[band_idx] * density_low[idx] +
                0.95 * iodine_profile[band_idx] * enhancement_low[idx] * energy_gate[band_idx] +
                0.45 * calcium_profile[band_idx] * hot_low[idx] +
                0.25 * kedge_profile[band_idx] * edge_low[idx];
            band_stack[band_idx].data[idx] = (lowres_body == NULL || lowres_body[idx]) ? band_value : 0.0;
        }
    }

    free(pixel_low);
    free(density_low);
    free(enhancement_src);
    free(enhancement_low);
    free(hot_src);
    free(hot_low);
    free(edge_src);
    free(edge_low);
    return 1;
}

static int prepare_lowres_prior_static(
    int enable_cuda,
    const double *pixel_data,
    const double *structure_base,
    const uint8_t *lowres_body,
    int full_rows,
    int full_cols,
    int low_rows,
    int low_cols,
    double p50,
    double p995,
    double p985,
    double *density_norm,
    double *structure_norm,
    KEdgeStatus *status
) {
    double *structure_pos = (double *)malloc((size_t)full_rows * full_cols * sizeof(double));
    size_t plane_size = (size_t)low_rows * low_cols;
    double *pixel_low = NULL;
    double *structure_low = NULL;
    if (structure_pos == NULL) {
        free(structure_pos);
        kedge_status_set(status, "Out of memory while preparing shared spectral-prior buffers.");
        return 0;
    }
    if (enable_cuda && cuda_fast_numeric_mode_enabled()) {
        if (!kedge_cuda_downsample_normalize_in_mask(pixel_data, lowres_body, full_rows, full_cols, density_norm, status) ||
            !kedge_cuda_prepare_structure_norm_from_cache(pixel_data, lowres_body, full_rows, full_cols, p50, p995, p985, structure_norm, status)) {
            free(structure_pos);
            return 0;
        }
    } else {
        orient_full_map_positive(structure_base, pixel_data, full_rows, full_cols, p50, p995, p985, structure_pos);
        pixel_low = (double *)malloc(plane_size * sizeof(double));
        structure_low = (double *)malloc(plane_size * sizeof(double));
        if (pixel_low == NULL || structure_low == NULL) {
            free(pixel_low);
            free(structure_low);
            free(structure_pos);
            kedge_status_set(status, "Out of memory while preparing shared spectral-prior buffers.");
            return 0;
        }
        downsample_rows_mean(pixel_data, full_rows, full_cols, pixel_low);
        downsample_rows_mean(structure_pos, full_rows, full_cols, structure_low);
        normalize_in_mask(pixel_low, lowres_body, low_cols, low_rows, density_norm);
        normalize_in_mask(structure_low, lowres_body, low_cols, low_rows, structure_norm);
        free(pixel_low);
        free(structure_low);
    }
    free(structure_pos);
    return 1;
}

static int prepare_fullres_fusion_static(
    int enable_cuda,
    const double *base_full,
    const double *pixel_data,
    int full_rows,
    int full_cols,
    double p50,
    double p995,
    double p985,
    uint8_t *body,
    double *base_norm,
    KEdgeStatus *status
) {
    double *base_pos = (double *)malloc((size_t)full_rows * full_cols * sizeof(double));
    if (body == NULL || base_norm == NULL || base_pos == NULL) {
        free(base_pos);
        kedge_status_set(status, "Out of memory while preparing shared fusion buffers.");
        return 0;
    }
    build_body_mask_full(pixel_data, full_rows, full_cols, p50, body);
    if (enable_cuda && cuda_fast_numeric_mode_enabled()) {
        if (!kedge_cuda_prepare_base_norm_from_cache(pixel_data, body, full_rows, full_cols, p50, p995, p985, base_norm, status)) {
            free(base_pos);
            return 0;
        }
    } else {
        orient_full_map_positive(base_full, pixel_data, full_rows, full_cols, p50, p995, p985, base_pos);
        if (!normalize_in_mask_cuda_aware(enable_cuda, base_pos, body, full_cols, full_rows, base_norm, status)) {
            free(base_pos);
            return 0;
        }
    }
    free(base_pos);
    return 1;
}

static void build_spectral_prior_lowres(
    const double *direct_low,
    const double *pixel_data,
    const uint8_t *lowres_body,
    int full_rows,
    int full_cols,
    int low_rows,
    int low_cols,
    double p50,
    double p995,
    double p985,
    const double *density_norm,
    const double *structure_norm,
    double *coarse_out
) {
    size_t plane_size = (size_t)low_rows * low_cols;
    double *row_profile = (double *)malloc((size_t)low_rows * sizeof(double));
    double *col_profile = (double *)malloc((size_t)low_cols * sizeof(double));
    double *row_norm = (double *)malloc((size_t)low_rows * sizeof(double));
    double *col_norm = (double *)malloc((size_t)low_cols * sizeof(double));
    double *direct_norm = (double *)malloc(plane_size * sizeof(double));
    int y;
    int x;
    int row_step;
    int col_step;
    if (
        row_profile == NULL || col_profile == NULL || row_norm == NULL || col_norm == NULL ||
        direct_norm == NULL || density_norm == NULL || structure_norm == NULL
    ) {
        free(row_profile);
        free(col_profile);
        free(row_norm);
        free(col_norm);
        free(direct_norm);
        memset(coarse_out, 0, plane_size * sizeof(double));
        return;
    }
    adaptive_lowres_block_shape(low_rows, low_cols, &row_step, &col_step);
    suppress_lowres_block_artifacts(
        direct_low,
        low_rows,
        low_cols,
        kedge_projection_block_suppress_strength,
        lowres_body,
        row_step,
        col_step,
        direct_norm
    );
    for (y = 0; y < low_rows; ++y) {
        row_profile[y] = masked_median_axis1(direct_norm, lowres_body, low_cols, y);
    }
    for (x = 0; x < low_cols; ++x) {
        col_profile[x] = masked_median_axis0(direct_norm, lowres_body, low_cols, low_rows, x);
    }
    smooth_1d(row_profile, low_rows, 9, row_norm);
    smooth_1d(col_profile, low_cols, 257, col_norm);
    robust_normalize_1d(row_norm, low_rows);
    robust_normalize_1d(col_norm, low_cols);
    normalize_in_mask(direct_norm, lowres_body, low_cols, low_rows, direct_norm);
    for (y = 0; y < low_rows; ++y) {
        for (x = 0; x < low_cols; ++x) {
            size_t idx = (size_t)y * low_cols + x;
            double separable_prior = 0.5 * row_norm[y] + 0.5 * col_norm[x];
            double spectral_prior = 0.55 * separable_prior + 0.45 * direct_norm[idx];
            double coarse = (1.0 - 0.35) * spectral_prior + 0.35 * structure_norm[idx];
            coarse = (1.0 - 0.55) * coarse + 0.55 * density_norm[idx];
            coarse_out[idx] = lowres_body[idx] ? coarse : 0.0;
        }
    }
    normalize_in_mask(coarse_out, lowres_body, low_cols, low_rows, coarse_out);
    orient_map_positive(coarse_out, low_rows, low_cols, pixel_data, full_rows, full_cols, p50, p995, p985, coarse_out);
    for (y = 0; y < low_rows; ++y) {
        for (x = 0; x < low_cols; ++x) {
            if (!lowres_body[(size_t)y * low_cols + x]) {
                coarse_out[(size_t)y * low_cols + x] = 0.0;
            }
        }
    }
    free(row_profile);
    free(col_profile);
    free(row_norm);
    free(col_norm);
    free(direct_norm);
}

static int build_spectral_prior_lowres_cuda_aware(
    int enable_cuda,
    const double *direct_low,
    const double *pixel_data,
    const uint8_t *lowres_body,
    int full_rows,
    int full_cols,
    int low_rows,
    int low_cols,
    double p50,
    double p995,
    double p985,
    const double *density_norm,
    const double *structure_norm,
    double *coarse_out,
    KEdgeStatus *status
) {
    if (!enable_cuda || !cuda_fast_numeric_mode_enabled()) {
        build_spectral_prior_lowres(
            direct_low,
            pixel_data,
            lowres_body,
            full_rows,
            full_cols,
            low_rows,
            low_cols,
            p50,
            p995,
            p985,
            density_norm,
            structure_norm,
            coarse_out
        );
        return 1;
    }
    size_t plane_size = (size_t)low_rows * low_cols;
    double *row_profile = (double *)malloc((size_t)low_rows * sizeof(double));
    double *col_profile = (double *)malloc((size_t)low_cols * sizeof(double));
    double *row_norm = (double *)malloc((size_t)low_rows * sizeof(double));
    double *col_norm = (double *)malloc((size_t)low_cols * sizeof(double));
    double *direct_norm = (double *)malloc(plane_size * sizeof(double));
    int y;
    int x;
    int row_step;
    int col_step;
    if (
        row_profile == NULL || col_profile == NULL || row_norm == NULL || col_norm == NULL ||
        direct_norm == NULL || density_norm == NULL || structure_norm == NULL
    ) {
        free(row_profile);
        free(col_profile);
        free(row_norm);
        free(col_norm);
        free(direct_norm);
        kedge_status_set(status, "Out of memory while preparing CUDA spectral prior buffers.");
        return 0;
    }
    adaptive_lowres_block_shape(low_rows, low_cols, &row_step, &col_step);
    if (!suppress_lowres_block_artifacts_cuda_aware(
        enable_cuda,
        direct_low,
        low_rows,
        low_cols,
        kedge_projection_block_suppress_strength,
        lowres_body,
        row_step,
        col_step,
        direct_norm,
        status
    )) {
        free(row_profile);
        free(col_profile);
        free(row_norm);
        free(col_norm);
        free(direct_norm);
        return 0;
    }
    if (!masked_median_profiles_cuda_aware(
        enable_cuda,
        direct_norm,
        lowres_body,
        low_cols,
        low_rows,
        row_profile,
        col_profile,
        status
    )) {
        free(row_profile);
        free(col_profile);
        free(row_norm);
        free(col_norm);
        free(direct_norm);
        return 0;
    }
    if (!smooth_1d_cuda_aware(enable_cuda, row_profile, low_rows, 9, row_norm, status) ||
        !smooth_1d_cuda_aware(enable_cuda, col_profile, low_cols, 257, col_norm, status)) {
        free(row_profile);
        free(col_profile);
        free(row_norm);
        free(col_norm);
        free(direct_norm);
        return 0;
    }
    if (!robust_normalize_1d_cuda_aware(enable_cuda, row_norm, low_rows, status) ||
        !robust_normalize_1d_cuda_aware(enable_cuda, col_norm, low_cols, status)) {
        free(row_profile);
        free(col_profile);
        free(row_norm);
        free(col_norm);
        free(direct_norm);
        return 0;
    }
    if (!normalize_in_mask_cuda_aware(enable_cuda, direct_norm, lowres_body, low_cols, low_rows, direct_norm, status)) {
        free(row_profile);
        free(col_profile);
        free(row_norm);
        free(col_norm);
        free(direct_norm);
        return 0;
    }
    if (!kedge_cuda_combine_spectral_prior_lowres(
        row_norm,
        col_norm,
        direct_norm,
        density_norm,
        structure_norm,
        lowres_body,
        low_rows,
        low_cols,
        coarse_out,
        status
    )) {
        free(row_profile);
        free(col_profile);
        free(row_norm);
        free(col_norm);
        free(direct_norm);
        return 0;
    }
    if (!normalize_in_mask_cuda_aware(enable_cuda, coarse_out, lowres_body, low_cols, low_rows, coarse_out, status)) {
        free(row_profile);
        free(col_profile);
        free(row_norm);
        free(col_norm);
        free(direct_norm);
        return 0;
    }
    orient_map_positive(coarse_out, low_rows, low_cols, pixel_data, full_rows, full_cols, p50, p995, p985, coarse_out);
    for (y = 0; y < low_rows; ++y) {
        for (x = 0; x < low_cols; ++x) {
            if (!lowres_body[(size_t)y * low_cols + x]) {
                coarse_out[(size_t)y * low_cols + x] = 0.0;
            }
        }
    }
    free(row_profile);
    free(col_profile);
    free(row_norm);
    free(col_norm);
    free(direct_norm);
    return 1;
}

static void build_body_mask_full(const double *pixel_data, int rows, int cols, double mid, uint8_t *mask) {
    size_t count = (size_t)rows * (size_t)cols;
    size_t i;
    for (i = 0; i < count; ++i) {
        mask[i] = pixel_data[i] > mid ? 1U : 0U;
    }
}

static void fuse_structure_map(
    const double *coarse_low,
    const double *pixel_data,
    const uint8_t *body,
    const double *base_norm,
    int full_rows,
    int full_cols,
    int low_rows,
    int low_cols,
    double p50,
    double p995,
    double p985,
    double *full_out
) {
    double *coarse_up = (double *)malloc((size_t)full_rows * full_cols * sizeof(double));
    double *coarse_norm = (double *)malloc((size_t)full_rows * full_cols * sizeof(double));
    double *coarse_pos = (double *)malloc((size_t)low_rows * low_cols * sizeof(double));
    size_t pixel_count = (size_t)full_rows * full_cols;
    size_t i;
    if (body == NULL || base_norm == NULL || coarse_up == NULL || coarse_norm == NULL || coarse_pos == NULL) {
        free(coarse_up);
        free(coarse_norm);
        free(coarse_pos);
        memset(full_out, 0, pixel_count * sizeof(double));
        return;
    }
    orient_map_positive(coarse_low, low_rows, low_cols, pixel_data, full_rows, full_cols, p50, p995, p985, coarse_pos);
    upsample_rows_linear(coarse_pos, low_rows, low_cols, 8, coarse_up);
    normalize_in_mask(coarse_up, body, full_cols, full_rows, coarse_norm);
    for (i = 0; i < pixel_count; ++i) {
        full_out[i] = body[i] ? (base_norm[i] * (1.0 - 0.75 + 0.75 * coarse_norm[i])) : 0.0;
    }
    free(coarse_up);
    free(coarse_norm);
    free(coarse_pos);
}

static int fuse_structure_map_cuda_aware(
    int enable_cuda,
    const double *coarse_low,
    const double *pixel_data,
    const uint8_t *body,
    const double *base_norm,
    int full_rows,
    int full_cols,
    int low_rows,
    int low_cols,
    double p50,
    double p995,
    double p985,
    double *full_out,
    KEdgeStatus *status
) {
    if (!enable_cuda || !cuda_fast_numeric_mode_enabled()) {
        fuse_structure_map(coarse_low, pixel_data, body, base_norm, full_rows, full_cols, low_rows, low_cols, p50, p995, p985, full_out);
        return 1;
    }
    double *coarse_pos = NULL;
    size_t low_count = (size_t)low_rows * low_cols;
    coarse_pos = (double *)malloc(low_count * sizeof(double));
    if (body == NULL || base_norm == NULL || coarse_pos == NULL) {
        free(coarse_pos);
        kedge_status_set(status, "Out of memory while preparing CUDA fusion inputs.");
        return 0;
    }
    orient_map_positive(coarse_low, low_rows, low_cols, pixel_data, full_rows, full_cols, p50, p995, p985, coarse_pos);
    if (!kedge_cuda_fuse_full_map(
        coarse_pos,
        low_rows,
        low_cols,
        body,
        base_norm,
        full_rows,
        full_cols,
        full_out,
        status
    )) {
        free(coarse_pos);
        return 0;
    }
    free(coarse_pos);
    return 1;
}

static int kedge_compute_decomposition_impl(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    int enable_cuda,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
) {
    KEdgeCurves curves;
    size_t markers[512];
    int marker_count;
    int use_cuda_fast_math = (enable_cuda && cuda_fast_numeric_mode_enabled());
    int full_rows;
    int full_cols;
    int low_rows;
    int low_cols;
    KEdgeImageF64 g0_tiles[KEDGE_TILE_COUNT];
    KEdgeImageF64 g1_tiles[KEDGE_TILE_COUNT];
    KEdgeImageF64 g2_tiles[KEDGE_TILE_COUNT];
    KEdgeImageF64 g3_tiles[KEDGE_TILE_COUNT];
    KEdgeImageF64 diff_tiles[KEDGE_TILE_COUNT];
    KEdgeImageF64 structure_tiles[KEDGE_TILE_COUNT];
    KEdgeImageF64 structure_full;
    KEdgeImageF64 *band_stack = result->band_stack;
    double *pixel_data = NULL;
    uint8_t *lowres_body = NULL;
    double *prior_density_norm = NULL;
    double *prior_structure_norm = NULL;
    uint8_t *fusion_body = NULL;
    double *fusion_base_norm = NULL;
    static const unsigned char jp2c_pattern[4] = {'j', 'p', '2', 'c'};
    static const unsigned char jp_signature_pattern[8] = {'j', 'P', ' ', ' ', 0x0D, 0x0A, 0x87, 0x0A};
    static const unsigned char ftypjp2_pattern[8] = {'f', 't', 'y', 'p', 'j', 'p', '2', ' '};
    static const unsigned char soc_pattern[2] = {0xFF, 0x4F};
    static const unsigned char siz_pattern[2] = {0xFF, 0x51};
    size_t first_jp2c = (size_t)-1;
    size_t first_jp_signature = (size_t)-1;
    size_t first_ftypjp2 = (size_t)-1;
    size_t first_soc = (size_t)-1;
    size_t first_siz = (size_t)-1;
    size_t soc_count = 0;
    size_t siz_count = 0;
    int use_marker_path = 0;
    int i;
    double t_init = get_time_s();
    memset(result, 0, sizeof(*result));
    memset(&curves, 0, sizeof(curves));
    memset(g0_tiles, 0, sizeof(g0_tiles));
    memset(g1_tiles, 0, sizeof(g1_tiles));
    memset(g2_tiles, 0, sizeof(g2_tiles));
    memset(g3_tiles, 0, sizeof(g3_tiles));
    memset(diff_tiles, 0, sizeof(diff_tiles));
    memset(structure_tiles, 0, sizeof(structure_tiles));
    memset(&structure_full, 0, sizeof(structure_full));
    memset(result->band_stack, 0, sizeof(result->band_stack));
    result->band_stack_cached_on_device = 0;

    if (!choose_curve_shift(dataset->efe1_blob, dataset->efe1_blob_size, &curves)) {
        kedge_status_set(status, "Failed to parse material curves from (EFE1,1001).");
        return 0;
    }
    compute_bin_curve_matrix(&curves, result->bin_curve_matrix);
    build_material_vectors(result->bin_curve_matrix, result->material_vectors, result->iodine_kedge_vector, result->gadolinium_kedge_vector);

    marker_count = kedge_find_jp2_markers(dataset->efe1_blob, dataset->efe1_blob_size, markers, 512);
    first_jp2c = find_first_pattern(dataset->efe1_blob, dataset->efe1_blob_size, jp2c_pattern, sizeof(jp2c_pattern));
    first_jp_signature = find_first_pattern(dataset->efe1_blob, dataset->efe1_blob_size, jp_signature_pattern, sizeof(jp_signature_pattern));
    first_ftypjp2 = find_first_pattern(dataset->efe1_blob, dataset->efe1_blob_size, ftypjp2_pattern, sizeof(ftypjp2_pattern));
    first_soc = find_first_pattern(dataset->efe1_blob, dataset->efe1_blob_size, soc_pattern, sizeof(soc_pattern));
    first_siz = find_first_pattern(dataset->efe1_blob, dataset->efe1_blob_size, siz_pattern, sizeof(siz_pattern));
    soc_count = count_pattern_occurrences(dataset->efe1_blob, dataset->efe1_blob_size, soc_pattern, sizeof(soc_pattern));
    siz_count = count_pattern_occurrences(dataset->efe1_blob, dataset->efe1_blob_size, siz_pattern, sizeof(siz_pattern));
    use_marker_path = (marker_count >= 256);
    if (use_marker_path) {
        full_rows = KEDGE_FULL_ROWS;
        full_cols = KEDGE_FULL_COLS;
        low_rows = KEDGE_LOW_ROWS;
        low_cols = KEDGE_LOW_COLS;
    } else {
        full_rows = dataset->rows;
        full_cols = dataset->cols;
        if (full_rows <= 0 || full_cols <= 0 || full_rows < 8 || (full_rows % 8) != 0) {
            kedge_status_set(
                status,
                "Unsupported source dimensions for fallback rows=%d cols=%d.",
                dataset->rows,
                dataset->cols
            );
            return 0;
        }
        low_rows = full_rows / 8;
        low_cols = full_cols;
    }
    fprintf(stderr, "[DEBUG] setup_and_parse took %.4f s\n", get_time_s() - t_init);
    /* #region debug-point fallback-path */
    fprintf(
        stderr,
        "[c-marker-zero] marker_count=%d use_marker_path=%d first_soc=%zu soc_count=%zu first_siz=%zu siz_count=%zu\n",
        marker_count,
        use_marker_path,
        first_soc,
        soc_count,
        first_siz,
        siz_count
    );
    /* #endregion */
    double t_alloc = get_time_s();
    for (i = 0; i < 8; ++i) {
        if (!alloc_image_f64(&band_stack[i], low_cols, low_rows, status)) {
            goto fail;
        }
    }
    if (!alloc_image_f64(&structure_full, full_cols, full_rows, status) ||
        !alloc_image_f64(&result->full_maps[0], full_cols, full_rows, status) ||
        !alloc_image_f64(&result->full_maps[1], full_cols, full_rows, status) ||
        !alloc_image_f64(&result->full_maps[2], full_cols, full_rows, status) ||
        !alloc_image_f64(&result->full_maps[3], full_cols, full_rows, status) ||
        !alloc_image_f64(&result->full_maps[4], full_cols, full_rows, status) ||
        !alloc_image_f64(&result->full_maps[5], full_cols, full_rows, status)) {
        goto fail;
    }
    for (i = 0; i < 6; ++i) {
        if (!alloc_image_f64(&result->coarse_maps[i], low_cols, low_rows, status)) {
            goto fail;
        }
    }
    fprintf(stderr, "[DEBUG] alloc_images took %.4f s\n", get_time_s() - t_alloc);
    double t_start = get_time_s();
    if (!make_double_pixel_data(dataset, &pixel_data, status)) {
        goto fail;
    }
    fprintf(stderr, "[DEBUG] make_double_pixel_data took %.4f s\n", get_time_s() - t_start);
    /* #region debug-point fallback-pixel-data */
    fprintf(stderr, "[c-marker-zero] pixel_data_ready rows=%d cols=%d\n", dataset->rows, dataset->cols);
    /* #endregion */

    double pixel_p50 = 0.0, pixel_p995 = 0.0, pixel_p985 = 0.0;
    {
        double tpct = get_time_s();
        size_t full_count = (size_t)full_rows * full_cols;
        double *tmp = (double *)malloc(full_count * sizeof(double));
        size_t valid_count = 0;
        if (tmp != NULL) {
            for (i = 0; i < full_count; ++i) {
                if (isfinite(pixel_data[i])) {
                    tmp[valid_count++] = pixel_data[i];
                }
            }
            if (valid_count > 0) {
                if (enable_cuda && kedge_cuda_available()) {
                    if (!kedge_cuda_compute_three_percentiles(tmp, valid_count, &pixel_p50, &pixel_p985, &pixel_p995, status)) {
                        // fallback to CPU sort
                        qsort(tmp, valid_count, sizeof(double), compare_double_ascending);
                        goto do_cpu_pct;
                    }
                } else {
do_cpu_pct:
                    qsort(tmp, valid_count, sizeof(double), compare_double_ascending);
                    {
                        double rank, t; size_t lo, hi;
                        rank = (50.0 / 100.0) * (double)(valid_count - 1);
                        lo = (size_t)floor(rank); hi = (size_t)ceil(rank); t = rank - (double)lo;
                        pixel_p50 = tmp[lo] * (1.0 - t) + tmp[hi] * t;
                        
                        rank = (99.5 / 100.0) * (double)(valid_count - 1);
                        lo = (size_t)floor(rank); hi = (size_t)ceil(rank); t = rank - (double)lo;
                        pixel_p995 = tmp[lo] * (1.0 - t) + tmp[hi] * t;
                        
                        rank = (98.5 / 100.0) * (double)(valid_count - 1);
                        lo = (size_t)floor(rank); hi = (size_t)ceil(rank); t = rank - (double)lo;
                        pixel_p985 = tmp[lo] * (1.0 - t) + tmp[hi] * t;
                    }
                }
            }
            free(tmp);
        }
        fprintf(stderr, "[DEBUG] percentiles took %.4f s\n", get_time_s() - tpct);
    }

    double t_lowres = get_time_s();
    lowres_body = (uint8_t *)malloc((size_t)low_rows * low_cols);
    if (lowres_body == NULL || !build_lowres_body(pixel_data, dataset->rows, dataset->cols, pixel_p50, lowres_body)) {
        kedge_status_set(status, "Failed to build low-resolution body mask.");
        goto fail;
    }
    fprintf(stderr, "[DEBUG] build_lowres_body took %.4f s\n", get_time_s() - t_lowres);
    /* #region debug-point fallback-lowres-body */
    fprintf(stderr, "[c-marker-zero] lowres_body_ready\n");
    /* #endregion */

    dump_stage_mask_u8(dataset->source_path, "stage00_lowres_body", lowres_body, (uint32_t)low_rows, (uint32_t)low_cols);
    if (use_marker_path) {
        int use_cuda = 0;
#ifdef KEDGE_USE_CUDA
        use_cuda = enable_cuda && kedge_cuda_available();
#endif
        double td0 = get_time_s();
        if (!kedge_decode_all_groups_parallel(
            use_cuda,
            dataset->efe1_blob,
            dataset->efe1_blob_size,
            markers,
            marker_count,
            temp_dir,
            g0_tiles,
            g1_tiles,
            g2_tiles,
            g3_tiles,
            status
        )) {
            if (use_cuda) {
                fprintf(stderr, "[DEBUG] nvJPEG2000 hardware decode_all_groups failed, falling back to CPU OpenJPEG.\n");
                kedge_status_reset(status);
                if (!kedge_decode_all_groups_parallel(
                    0,
                    dataset->efe1_blob,
                    dataset->efe1_blob_size,
                    markers,
                    marker_count,
                    temp_dir,
                    g0_tiles,
                    g1_tiles,
                    g2_tiles,
                    g3_tiles,
                    status
                )) {
                    goto fail;
                }
            } else {
                goto fail;
            }
        }
        fprintf(stderr, "[DEBUG] decode_all_groups took %.4f s\n", get_time_s() - td0);
        if (enable_cuda) {
            double tr0 = get_time_s();
            if (!kedge_cuda_marker_tile_reconstruct(
                g0_tiles,
                g1_tiles,
                g2_tiles,
                g3_tiles,
                &structure_full,
                band_stack,
                !use_cuda_fast_math,
                status
            )) {
                goto fail;
            }
            result->band_stack_cached_on_device = use_cuda_fast_math ? 1 : 0;
            fprintf(stderr, "[DEBUG] marker_tile_reconstruct took %.4f s\n", get_time_s() - tr0);
        } else {
            for (i = 0; i < KEDGE_TILE_COUNT; ++i) {
                size_t pixel_count;
                size_t idx;
                if (!alloc_image_f64(&diff_tiles[i], KEDGE_TILE_SIDE, KEDGE_TILE_SIDE, status) ||
                    !alloc_image_f64(&structure_tiles[i], KEDGE_TILE_SIDE, KEDGE_TILE_SIDE, status)) {
                    goto fail;
                }
                pixel_count = (size_t)KEDGE_TILE_SIDE * KEDGE_TILE_SIDE;
                for (idx = 0; idx < pixel_count; ++idx) {
                    diff_tiles[i].data[idx] = g1_tiles[i].data[idx] - g0_tiles[i].data[idx];
                    structure_tiles[i].data[idx] = g3_tiles[i].data[idx] - g2_tiles[i].data[idx];
                }
            }
            assemble_interleaved(structure_tiles, &structure_full);
            assemble_band_stack(diff_tiles, band_stack);
            result->band_stack_cached_on_device = 0;
        }
        /* #region debug-point marker-decode-ready */
        fprintf(stderr, "[c-marker-zero] marker_decode_ready\n");
        /* #endregion */
        dump_stage_image_f64(dataset->source_path, "stage01_structure_full_raw", &structure_full);
        if (use_cuda_fast_math && stage_dump_enabled()) {
            if (!kedge_cuda_sync_band_stack_from_cache(band_stack, status)) {
                goto fail;
            }
        }
        dump_stage_image_array_f64(dataset->source_path, "stage01_band_stack_raw", band_stack, 8U);
    } else {
        if (!build_pixel_energy_surrogate_stack(
            dataset,
            pixel_data,
            lowres_body,
            result->bin_curve_matrix,
            &structure_full,
            band_stack,
            status
        )) {
            goto fail;
        }
        result->band_stack_cached_on_device = 0;
        /* #region debug-point fallback-band-stack */
        fprintf(stderr, "[c-marker-zero] fallback_band_stack_ready\n");
        /* #endregion */
        dump_stage_image_f64(dataset->source_path, "stage01_structure_full_raw", &structure_full);
        dump_stage_image_array_f64(dataset->source_path, "stage01_band_stack_raw", band_stack, 8U);
    }
    if (use_cuda_fast_math) {
        double t_suppress = get_time_s();
        int suppress_row_step;
        int suppress_col_step;
        adaptive_lowres_block_shape(low_rows, low_cols, &suppress_row_step, &suppress_col_step);
        if (!kedge_cuda_suppress_cached_band_stack(
            lowres_body,
            low_rows,
            low_cols,
            suppress_row_step,
            suppress_col_step,
            kedge_projection_block_suppress_strength,
            status
        )) {
            goto fail;
        }
        if (stage_dump_enabled()) {
            if (!kedge_cuda_sync_band_stack_from_cache(band_stack, status)) {
                goto fail;
            }
        }
        fprintf(stderr, "[DEBUG] suppress_cached_band_stack took %.4f s\n", get_time_s() - t_suppress);
    } else {
        suppress_band_stack_block_artifacts(enable_cuda, band_stack, lowres_body, status);
    }
    /* #region debug-point post-suppress */
    fprintf(stderr, "[c-marker-zero] band_stack_suppressed\n");
    /* #endregion */
    dump_stage_image_array_f64(dataset->source_path, "stage02_band_stack_suppressed", band_stack, 8U);

    if (use_cuda_fast_math) {
        double t_proj = get_time_s();
        if (!kedge_cuda_project_6_maps_from_cache(
            result->material_vectors,
            result->iodine_kedge_vector,
            result->gadolinium_kedge_vector,
            low_rows,
            low_cols,
            result->coarse_maps,
            status
        )) {
            goto fail;
        }
        fprintf(stderr, "[DEBUG] project_6_maps_from_cache took %.4f s\n", get_time_s() - t_proj);
    } else {
        compute_direct_low(band_stack, result->material_vectors[KEDGE_MATERIAL_WATER], &result->coarse_maps[0]);
        compute_direct_low(band_stack, result->material_vectors[KEDGE_MATERIAL_IODINE], &result->coarse_maps[1]);
        compute_direct_low(band_stack, result->material_vectors[KEDGE_MATERIAL_CALCIUM], &result->coarse_maps[2]);
        compute_direct_low(band_stack, result->material_vectors[KEDGE_MATERIAL_GADOLINIUM], &result->coarse_maps[3]);
        compute_direct_low(band_stack, result->iodine_kedge_vector, &result->coarse_maps[4]);
        compute_direct_low(band_stack, result->gadolinium_kedge_vector, &result->coarse_maps[5]);
    }
    /* #region debug-point direct-low-ready */
    fprintf(stderr, "[c-marker-zero] direct_low_ready\n");
    /* #endregion */
    dump_stage_image_array_f64(dataset->source_path, "stage03_coarse_direct", result->coarse_maps, 6U);

    prior_density_norm = (double *)malloc((size_t)low_rows * low_cols * sizeof(double));
    prior_structure_norm = (double *)malloc((size_t)low_rows * low_cols * sizeof(double));
    fusion_body = (uint8_t *)malloc((size_t)full_rows * full_cols);
    fusion_base_norm = (double *)malloc((size_t)full_rows * full_cols * sizeof(double));
    if (prior_density_norm == NULL || prior_structure_norm == NULL || fusion_body == NULL || fusion_base_norm == NULL) {
        kedge_status_set(status, "Out of memory while preparing shared decomposition buffers.");
        goto fail;
    }
    double t_prep1 = get_time_s();
    if (!prepare_lowres_prior_static(
        enable_cuda,
        pixel_data,
        structure_full.data,
        lowres_body,
        full_rows,
        full_cols,
        low_rows,
        low_cols,
        pixel_p50,
        pixel_p995,
        pixel_p985,
        prior_density_norm,
        prior_structure_norm,
        status
    )) {
        goto fail;
    }
    fprintf(stderr, "[DEBUG] prepare_lowres_prior_static took %.4f s\n", get_time_s() - t_prep1);
    double t_prep2 = get_time_s();
    if (!prepare_fullres_fusion_static(
        enable_cuda,
        structure_full.data,
        pixel_data,
        full_rows,
        full_cols,
        pixel_p50,
        pixel_p995,
        pixel_p985,
        fusion_body,
        fusion_base_norm,
        status
    )) {
        goto fail;
    }
    fprintf(stderr, "[DEBUG] prepare_fullres_fusion_static took %.4f s\n", get_time_s() - t_prep2);

    for (i = 0; i < 6; ++i) {
        double tl0 = get_time_s();
        if (!build_spectral_prior_lowres_cuda_aware(
            enable_cuda,
            result->coarse_maps[i].data,
            pixel_data,
            lowres_body,
            full_rows,
            full_cols,
            low_rows,
            low_cols,
            pixel_p50,
            pixel_p995,
            pixel_p985,
            prior_density_norm,
            prior_structure_norm,
            result->coarse_maps[i].data,
            status
        )) {
            goto fail;
        }
        double tl1 = get_time_s();
        if (!fuse_structure_map_cuda_aware(
            enable_cuda,
            result->coarse_maps[i].data,
            pixel_data,
            fusion_body,
            fusion_base_norm,
            full_rows,
            full_cols,
            low_rows,
            low_cols,
            pixel_p50,
            pixel_p995,
            pixel_p985,
            result->full_maps[i].data,
            status
        )) {
            goto fail;
        }
        double tl2 = get_time_s();
        /* #region debug-point full-map-loop */
        fprintf(stderr, "[c-marker-zero] full_map_ready index=%d (prior=%.4fs, fuse=%.4fs)\n", i, tl1-tl0, tl2-tl1);
        /* #endregion */
    }
    dump_stage_image_array_f64(dataset->source_path, "stage04_coarse_postprior", result->coarse_maps, 6U);
    dump_stage_image_array_f64(dataset->source_path, "stage05_full_maps_final", result->full_maps, 6U);

    free(prior_density_norm);
    free(prior_structure_norm);
    free(fusion_body);
    free(fusion_base_norm);
    free(pixel_data);
    free(lowres_body);
    for (i = 0; i < KEDGE_TILE_COUNT; ++i) {
        free_image_f64(&g0_tiles[i]);
        free_image_f64(&g1_tiles[i]);
        free_image_f64(&g2_tiles[i]);
        free_image_f64(&g3_tiles[i]);
        free_image_f64(&diff_tiles[i]);
        free_image_f64(&structure_tiles[i]);
    }
    free_image_f64(&structure_full);
    return 1;

fail:
    free(prior_density_norm);
    free(prior_structure_norm);
    free(fusion_body);
    free(fusion_base_norm);
    free(pixel_data);
    free(lowres_body);
    for (i = 0; i < KEDGE_TILE_COUNT; ++i) {
        free_image_f64(&g0_tiles[i]);
        free_image_f64(&g1_tiles[i]);
        free_image_f64(&g2_tiles[i]);
        free_image_f64(&g3_tiles[i]);
        free_image_f64(&diff_tiles[i]);
        free_image_f64(&structure_tiles[i]);
    }
    free_image_f64(&structure_full);
    kedge_free_decomposition(result);
    return 0;
}

int kedge_compute_decomposition_cpu(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
) {
    return kedge_compute_decomposition_impl(dataset, temp_dir, 0, result, status);
}

int kedge_compute_decomposition_cuda(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
) {
    return kedge_compute_decomposition_impl(dataset, temp_dir, 1, result, status);
}

int kedge_compute_decomposition_cuda_full(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
) {
    return kedge_compute_decomposition_impl(dataset, temp_dir, 2, result, status);
}

int kedge_compute_decomposition_with_backend(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    KEdgeComputeBackend backend,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
) {
    KEdgeStatus cuda_status;
    if (backend == KEDGE_BACKEND_CPU) {
        return kedge_compute_decomposition_cpu(dataset, temp_dir, result, status);
    }
    if (backend == KEDGE_BACKEND_CUDA) {
        if (!kedge_cuda_available()) {
            kedge_status_set(status, "CUDA backend requested but CUDA runtime/device is unavailable.");
            return 0;
        }
        return kedge_compute_decomposition_cuda(dataset, temp_dir, result, status);
    }
    if (backend == KEDGE_BACKEND_CUDA_FULL) {
        if (!kedge_cuda_available()) {
            kedge_status_set(status, "CUDA_FULL backend requested but CUDA runtime/device is unavailable.");
            return 0;
        }
        return kedge_compute_decomposition_cuda_full(dataset, temp_dir, result, status);
    }
    if (kedge_cuda_available()) {
        kedge_status_reset(&cuda_status);
        if (kedge_compute_decomposition_cuda(dataset, temp_dir, result, &cuda_status)) {
            return 1;
        }
    }
    return kedge_compute_decomposition_cpu(dataset, temp_dir, result, status);
}

int kedge_compute_decomposition(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
) {
    return kedge_compute_decomposition_with_backend(dataset, temp_dir, KEDGE_BACKEND_AUTO, result, status);
}

void kedge_free_decomposition(KEdgeDecompositionResult *result) {
    int i;
    if (result == NULL) {
        return;
    }
    for (i = 0; i < 6; ++i) {
        free_image_f64(&result->full_maps[i]);
        free_image_f64(&result->coarse_maps[i]);
    }
    for (i = 0; i < 8; ++i) {
        free_image_f64(&result->band_stack[i]);
    }
    memset(result, 0, sizeof(*result));
}
