#ifndef PURE_C_KEDGE_H
#define PURE_C_KEDGE_H

#include <stddef.h>
#include <stdint.h>

#define KEDGE_MAX_UID 128
#define KEDGE_MAX_TEXT 256
#define KEDGE_CURVE_POINT_COUNT 200
#define KEDGE_BIN_COUNT 8
#define KEDGE_MATERIAL_COUNT 4
#define KEDGE_TILE_COUNT 60
#define KEDGE_TILE_SIDE 256
#define KEDGE_TILE_BAND_HEIGHT 32
#define KEDGE_FULL_ROWS 2048
#define KEDGE_FULL_COLS 2048
#define KEDGE_LOW_ROWS 256
#define KEDGE_LOW_COLS 2048
#define KEDGE_PATH_CAP 1024

enum {
    KEDGE_MATERIAL_WATER = 0,
    KEDGE_MATERIAL_IODINE = 1,
    KEDGE_MATERIAL_CALCIUM = 2,
    KEDGE_MATERIAL_GADOLINIUM = 3
};

typedef struct {
    size_t width;
    size_t height;
    double *data;
} KEdgeImageF64;

typedef struct {
    size_t width;
    size_t height;
    int16_t *data;
} KEdgeImageI16;

typedef struct {
    double materials[KEDGE_MATERIAL_COUNT][KEDGE_CURVE_POINT_COUNT];
    int selected_shift;
} KEdgeCurves;

typedef struct {
    KEdgeImageF64 full_maps[6];
    KEdgeImageF64 coarse_maps[6];
    KEdgeImageF64 band_stack[8];
    int band_stack_cached_on_device;
    double bin_curve_matrix[KEDGE_BIN_COUNT][KEDGE_MATERIAL_COUNT];
    double material_vectors[KEDGE_MATERIAL_COUNT][KEDGE_BIN_COUNT];
    double iodine_kedge_vector[KEDGE_BIN_COUNT];
    double gadolinium_kedge_vector[KEDGE_BIN_COUNT];
} KEdgeDecompositionResult;

typedef struct {
    char source_path[KEDGE_PATH_CAP];
    char transfer_syntax_uid[KEDGE_MAX_UID];
    char study_instance_uid[KEDGE_MAX_UID];
    char series_instance_uid[KEDGE_MAX_UID];
    char sop_instance_uid[KEDGE_MAX_UID];
    char patient_name[KEDGE_MAX_TEXT];
    char patient_id[KEDGE_MAX_TEXT];
    char study_date[32];
    char study_time[32];
    char photometric_interpretation[64];
    char series_description[KEDGE_MAX_TEXT];
    int rows;
    int cols;
    int bits_allocated;
    int bits_stored;
    int high_bit;
    int pixel_representation;
    int samples_per_pixel;
    int instance_number;
    int have_image_position_patient;
    int have_slice_location;
    double rescale_slope;
    double rescale_intercept;
    double image_position_patient[3];
    double slice_location;
    int16_t *pixel_data_i16;
    size_t pixel_value_count;
    unsigned char *efe1_blob;
    size_t efe1_blob_size;
} KEdgeDicomDataset;

typedef struct {
    char error[512];
} KEdgeStatus;

typedef enum {
    KEDGE_BACKEND_CPU = 0,
    KEDGE_BACKEND_CUDA = 1,
    KEDGE_BACKEND_AUTO = 2,
    KEDGE_BACKEND_CUDA_FULL = 3
} KEdgeComputeBackend;

#ifdef __cplusplus
extern "C" {
#endif

extern const char *kedge_material_names[KEDGE_MATERIAL_COUNT];
extern const char *kedge_output_keys[6];

int kedge_dicom_read_file(const char *path, KEdgeDicomDataset *dataset, KEdgeStatus *status);
void kedge_dicom_free(KEdgeDicomDataset *dataset);

int kedge_find_jp2_markers(const unsigned char *blob, size_t blob_size, size_t *markers, size_t max_markers);
int kedge_decode_openjpeg_image_from_memory(
    const unsigned char *data,
    size_t data_size,
    KEdgeImageF64 *image,
    KEdgeStatus *status
);

int kedge_decode_nvjpeg2k_image_from_memory(
    const unsigned char *data,
    size_t size,
    KEdgeImageF64 *image,
    KEdgeStatus *status
);

int kedge_init_nvjpeg2k(KEdgeStatus *status);
int kedge_cuda_compute_three_percentiles(
    const double *values,
    size_t count,
    double *p50,
    double *p985,
    double *p995,
    KEdgeStatus *status
);
int kedge_decode_group_tiles(
    const unsigned char *blob,
    size_t blob_size,
    const size_t *markers,
    size_t marker_count,
    int group_id,
    const char *temp_dir,
    KEdgeImageF64 *tiles,
    KEdgeStatus *status
);
int kedge_decode_all_groups_parallel(
    int enable_cuda,
    const unsigned char *blob,
    size_t blob_size,
    const size_t *markers,
    size_t marker_count,
    const char *temp_dir,
    KEdgeImageF64 *g0_tiles,
    KEdgeImageF64 *g1_tiles,
    KEdgeImageF64 *g2_tiles,
    KEdgeImageF64 *g3_tiles,
    KEdgeStatus *status
);

int kedge_decode_nvjpeg2k_batched(
    int count,
    const unsigned char **data_ptrs,
    const size_t *data_sizes,
    KEdgeImageF64 **out_images,
    KEdgeStatus *status
);

int kedge_compute_decomposition(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
);
int kedge_compute_decomposition_cpu(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
);
int kedge_compute_decomposition_cuda(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
);
int kedge_compute_decomposition_cuda_full(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
);
int kedge_compute_decomposition_with_backend(
    const KEdgeDicomDataset *dataset,
    const char *temp_dir,
    KEdgeComputeBackend backend,
    KEdgeDecompositionResult *result,
    KEdgeStatus *status
);
int kedge_cuda_available(void);
void kedge_free_decomposition(KEdgeDecompositionResult *result);

int kedge_convert_map_to_int16(const KEdgeImageF64 *source, KEdgeImageI16 *dest, double *slope_out, KEdgeStatus *status);
void kedge_free_image_i16(KEdgeImageI16 *image);

int kedge_export_derived_dicom(
    const KEdgeDicomDataset *source,
    const KEdgeImageI16 *pixel_array,
    const char *output_path,
    const char *series_desc,
    const char *derivation_desc,
    double slope,
    KEdgeStatus *status
);

void kedge_status_reset(KEdgeStatus *status);
void kedge_status_set(KEdgeStatus *status, const char *fmt, ...);
int kedge_make_directory(const char *path);
int kedge_path_join(char *buffer, size_t buffer_size, const char *left, const char *right);
int kedge_write_text_file(const char *path, const char *text, KEdgeStatus *status);
double kedge_percentile(double *values, size_t count, double pct);
void kedge_generate_uid(char *buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif
