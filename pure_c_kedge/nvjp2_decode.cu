#include "kedge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nvjpeg2k.h>

#ifdef _WIN32
#include <windows.h>
static double nvjp2_get_time_s(void) {
    LARGE_INTEGER freq;
    LARGE_INTEGER time;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double nvjp2_get_time_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

// CUDA kernel to convert uint8, uint16, or int16 to double
__global__ void convert_to_double_kernel(const void* src, double* dst, size_t total_pixels, int type_code) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        if (type_code == 0) {
            dst[idx] = (double)(((const uint8_t*)src)[idx]);
        } else if (type_code == 1) {
            dst[idx] = (double)(((const int16_t*)src)[idx]);
        } else if (type_code == 2) {
            dst[idx] = (double)(((const uint16_t*)src)[idx]);
        }
    }
}

#define NVJPEG2K_MAX_HANDLES 32

// Global nvJPEG2000 handle
static nvjpeg2kHandle_t g_nvjpeg2k_handle = NULL;
static cudaStream_t g_cu_streams[NVJPEG2K_MAX_HANDLES];
static nvjpeg2kStream_t g_streams[NVJPEG2K_MAX_HANDLES];
static nvjpeg2kDecodeState_t g_decode_states[NVJPEG2K_MAX_HANDLES];
static int g_nvjpeg2k_initialized = 0;

int kedge_init_nvjpeg2k(KEdgeStatus *status) {
    if (g_nvjpeg2k_initialized) return 1;
    
    nvjpeg2kStatus_t st = nvjpeg2kCreateSimple(&g_nvjpeg2k_handle);
    if (st != NVJPEG2K_STATUS_SUCCESS) {
        kedge_status_set(status, "nvJPEG2000: Failed to create handle (error %d)", st);
        return 0;
    }

    for (int i = 0; i < NVJPEG2K_MAX_HANDLES; i++) {
        g_cu_streams[i] = NULL;
        g_streams[i] = NULL;
        g_decode_states[i] = NULL;
        if (cudaStreamCreateWithFlags(&g_cu_streams[i], cudaStreamNonBlocking) != cudaSuccess) {
            return 0;
        }
        if (nvjpeg2kStreamCreate(&g_streams[i]) != NVJPEG2K_STATUS_SUCCESS) {
            return 0;
        }
        if (nvjpeg2kDecodeStateCreate(g_nvjpeg2k_handle, &g_decode_states[i]) != NVJPEG2K_STATUS_SUCCESS) {
            return 0;
        }
    }

    g_nvjpeg2k_initialized = 1;
    return 1;
}

void cleanup_nvjpeg2k() {
    if (g_nvjpeg2k_initialized) {
        for (int i = 0; i < NVJPEG2K_MAX_HANDLES; i++) {
            if (g_decode_states[i]) { nvjpeg2kDecodeStateDestroy(g_decode_states[i]); g_decode_states[i] = NULL; }
            if (g_streams[i]) { nvjpeg2kStreamDestroy(g_streams[i]); g_streams[i] = NULL; }
            if (g_cu_streams[i]) { cudaStreamDestroy(g_cu_streams[i]); g_cu_streams[i] = NULL; }
        }
        if (g_nvjpeg2k_handle != NULL) {
            nvjpeg2kDestroy(g_nvjpeg2k_handle);
            g_nvjpeg2k_handle = NULL;
        }
        g_nvjpeg2k_initialized = 0;
    }
}

// Decodes a JPEG2000 bitstream in memory to a host double array (to match OpenJPEG signature).
// We decode to device memory using nvJPEG2000, then convert to double and copy back.
int kedge_decode_nvjpeg2k_image_from_memory(const unsigned char *data, size_t size, KEdgeImageF64 *image, KEdgeStatus *status) {
    if (!kedge_init_nvjpeg2k(status)) {
        return 0;
    }

    cudaStream_t cu_stream;
    if (cudaStreamCreateWithFlags(&cu_stream, cudaStreamNonBlocking) != cudaSuccess) {
        kedge_status_set(status, "nvJPEG2000: cudaStreamCreateWithFlags failed");
        return 0;
    }

    nvjpeg2kStatus_t st;
    nvjpeg2kStream_t stream = NULL;
    nvjpeg2kDecodeState_t decode_state = NULL;
    
    st = nvjpeg2kStreamCreate(&stream);
    if (st != NVJPEG2K_STATUS_SUCCESS) {
        kedge_status_set(status, "nvJPEG2000: nvjpeg2kStreamCreate failed");
        cudaStreamDestroy(cu_stream);
        return 0;
    }

    st = nvjpeg2kStreamParse(g_nvjpeg2k_handle, data, size, 0, 0, stream);
    if (st != NVJPEG2K_STATUS_SUCCESS) {
        kedge_status_set(status, "nvJPEG2000: nvjpeg2kStreamParse failed");
        nvjpeg2kStreamDestroy(stream);
        cudaStreamDestroy(cu_stream);
        return 0;
    }

    nvjpeg2kImageInfo_t info;
    st = nvjpeg2kStreamGetImageInfo(stream, &info);
    if (st != NVJPEG2K_STATUS_SUCCESS) {
        kedge_status_set(status, "nvJPEG2000: nvjpeg2kStreamGetImageInfo failed");
        nvjpeg2kStreamDestroy(stream);
        cudaStreamDestroy(cu_stream);
        return 0;
    }

    nvjpeg2kImageComponentInfo_t comp_info;
    st = nvjpeg2kStreamGetImageComponentInfo(stream, &comp_info, 0);
    if (st != NVJPEG2K_STATUS_SUCCESS) {
        kedge_status_set(status, "nvJPEG2000: nvjpeg2kStreamGetImageComponentInfo failed");
        nvjpeg2kStreamDestroy(stream);
        cudaStreamDestroy(cu_stream);
        return 0;
    }

    st = nvjpeg2kDecodeStateCreate(g_nvjpeg2k_handle, &decode_state);
    if (st != NVJPEG2K_STATUS_SUCCESS) {
        kedge_status_set(status, "nvJPEG2000: nvjpeg2kDecodeStateCreate failed");
        nvjpeg2kStreamDestroy(stream);
        cudaStreamDestroy(cu_stream);
        return 0;
    }

    // Allocate device memory for decode output
    nvjpeg2kImage_t decode_output;
    memset(&decode_output, 0, sizeof(decode_output));
    
    size_t bytes_per_sample = 1;
    int type_code = 0; // 0: uint8, 1: int16, 2: uint16
    if (comp_info.precision > 8) {
        bytes_per_sample = 2;
        if (comp_info.sgn) {
            decode_output.pixel_type = NVJPEG2K_INT16;
            type_code = 1;
        } else {
            decode_output.pixel_type = NVJPEG2K_UINT16;
            type_code = 2;
        }
    } else {
        decode_output.pixel_type = NVJPEG2K_UINT8;
        type_code = 0;
    }

    decode_output.num_components = 1; // DICOM usually 1 for Kedge
    void *d_pixel_data = NULL;
    double *d_double_data = NULL;
    size_t pitch = info.image_width * bytes_per_sample;
    size_t total_pixels = info.image_width * info.image_height;
    
    cudaError_t cu_st = cudaMallocAsync(&d_pixel_data, pitch * info.image_height, cu_stream);
    if (cu_st != cudaSuccess) {
        kedge_status_set(status, "nvJPEG2000: cudaMallocAsync failed");
        nvjpeg2kDecodeStateDestroy(decode_state);
        nvjpeg2kStreamDestroy(stream);
        cudaStreamDestroy(cu_stream);
        return 0;
    }

    cu_st = cudaMallocAsync(&d_double_data, total_pixels * sizeof(double), cu_stream);
    if (cu_st != cudaSuccess) {
        kedge_status_set(status, "nvJPEG2000: cudaMallocAsync failed for double buffer");
        cudaFreeAsync(d_pixel_data, cu_stream);
        nvjpeg2kDecodeStateDestroy(decode_state);
        nvjpeg2kStreamDestroy(stream);
        cudaStreamDestroy(cu_stream);
        return 0;
    }

    void *d_pixel_data_arr[1] = { d_pixel_data };
    size_t pitch_arr[1] = { pitch };

    decode_output.pixel_data = d_pixel_data_arr;
    decode_output.pitch_in_bytes = pitch_arr;

    st = nvjpeg2kDecode(g_nvjpeg2k_handle, decode_state, stream, &decode_output, cu_stream);
    if (st != NVJPEG2K_STATUS_SUCCESS) {
        kedge_status_set(status, "nvJPEG2000: nvjpeg2kDecode failed");
        cudaFreeAsync(d_double_data, cu_stream);
        cudaFreeAsync(d_pixel_data, cu_stream);
        nvjpeg2kDecodeStateDestroy(decode_state);
        nvjpeg2kStreamDestroy(stream);
        cudaStreamDestroy(cu_stream);
        return 0;
    }

    // Launch conversion kernel
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    convert_to_double_kernel<<<grid_size, block_size, 0, cu_stream>>>(d_pixel_data, d_double_data, total_pixels, type_code);

    // Convert to double to match the OpenJPEG wrapper interface
    image->width = info.image_width;
    image->height = info.image_height;
    image->data = (double *)malloc(total_pixels * sizeof(double));
    
    if (image->data == NULL) {
        kedge_status_set(status, "nvJPEG2000: failed to allocate image->data");
        cudaFreeAsync(d_double_data, cu_stream);
        cudaFreeAsync(d_pixel_data, cu_stream);
        nvjpeg2kDecodeStateDestroy(decode_state);
        nvjpeg2kStreamDestroy(stream);
        cudaStreamDestroy(cu_stream);
        return 0;
    }

    cudaMemcpyAsync(image->data, d_double_data, total_pixels * sizeof(double), cudaMemcpyDeviceToHost, cu_stream);
    cudaStreamSynchronize(cu_stream);

    cudaFreeAsync(d_double_data, cu_stream);
    cudaFreeAsync(d_pixel_data, cu_stream);
    nvjpeg2kDecodeStateDestroy(decode_state);
    nvjpeg2kStreamDestroy(stream);
    cudaStreamDestroy(cu_stream);

    return 1;
}

// Decodes multiple JPEG2000 bitstreams concurrently
int kedge_decode_nvjpeg2k_batched(
    int count,
    const unsigned char **data_ptrs,
    const size_t *data_sizes,
    KEdgeImageF64 **out_images,
    KEdgeStatus *status
) {
    if (!kedge_init_nvjpeg2k(status)) {
        return 0;
    }

    if (count > NVJPEG2K_MAX_HANDLES) {
        kedge_status_set(status, "nvJPEG2000 batched: Count exceeds maximum handles (%d)", NVJPEG2K_MAX_HANDLES);
        return 0;
    }

    int i;
    int success = 1;
    double t0 = nvjp2_get_time_s();
    double t1 = 0, t2 = 0, t3 = 0;void **d_pixel_datas = (void **)malloc(count * sizeof(void *));
    void **h_pixel_datas = (void **)malloc(count * sizeof(void *));
    size_t *pitches = (size_t *)malloc(count * sizeof(size_t));
    int *bytes_per_samples = (int *)malloc(count * sizeof(int));
    int *is_signeds = (int *)malloc(count * sizeof(int));
    int *widths = (int *)malloc(count * sizeof(int));
    int *heights = (int *)malloc(count * sizeof(int));
    
    memset(d_pixel_datas, 0, count * sizeof(void *));
    memset(h_pixel_datas, 0, count * sizeof(void *));
    
    t1 = nvjp2_get_time_s();

    // Process in batches of NVJPEG2K_MAX_HANDLES
    for (int batch_start = 0; batch_start < count && success; batch_start += NVJPEG2K_MAX_HANDLES) {
        int batch_size = count - batch_start;
        if (batch_size > NVJPEG2K_MAX_HANDLES) batch_size = NVJPEG2K_MAX_HANDLES;

        // 1. Parse, Allocate, and Decode
        for (int j = 0; j < batch_size; ++j) {
            int i = batch_start + j;
            int sid = j;
            
            nvjpeg2kStatus_t st = nvjpeg2kStreamParse(g_nvjpeg2k_handle, data_ptrs[i], data_sizes[i], 0, 0, g_streams[sid]);
            if (st != NVJPEG2K_STATUS_SUCCESS) { success = 0; break; }
            
            nvjpeg2kImageInfo_t info;
            st = nvjpeg2kStreamGetImageInfo(g_streams[sid], &info);
            if (st != NVJPEG2K_STATUS_SUCCESS) { success = 0; break; }
            
            nvjpeg2kImageComponentInfo_t comp_info;
            st = nvjpeg2kStreamGetImageComponentInfo(g_streams[sid], &comp_info, 0);
            if (st != NVJPEG2K_STATUS_SUCCESS) { success = 0; break; }
            
            int bytes_per_sample = (comp_info.precision > 8) ? 2 : 1;
            bytes_per_samples[i] = bytes_per_sample;
            is_signeds[i] = comp_info.sgn;
            widths[i] = info.image_width;
            heights[i] = info.image_height;
            pitches[i] = info.image_width * bytes_per_sample;
            
            if (cudaMallocAsync(&d_pixel_datas[i], pitches[i] * info.image_height, g_cu_streams[sid]) != cudaSuccess) {
                success = 0; break;
            }
            
            out_images[i]->width = info.image_width;
            out_images[i]->height = info.image_height;
            out_images[i]->data = (double *)malloc(info.image_width * info.image_height * sizeof(double));
            if (!out_images[i]->data) { success = 0; break; }
            
            nvjpeg2kImage_t decode_output;
            memset(&decode_output, 0, sizeof(decode_output));
            
            int type_code = 0;
            if (bytes_per_samples[i] > 1) {
                decode_output.pixel_type = is_signeds[i] ? NVJPEG2K_INT16 : NVJPEG2K_UINT16;
                type_code = is_signeds[i] ? 1 : 2;
            } else {
                decode_output.pixel_type = NVJPEG2K_UINT8;
                type_code = 0;
            }
            decode_output.num_components = 1;
            
            void *d_arr[1] = { d_pixel_datas[i] };
            size_t p_arr[1] = { pitches[i] };
            decode_output.pixel_data = d_arr;
            decode_output.pitch_in_bytes = p_arr;
            
            st = nvjpeg2kDecode(g_nvjpeg2k_handle, g_decode_states[sid], g_streams[sid], &decode_output, g_cu_streams[sid]);
            if (st != NVJPEG2K_STATUS_SUCCESS) {
                success = 0;
                break;
            }

            size_t total_pixels = widths[i] * heights[i];
            void *d_double_data = NULL;
            if (cudaMallocAsync(&d_double_data, total_pixels * sizeof(double), g_cu_streams[sid]) != cudaSuccess) {
                success = 0; break;
            }
            h_pixel_datas[i] = d_double_data;

            int block_size = 256;
            int grid_size = (total_pixels + block_size - 1) / block_size;
            convert_to_double_kernel<<<grid_size, block_size, 0, g_cu_streams[sid]>>>(d_pixel_datas[i], (double*)d_double_data, total_pixels, type_code);
            
            cudaMemcpyAsync(out_images[i]->data, h_pixel_datas[i], total_pixels * sizeof(double), cudaMemcpyDeviceToHost, g_cu_streams[sid]);
        }
        
        // 2. Synchronize batch
        for (int j = 0; j < batch_size; ++j) {
            cudaStreamSynchronize(g_cu_streams[j]);
        }
    }
    
    if (!success) {
        kedge_status_set(status, "nvJPEG2000 batched: Decode failed");
        goto cleanup;
    }
    
    t3 = nvjp2_get_time_s();
    fprintf(stderr, "[DEBUG] batched parallel decode/copy: %.4f\n", t3 - t1);

cleanup:
    for (i = 0; i < count; ++i) {
        int sid = i % NVJPEG2K_MAX_HANDLES;
        if (d_pixel_datas[i]) cudaFreeAsync(d_pixel_datas[i], g_cu_streams[sid]);
        if (h_pixel_datas[i]) cudaFreeAsync(h_pixel_datas[i], g_cu_streams[sid]);
        
        if (!success && out_images[i] && out_images[i]->data) {
            free(out_images[i]->data);
            out_images[i]->data = NULL;
        }
    }
    
    free(d_pixel_datas);
    free(h_pixel_datas);
    free(pitches);
    free(bytes_per_samples);
    free(is_signeds);
    free(widths);
    free(heights);
    
    return success;
}

#ifdef __cplusplus
}
#endif
