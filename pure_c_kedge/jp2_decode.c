#include "kedge.h"

#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include <openjpeg.h>

static const int kedge_empty_tiles[4] = {0, 7, 56, 63};

static int is_empty_tile_position(int tile_pos) {
    size_t i;
    for (i = 0; i < 4; ++i) {
        if (kedge_empty_tiles[i] == tile_pos) {
            return 1;
        }
    }
    return 0;
}

static void opj_error_callback(const char *msg, void *client_data) {
    KEdgeStatus *status = (KEdgeStatus *)client_data;
    if (status != NULL && msg != NULL && status->error[0] == '\0') {
        kedge_status_set(status, "OpenJPEG error: %s", msg);
    }
}

static void free_image_f64(KEdgeImageF64 *image) {
    if (image == NULL) {
        return;
    }
    free(image->data);
    image->data = NULL;
    image->width = 0;
    image->height = 0;
}

static int kedge_marker_tile_worker_count(void) {
    const char *text = getenv("KEDGE_MARKER_TILE_THREADS");
    int value = 0;
    if (text == NULL || text[0] == '\0') {
        text = getenv("KEDGE_MARKER_GROUP_THREADS");
    }
    if (text == NULL || text[0] == '\0') {
        text = getenv("NUMBER_OF_PROCESSORS");
    }
    if (text != NULL && text[0] != '\0') {
        value = atoi(text);
    }
    if (value <= 0) {
        value = 4;
    }
    if (value > 8) {
        value = 8;
    }
    return value;
}

static int kedge_openjpeg_thread_count(void) {
    const char *text = getenv("KEDGE_OPENJPEG_THREADS");
    int value = 0;
    if (text != NULL && text[0] != '\0') {
        value = atoi(text);
    } else {
        value = (kedge_marker_tile_worker_count() > 1) ? 1 : 2;
    }
    if (value <= 0) {
        value = 1;
    }
    if (value > 8) {
        value = 8;
    }
    return value;
}

typedef struct {
    const unsigned char *data;
    OPJ_SIZE_T size;
    OPJ_SIZE_T offset;
} KEdgeMemoryStream;

typedef struct {
    opj_dparameters_t parameters;
    int decode_threads;
} KEdgeDecodeContext;

static OPJ_SIZE_T kedge_opj_read_from_memory(void *buffer, OPJ_SIZE_T bytes, void *user_data) {
    KEdgeMemoryStream *stream = (KEdgeMemoryStream *)user_data;
    OPJ_SIZE_T remaining;
    if (stream == NULL || buffer == NULL) {
        return (OPJ_SIZE_T)-1;
    }
    if (stream->offset >= stream->size) {
        return (OPJ_SIZE_T)-1;
    }
    remaining = stream->size - stream->offset;
    if (bytes > remaining) {
        bytes = remaining;
    }
    memcpy(buffer, stream->data + stream->offset, (size_t)bytes);
    stream->offset += bytes;
    return bytes;
}

static OPJ_OFF_T kedge_opj_skip_in_memory(OPJ_OFF_T bytes, void *user_data) {
    KEdgeMemoryStream *stream = (KEdgeMemoryStream *)user_data;
    OPJ_OFF_T remaining;
    if (stream == NULL) {
        return (OPJ_OFF_T)-1;
    }
    remaining = (OPJ_OFF_T)(stream->size - stream->offset);
    if (bytes > remaining) {
        bytes = remaining;
    }
    if (bytes < 0) {
        if (-bytes > (OPJ_OFF_T)stream->offset) {
            bytes = -(OPJ_OFF_T)stream->offset;
        }
    }
    stream->offset = (OPJ_SIZE_T)((OPJ_OFF_T)stream->offset + bytes);
    return bytes;
}

static OPJ_BOOL kedge_opj_seek_in_memory(OPJ_OFF_T offset, void *user_data) {
    KEdgeMemoryStream *stream = (KEdgeMemoryStream *)user_data;
    if (stream == NULL || offset < 0 || (OPJ_SIZE_T)offset > stream->size) {
        return OPJ_FALSE;
    }
    stream->offset = (OPJ_SIZE_T)offset;
    return OPJ_TRUE;
}

static int kedge_init_decode_context(KEdgeDecodeContext *ctx, KEdgeStatus *status) {
    memset(ctx, 0, sizeof(*ctx));
    opj_set_default_decoder_parameters(&ctx->parameters);
    ctx->decode_threads = kedge_openjpeg_thread_count();
    (void)status;
    return 1;
}

static void kedge_destroy_decode_context(KEdgeDecodeContext *ctx) {
    (void)ctx;
}

static int decode_j2k_memory_with_context(
    KEdgeDecodeContext *ctx,
    const unsigned char *data,
    size_t data_size,
    KEdgeImageF64 *image,
    KEdgeStatus *status
) {
    opj_codec_t *codec = NULL;
    opj_stream_t *stream = NULL;
    opj_image_t *decoded = NULL;
    KEdgeMemoryStream mem_stream;
    size_t pixel_count;
    size_t i;

    memset(image, 0, sizeof(*image));
    if (ctx == NULL) {
        kedge_status_set(status, "OpenJPEG decode context is not initialized.");
        return 0;
    }
    if (data == NULL || data_size == 0) {
        kedge_status_set(status, "Empty in-memory J2K codestream.");
        return 0;
    }
    kedge_status_reset(status);
    {
        OPJ_CODEC_FORMAT codec_format = OPJ_CODEC_J2K;
        if (data_size >= 12 &&
            data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x00 && data[3] == 0x0c &&
            data[4] == 0x6a && data[5] == 0x50 && data[6] == 0x20 && data[7] == 0x20 &&
            data[8] == 0x0d && data[9] == 0x0a && data[10] == 0x87 && data[11] == 0x0a) {
            codec_format = OPJ_CODEC_JP2;
        }
        codec = opj_create_decompress(codec_format);
    }
    if (codec == NULL) {
        kedge_status_set(status, "Failed to create OpenJPEG decoder.");
        return 0;
    }
    opj_set_error_handler(codec, opj_error_callback, status);
    opj_set_warning_handler(codec, opj_error_callback, status);
    opj_set_info_handler(codec, NULL, NULL);
    if (!opj_setup_decoder(codec, &ctx->parameters)) {
        kedge_status_set(status, "Failed to initialize OpenJPEG decoder.");
        opj_destroy_codec(codec);
        return 0;
    }
    if (ctx->decode_threads > 1) {
        opj_codec_set_threads(codec, ctx->decode_threads);
    }
    mem_stream.data = data;
    mem_stream.size = (OPJ_SIZE_T)data_size;
    mem_stream.offset = 0;
    stream = opj_stream_create(65536, OPJ_TRUE);
    if (stream == NULL) {
        opj_destroy_codec(codec);
        kedge_status_set(status, "Failed to create OpenJPEG memory stream.");
        return 0;
    }
    opj_stream_set_user_data(stream, &mem_stream, NULL);
    opj_stream_set_user_data_length(stream, mem_stream.size);
    opj_stream_set_read_function(stream, kedge_opj_read_from_memory);
    opj_stream_set_skip_function(stream, kedge_opj_skip_in_memory);
    opj_stream_set_seek_function(stream, kedge_opj_seek_in_memory);
    if (!opj_read_header(stream, codec, &decoded) || decoded == NULL) {
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        if (status->error[0] == '\0') {
            kedge_status_set(status, "Failed to read OpenJPEG header from memory stream.");
        }
        return 0;
    }
    if (!opj_decode(codec, stream, decoded) || !opj_end_decompress(codec, stream)) {
        opj_image_destroy(decoded);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        if (status->error[0] == '\0') {
            kedge_status_set(status, "OpenJPEG decode failed from memory stream.");
        }
        return 0;
    }
    if (decoded->numcomps < 1 || decoded->comps[0].data == NULL) {
        opj_image_destroy(decoded);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        kedge_status_set(status, "OpenJPEG returned an empty image from memory stream.");
        return 0;
    }

    image->width = decoded->comps[0].w;
    image->height = decoded->comps[0].h;
    pixel_count = image->width * image->height;
    image->data = (double *)malloc(pixel_count * sizeof(double));
    if (image->data == NULL) {
        opj_image_destroy(decoded);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        kedge_status_set(status, "Out of memory while storing OpenJPEG decode result.");
        return 0;
    }
    for (i = 0; i < pixel_count; ++i) {
        image->data[i] = (double)decoded->comps[0].data[i];
    }

    opj_image_destroy(decoded);
    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    return 1;
}

int kedge_decode_openjpeg_image_from_memory(
    const unsigned char *data,
    size_t data_size,
    KEdgeImageF64 *image,
    KEdgeStatus *status
) {
    KEdgeDecodeContext ctx;
    int ok;
    if (!kedge_init_decode_context(&ctx, status)) {
        return 0;
    }
    ok = decode_j2k_memory_with_context(&ctx, data, data_size, image, status);
    kedge_destroy_decode_context(&ctx);
    return ok;
}

int kedge_find_jp2_markers(const unsigned char *blob, size_t blob_size, size_t *markers, size_t max_markers) {
    size_t pos;
    size_t found = 0;
    if (blob == NULL || markers == NULL || blob_size < 4) {
        return 0;
    }
    for (pos = 0; pos + 4 <= blob_size; ++pos) {
        if (blob[pos] == 'j' && blob[pos + 1] == 'p' && blob[pos + 2] == '2' && blob[pos + 3] == 'c') {
            if (found < max_markers) {
                markers[found] = pos;
            }
            found++;
        }
    }
    return (int)found;
}

int kedge_decode_group_tiles(
    const unsigned char *blob,
    size_t blob_size,
    const size_t *markers,
    size_t marker_count,
    int group_id,
    const char *temp_dir,
    KEdgeImageF64 *tiles,
    KEdgeStatus *status
) {
    int start_slot = group_id * 64;
    int slot;
    int out_idx = 0;
    KEdgeDecodeContext ctx;
    if (blob == NULL || markers == NULL || tiles == NULL || temp_dir == NULL) {
        kedge_status_set(status, "Invalid arguments for kedge_decode_group_tiles.");
        return 0;
    }
    (void)temp_dir;
    if (!kedge_init_decode_context(&ctx, status)) {
        return 0;
    }
    for (slot = start_slot; slot < start_slot + 64; ++slot) {
        size_t start;
        size_t end;
        if (is_empty_tile_position(slot - start_slot)) {
            continue;
        }
        if ((size_t)slot >= marker_count) {
            kedge_status_set(status, "Insufficient jp2c markers to decode group %d.", group_id);
            return 0;
        }
        start = markers[slot] + 4;
        end = ((size_t)(slot + 1) < marker_count) ? markers[slot + 1] : blob_size;
        if (!decode_j2k_memory_with_context(&ctx, blob + start, end - start, &tiles[out_idx], status)) {
            kedge_destroy_decode_context(&ctx);
            return 0;
        }
        out_idx++;
    }
    kedge_destroy_decode_context(&ctx);
    return 1;
}

typedef struct {
    const unsigned char *blob;
    size_t blob_size;
    const size_t *markers;
    size_t marker_count;
    int slot;
    KEdgeImageF64 *tile_out;
} KEdgeDecodeTileTask;

typedef struct {
    const KEdgeDecodeTileTask *tasks;
    LONG task_count;
    volatile LONG next_task_index;
    volatile LONG failure_index;
    int enable_cuda;
} KEdgeDecodeTilePool;

typedef struct {
    KEdgeDecodeTilePool *pool;
    KEdgeStatus status;
    int ok;
} KEdgeDecodeTileWorker;

static unsigned __stdcall kedge_decode_tile_worker(void *arg) {
    KEdgeDecodeTileWorker *worker = (KEdgeDecodeTileWorker *)arg;
    KEdgeDecodeContext ctx;
    if (worker == NULL || worker->pool == NULL) {
        return 1U;
    }
    worker->ok = 1;
    kedge_status_reset(&worker->status);
    
    if (!kedge_init_decode_context(&ctx, &worker->status)) {
        worker->ok = 0;
        InterlockedCompareExchange(&worker->pool->failure_index, 0, -1);
        return 1U;
    }
    
    for (;;) {
        LONG task_index = InterlockedIncrement(&worker->pool->next_task_index) - 1;
        const KEdgeDecodeTileTask *task;
        size_t start;
        size_t end;
        if (task_index >= worker->pool->task_count) {
            break;
        }
        if (worker->pool->failure_index >= 0) {
            break;
        }
        task = &worker->pool->tasks[task_index];
        if ((size_t)task->slot >= task->marker_count) {
            kedge_status_set(&worker->status, "Insufficient jp2c markers to decode slot %d.", task->slot);
            worker->ok = 0;
            InterlockedCompareExchange(&worker->pool->failure_index, task_index, -1);
            break;
        }
        start = task->markers[task->slot] + 4;
        end = ((size_t)(task->slot + 1) < task->marker_count) ? task->markers[task->slot + 1] : task->blob_size;
        
        int decode_ok = decode_j2k_memory_with_context(&ctx, task->blob + start, end - start, task->tile_out, &worker->status);
        
        if (!decode_ok) {
            worker->ok = 0;
            InterlockedCompareExchange(&worker->pool->failure_index, task_index, -1);
            break;
        }
    }
    
    kedge_destroy_decode_context(&ctx);
    return worker->ok ? 0U : 1U;
}

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
) {
    KEdgeImageF64 *outputs[4];
    KEdgeDecodeTileTask tasks[240];
    KEdgeDecodeTilePool pool;
    KEdgeDecodeTileWorker workers[32];
    HANDLE handles[32];
    unsigned thread_ids[32];
    int started[32];
    int i;
    int worker_count;
    int task_count;
    if (
        blob == NULL || markers == NULL || temp_dir == NULL ||
        g0_tiles == NULL || g1_tiles == NULL || g2_tiles == NULL || g3_tiles == NULL || status == NULL
    ) {
        kedge_status_set(status, "Invalid arguments for kedge_decode_all_groups_parallel.");
        return 0;
    }
    outputs[0] = g0_tiles;
    outputs[1] = g1_tiles;
    outputs[2] = g2_tiles;
    outputs[3] = g3_tiles;
    memset(handles, 0, sizeof(handles));
    memset(started, 0, sizeof(started));
    memset(workers, 0, sizeof(workers));

    task_count = 0;
    for (i = 0; i < 4; ++i) {
        int slot;
        int out_idx = 0;
        for (slot = i * 64; slot < i * 64 + 64; ++slot) {
            if (is_empty_tile_position(slot - i * 64)) {
                continue;
            }
            tasks[task_count].blob = blob;
            tasks[task_count].blob_size = blob_size;
            tasks[task_count].markers = markers;
            tasks[task_count].marker_count = marker_count;
            tasks[task_count].slot = slot;
            tasks[task_count].tile_out = &outputs[i][out_idx];
            task_count++;
            out_idx++;
        }
    }

    worker_count = kedge_marker_tile_worker_count();
#ifdef KEDGE_USE_CUDA
    if (enable_cuda == 2 || (enable_cuda && getenv("KEDGE_FORCE_NVJPEG2K") != NULL)) {
        const unsigned char *data_ptrs[240];
        size_t data_sizes[240];
        KEdgeImageF64 *out_images[240];
        
        for (i = 0; i < task_count; ++i) {
            size_t start = tasks[i].markers[tasks[i].slot] + 4;
            size_t end = ((size_t)(tasks[i].slot + 1) < tasks[i].marker_count) ? tasks[i].markers[tasks[i].slot + 1] : tasks[i].blob_size;
            data_ptrs[i] = tasks[i].blob + start;
            data_sizes[i] = end - start;
            out_images[i] = tasks[i].tile_out;
        }
        
        if (!kedge_decode_nvjpeg2k_batched(task_count, data_ptrs, data_sizes, out_images, status)) {
            return 0;
        }
        return 1;
    }
#endif
    if (worker_count > task_count) {
        worker_count = task_count;
    }
    if (worker_count > (int)(sizeof(workers) / sizeof(workers[0]))) {
        worker_count = (int)(sizeof(workers) / sizeof(workers[0]));
    }
    
    if (worker_count <= 1) {
        KEdgeDecodeContext ctx;
        if (!kedge_init_decode_context(&ctx, status)) {
            return 0;
        }
        for (i = 0; i < task_count; ++i) {
            size_t start;
            size_t end;
            if ((size_t)tasks[i].slot >= marker_count) {
                kedge_destroy_decode_context(&ctx);
                kedge_status_set(status, "Insufficient jp2c markers to decode slot %d.", tasks[i].slot);
                return 0;
            }
            start = markers[tasks[i].slot] + 4;
            end = ((size_t)(tasks[i].slot + 1) < marker_count) ? markers[tasks[i].slot + 1] : blob_size;
            
            int decode_ok = decode_j2k_memory_with_context(&ctx, blob + start, end - start, tasks[i].tile_out, status);
            if (!decode_ok) {
                kedge_destroy_decode_context(&ctx);
                return 0;
            }
        }
        kedge_destroy_decode_context(&ctx);
        return 1;
    }
    pool.tasks = tasks;
    pool.task_count = task_count;
    pool.next_task_index = 0;
    pool.failure_index = -1;
    pool.enable_cuda = enable_cuda;
    (void)temp_dir;
    for (i = 0; i < worker_count; ++i) {
        workers[i].pool = &pool;
        handles[i] = (HANDLE)_beginthreadex(NULL, 0, kedge_decode_tile_worker, &workers[i], 0, &thread_ids[i]);
        if (handles[i] != NULL) {
            started[i] = 1;
        } else {
            workers[i].ok = 0;
            kedge_status_set(&workers[i].status, "Failed to start marker tile worker %d.", i);
            InterlockedCompareExchange(&pool.failure_index, i, -1);
        }
    }
    for (i = 0; i < worker_count; ++i) {
        if (started[i]) {
            WaitForSingleObject(handles[i], INFINITE);
            CloseHandle(handles[i]);
            handles[i] = NULL;
        }
    }
    for (i = 0; i < worker_count; ++i) {
        if (!workers[i].ok) {
            if (workers[i].status.error[0] != '\0') {
                kedge_status_set(status, "%s", workers[i].status.error);
            } else {
                kedge_status_set(status, "Marker tile worker %d failed.", i);
            }
            return 0;
        }
    }
    return 1;
}
