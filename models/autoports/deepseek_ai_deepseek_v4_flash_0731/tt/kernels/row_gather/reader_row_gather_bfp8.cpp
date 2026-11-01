// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gather rows of a TILE BFP8_b [rows, W] cache straight from its tiles, without untilizing the table.
//
// Output tile t (row-major over [N / 32, W / 32]) holds rows index[32 * (t / Wt) + r], r = 0..31, of tile column
// t % Wt. A BFP8_b 32x32 tile is 64 exponent bytes (one per face row, faces 0..3, 16 rows each) followed by four
// 256-byte faces of 16 x 16 one-byte sign/mantissa values. Source row s of a tile lives in faces 2 * (s / 16) and
// 2 * (s / 16) + 1, face row s % 16: two exponent bytes and two 16-byte mantissa rows. They are read in 64-byte
// DRAM-aligned chunks into a staging buffer, then copied to face row r % 16 of output faces 2 * (r / 16) and +1.
// The compute kernel unpacks the assembled BFP8 tile and packs it as BF16, the same conversion the untilize inside
// ttnn.embedding applies, so the result matches the embedding of the untilized table.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t CHUNK = 64;        // DRAM read granularity / alignment
constexpr uint32_t EXP_BYTES = 64;    // exponent section of a BFP8 tile
constexpr uint32_t FACE_BYTES = 256;  // 16 x 16 mantissa bytes
constexpr uint32_t ROW_BYTES = 16;    // one face row of mantissas
constexpr uint32_t STAGE_ROW = 3 * CHUNK;
}  // namespace

void kernel_main() {
    const uint32_t table_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t index_tiled = get_compile_time_arg_val(1);  // 1: [1, N] uint32 TILE, 0: ROW_MAJOR
    constexpr auto table_args = TensorAccessorArgs<2>();
    constexpr auto index_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_tile = 0;
    constexpr uint32_t cb_index = 1;
    constexpr uint32_t cb_stage = 2;

    const auto table = TensorAccessor(table_args, table_addr);
    const auto index = TensorAccessor(index_args, index_addr);

    Noc noc;
    CircularBuffer out_cb(cb_tile);
    CircularBuffer index_cb(cb_index);
    CircularBuffer stage_cb(cb_stage);
    index_cb.reserve_back(1);
    stage_cb.reserve_back(1);
    const uint32_t index_l1 = index_cb.get_write_ptr();
    const uint32_t stage_l1 = stage_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_l1);
    volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_l1);

    uint32_t loaded_block = 0xFFFFFFFF;
    for (uint32_t tile = start_tile; tile < start_tile + num_tiles; ++tile) {
        const uint32_t block = tile / width_tiles;
        const uint32_t col = tile % width_tiles;
        if (block != loaded_block) {
            if constexpr (index_tiled) {
                // Entries 32 * block .. +31: row 0 of face 0 (entries 0-15) and of face 1 (16-31) of index tile
                // `block`.
                noc.async_read(
                    index, CoreLocalMem<uint32_t>(index_l1), CHUNK, {.page_id = block, .offset_bytes = 0}, {});
                noc.async_read(
                    index,
                    CoreLocalMem<uint32_t>(index_l1 + CHUNK),
                    CHUNK,
                    {.page_id = block, .offset_bytes = 1024},
                    {});
            } else {
                noc.async_read(
                    index,
                    CoreLocalMem<uint32_t>(index_l1),
                    2 * CHUNK,
                    {.page_id = 0, .offset_bytes = block * 2 * CHUNK},
                    {});
            }
            noc.async_read_barrier();
            loaded_block = block;
        }
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t row = rows[r];
            const uint32_t page = (row >> 5) * width_tiles + col;
            const uint32_t s = row & 31;
            const uint32_t mantissa = EXP_BYTES + (s >> 4) * 2 * FACE_BYTES + (s & 15) * ROW_BYTES;
            const uint32_t dst = stage_l1 + r * STAGE_ROW;
            noc.async_read(table, CoreLocalMem<uint32_t>(dst), CHUNK, {.page_id = page, .offset_bytes = 0}, {});
            noc.async_read(
                table,
                CoreLocalMem<uint32_t>(dst + CHUNK),
                CHUNK,
                {.page_id = page, .offset_bytes = mantissa & ~(CHUNK - 1)},
                {});
            noc.async_read(
                table,
                CoreLocalMem<uint32_t>(dst + 2 * CHUNK),
                CHUNK,
                {.page_id = page, .offset_bytes = (mantissa + FACE_BYTES) & ~(CHUNK - 1)},
                {});
        }
        out_cb.reserve_back(1);
        noc.async_read_barrier();

        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_cb.get_write_ptr());
        uint32_t exps[EXP_BYTES / 4] = {};
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t s = rows[r] & 31;
            const uint32_t src_face = (s >> 4) * 2;
            const uint32_t dst_face = (r >> 4) * 2;
            const uint32_t base = r * STAGE_ROW / 4;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t src_byte = (src_face + half) * 16 + (s & 15);
                const uint32_t exp = (stage[base + src_byte / 4] >> (8 * (src_byte % 4))) & 0xFF;
                const uint32_t dst_byte = (dst_face + half) * 16 + (r & 15);
                exps[dst_byte / 4] |= exp << (8 * (dst_byte % 4));
            }
            const uint32_t in_chunk = ((s & 15) * ROW_BYTES) % CHUNK / 4;  // same for both faces
            const uint32_t dst_row = (EXP_BYTES + dst_face * FACE_BYTES + (r & 15) * ROW_BYTES) / 4;
            for (uint32_t w = 0; w < ROW_BYTES / 4; ++w) {
                out[dst_row + w] = stage[base + CHUNK / 4 + in_chunk + w];
                out[dst_row + FACE_BYTES / 4 + w] = stage[base + 2 * CHUNK / 4 + in_chunk + w];
            }
        }
        for (uint32_t w = 0; w < EXP_BYTES / 4; ++w) {
            out[w] = exps[w];
        }
        out_cb.push_back(1);
    }
}
