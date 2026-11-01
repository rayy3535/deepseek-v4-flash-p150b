// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// data movement in for one compressor history's pooling, this core's output column
// tile c (pool_cols_compute.cpp). The raw history rows [R, Wtot] (bf16 TILE, row r is position pos - (R - 1) + r)
// are read straight into the score / kv tiles of column c:
//   ratio 4 (R = 8): rows 0..3 from the previous group's half (column tile c), rows 4..7 from the current group's
//                    half (column tile Wt + c); padding rows 8..31: scores -inf, kv 0
//   otherwise:       row tile rt, column tile c, unchanged
// and rows before the sequence start (pos - (R - 1) + r < 0) get -inf scores. Also the constant tiles.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t TILE = 1024;
constexpr uint32_t ONE_F32 = 0x3F800000;
constexpr uint32_t ONE_BF16_PAIR = 0x3F803F80;
constexpr uint32_t NEG_INF_BF16_PAIR = 0xFF80FF80;

FORCE_INLINE uint32_t at(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

// bf16 tile row r as 16 words (two 8-word face-row halves)
FORCE_INLINE void fill_row(volatile tt_l1_ptr uint32_t* tile, uint32_t r, uint32_t value) {
    const uint32_t a = at(r, 0) / 2, b = at(r, 16) / 2;
    for (uint32_t w = 0; w < 8; ++w) {
        tile[a + w] = value;
        tile[b + w] = value;
    }
}

FORCE_INLINE void copy_row(volatile tt_l1_ptr uint32_t* dst, volatile tt_l1_ptr uint32_t* src, uint32_t r) {
    const uint32_t a = at(r, 0) / 2, b = at(r, 16) / 2;
    for (uint32_t w = 0; w < 8; ++w) {
        dst[a + w] = src[a + w];
        dst[b + w] = src[b + w];
    }
}
}  // namespace

void kernel_main() {
    const uint32_t raw_addr = get_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(2);

    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t RT = (R + 31) / 32;
    constexpr uint32_t raw_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t kv_off = get_compile_time_arg_val(2);
    constexpr uint32_t score_off = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t ratio = get_compile_time_arg_val(5);
    constexpr uint32_t overlap = get_compile_time_arg_val(6);
    constexpr auto raw_args = TensorAccessorArgs<7>();
    constexpr auto pos_args = TensorAccessorArgs<raw_args.next_compile_time_args_offset()>();
    const auto raw = TensorAccessor(raw_args, raw_addr);
    const auto pos_t = TensorAccessor(pos_args, pos_addr);

    Noc noc;
    CircularBuffer s_cb(0), k_cb(1), scaler_cb(2), jcol_cb(3), jrow_cb(4), ones_cb(5), stage_cb(9), pos_cb(10);
    constexpr uint32_t page = 2048;

    pos_cb.reserve_back(1);
    noc.async_read(pos_t, CoreLocalMem<uint32_t>(pos_cb.get_write_ptr()), 4, {.page_id = 0}, {});
    s_cb.reserve_back(RT);
    k_cb.reserve_back(RT);
    stage_cb.reserve_back(1);
    for (uint32_t rt = 0; rt < RT; ++rt) {
        noc.async_read(
            raw,
            CoreLocalMem<uint32_t>(s_cb.get_write_ptr() + rt * page),
            page,
            {.page_id = rt * raw_width_tiles + score_off + c},
            {});
        noc.async_read(
            raw,
            CoreLocalMem<uint32_t>(k_cb.get_write_ptr() + rt * page),
            page,
            {.page_id = rt * raw_width_tiles + kv_off + c},
            {});
    }

    // constants (while the reads are in flight)
    scaler_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* scaler = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler_cb.get_write_ptr());
    for (uint32_t w = 0; w < TILE / 2; ++w) {
        scaler[w] = ONE_BF16_PAIR;
    }
    scaler_cb.push_back(1);
    jcol_cb.reserve_back(1);
    jrow_cb.reserve_back(1);
    ones_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* jcol = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(jcol_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* jrow = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(jrow_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_cb.get_write_ptr());
    for (uint32_t w = 0; w < TILE; ++w) {
        jcol[w] = 0;
        jrow[w] = 0;
        ones[w] = ONE_F32;
    }
    for (uint32_t i = 0; i < 32; ++i) {
        jcol[at(i, 0)] = ONE_F32;
        jrow[at(0, i)] = ONE_F32;
    }
    jcol_cb.push_back(1);
    jrow_cb.push_back(1);
    ones_cb.push_back(1);
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* k = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(k_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_cb.get_write_ptr());
    if constexpr (overlap) {
        // rows ratio .. 2 ratio - 1 from the current group's half
        noc.async_read(
            raw, CoreLocalMem<uint32_t>(stage_cb.get_write_ptr()), page, {.page_id = score_off + Wt + c}, {});
        noc.async_read_barrier();
        for (uint32_t r = ratio; r < 2 * ratio; ++r) {
            copy_row(s, stage, r);
        }
        noc.async_read(raw, CoreLocalMem<uint32_t>(stage_cb.get_write_ptr()), page, {.page_id = kv_off + Wt + c}, {});
        noc.async_read_barrier();
        for (uint32_t r = ratio; r < 2 * ratio; ++r) {
            copy_row(k, stage, r);
        }
    }
    for (uint32_t r = R % 32 ? R % 32 : 32; r < 32 && RT == 1; ++r) {  // padding rows of a single row tile
        fill_row(s, r, NEG_INF_BF16_PAIR);
        fill_row(k, r, 0);
    }
    const int32_t pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_cb.get_write_ptr())[0];
    const int32_t first_valid = static_cast<int32_t>(R) - 1 - pos;
    for (int32_t r = 0; r < first_valid && r < static_cast<int32_t>(R); ++r) {
        fill_row(s + (r / 32) * (TILE / 2), r % 32, NEG_INF_BF16_PAIR);
    }
    s_cb.push_back(RT);
    k_cb.push_back(RT);
}
