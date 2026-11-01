// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// data movement in for the fused mHC output map (hc_post_compute.cpp). For block row m:
//   L (bf16)  rows l < n: L[l, k] = comb[m][k, l] (k < n) and L[l, n] = post[m][l]
//   R_t       the residual tile t of row m (rows = lanes) with the sublayer output's row m of tile t copied into
//             row n
// so that L @ R_t = comb^T @ residual + post * output, the new residual tile. This core handles the work items
// first .. first + count - 1 (item = m * Ht + t).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t TILE = 1024;

FORCE_INLINE uint32_t at(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}
}  // namespace

void kernel_main() {
    const uint32_t residual_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t comb_addr = get_arg_val<uint32_t>(2);
    const uint32_t post_addr = get_arg_val<uint32_t>(3);
    const uint32_t first = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(1);
    constexpr auto residual_args = TensorAccessorArgs<2>();
    constexpr auto output_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();

    const auto residual = TensorAccessor(residual_args, residual_addr);
    const auto output = TensorAccessor(output_args, output_addr);
    const auto comb = TensorAccessor(comb_args, comb_addr);
    const auto post = TensorAccessor(post_args, post_addr);

    Noc noc;
    CircularBuffer l_cb(0), r_cb(1), stage_cb(2);
    constexpr uint32_t page = 2048;
    stage_cb.reserve_back(1);
    const uint32_t stage_addr = stage_cb.get_write_ptr();
    volatile tt_l1_ptr uint16_t* stage = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage_addr);
    {
        // both pages of the 2-page R buffer: rows past n stay zero for the whole program (L's columns > n are zero,
        // but 0 * NaN is NaN, so the residual's tile padding must not reach the matmul)
        volatile tt_l1_ptr uint32_t* r_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_cb.get_write_ptr());
        for (uint32_t w = 0; w < 2 * page / 4; ++w) {
            r_words[w] = 0;
        }
    }

    uint32_t current_row = 0xFFFFFFFF;
    for (uint32_t item = first; item < first + count; ++item) {
        const uint32_t m = item / hidden_tiles;
        const uint32_t t = item % hidden_tiles;
        if (m != current_row) {
            if (current_row != 0xFFFFFFFF) {
                l_cb.wait_front(1);  // the compute kernel pops the previous row's L when it moves on
            }
            l_cb.reserve_back(1);
            volatile tt_l1_ptr uint16_t* l = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l_cb.get_write_ptr());
            for (uint32_t i = 0; i < TILE; ++i) {
                l[i] = 0;
            }
            noc.async_read(comb, CoreLocalMem<uint32_t>(stage_cb.get_write_ptr()), page, {.page_id = m}, {});
            noc.async_read_barrier();
            for (uint32_t k = 0; k < n; ++k) {
                for (uint32_t j = 0; j < n; ++j) {
                    l[at(j, k)] = stage[at(k, j)];
                }
            }
            noc.async_read(post, CoreLocalMem<uint32_t>(stage_cb.get_write_ptr()), page, {.page_id = m}, {});
            noc.async_read_barrier();
            for (uint32_t j = 0; j < n; ++j) {
                l[at(j, n)] = stage[at(j, 0)];
            }
            l_cb.push_back(1);
            current_row = m;
        }
        // R_t: rows 0 .. n - 1 = the residual tile's lanes (the first n rows of faces 0 and 1), row n = the sublayer
        // output's row m, every other row zero. The two R pages were zeroed once at the start and only rows <= n are
        // ever written, so each item reads n rows of the residual tile and one output row instead of two whole tiles
        // and clearing 27 rows.
        r_cb.reserve_back(1);
        const uint32_t r_addr = r_cb.get_write_ptr();
        noc.async_read(residual, CoreLocalMem<uint32_t>(r_addr), n * 32, {.page_id = item, .offset_bytes = 0}, {});
        noc.async_read(
            residual, CoreLocalMem<uint32_t>(r_addr + 512), n * 32, {.page_id = item, .offset_bytes = 512}, {});
        // the sublayer output is [1, 1, M, H]: block row m is row m % 32 of tile (m / 32) * Ht + t; read the
        // 64-byte-aligned pair of face rows holding it (faces 0 / 1, or 2 / 3 for rows 16 .. 31)
        const uint32_t out_row = m % 32;
        const uint32_t pair = ((out_row >= 16) ? 1024 : 0) + ((out_row % 16) & ~1u) * 32;
        const uint32_t out_page = (m / 32) * hidden_tiles + t;
        noc.async_read(output, CoreLocalMem<uint32_t>(stage_addr), 64, {.page_id = out_page, .offset_bytes = pair}, {});
        noc.async_read(
            output, CoreLocalMem<uint32_t>(stage_addr + 64), 64, {.page_id = out_page, .offset_bytes = pair + 512}, {});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* r_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_addr);
        volatile tt_l1_ptr uint32_t* s_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_addr);
        const uint32_t odd = (out_row & 1) * 8;  // 16 bf16 = 8 words per face row
        for (uint32_t w = 0; w < 8; ++w) {
            r_words[n * 8 + w] = s_words[odd + w];             // face 0, row n
            r_words[128 + n * 8 + w] = s_words[16 + odd + w];  // face 1, row n
        }
        r_cb.push_back(1);
    }
}
