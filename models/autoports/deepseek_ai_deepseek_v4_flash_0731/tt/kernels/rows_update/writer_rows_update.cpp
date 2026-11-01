// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// for each of the two passes (the tile rows of positions start and start + M - 1), copy every block row that
// falls in that tile row from the untilized block into the untilized cache rows, hand them back for tilizing and
// write the re-tilized cache tiles. When both positions share a tile row the two passes write the same bytes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "rows_map.hpp"

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2);
    const uint32_t c0 = get_arg_val<uint32_t>(3);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Wc = get_compile_time_arg_val(1);
    constexpr uint32_t M = get_compile_time_arg_val(2);
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_heads = get_compile_time_arg_val(4);
    constexpr uint32_t modulo = get_compile_time_arg_val(5);
    constexpr uint32_t page_table_bytes = get_compile_time_arg_val(6);
    constexpr auto cache_args = TensorAccessorArgs<7>();
    constexpr auto pt_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto pt = TensorAccessor(pt_args, page_table_addr);
    const auto pos_t = TensorAccessor(pos_args, pos_addr);

    Noc noc;
    CircularBuffer untilized_cache_cb(2), untilized_cache2_cb(3), untilized_in_cb(4), out_cb(16), scratch_cb(8);
    constexpr uint32_t row_bytes = Wc * 32 * 2;  // one bf16 row of this chunk, untilized
    const uint32_t out_page = get_local_cb_interface(16).fifo_page_size;

    scratch_cb.reserve_back(1);
    const uint32_t scratch = scratch_cb.get_write_ptr();
    noc.async_read(pos_t, CoreLocalMem<uint32_t>(scratch), 4, {.page_id = 0}, {});
    noc.async_read(pt, CoreLocalMem<uint32_t>(scratch + 64), page_table_bytes, {.page_id = 0}, {});
    noc.async_read_barrier();
    const uint32_t start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[0];
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 64);

    const uint8_t noc_id = noc.get_noc_id();
    UnicastEndpoint local;
    untilized_in_cb.wait_front(Wc);
    const uint32_t in_rows = untilized_in_cb.get_read_ptr();
    for (uint32_t pass = 0; pass < 2; ++pass) {
        const RowSlot slot = row_slot<block_size, num_heads, Wt, modulo>(start + (pass ? M - 1 : 0), table);
        untilized_cache_cb.wait_front(Wc);
        untilized_cache2_cb.reserve_back(Wc);
        const uint32_t rows = untilized_cache_cb.get_read_ptr();
        for (uint32_t r = 0; r < M; ++r) {
            const RowSlot s = row_slot<block_size, num_heads, Wt, modulo>(start + r, table);
            if (s.tile_row != slot.tile_row) {
                continue;
            }
            noc.async_read(
                local,
                CoreLocalMem<uint32_t>(rows + s.row * row_bytes),
                row_bytes,
                {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = in_rows + r * row_bytes},
                {});
        }
        noc.async_read_barrier();
        untilized_cache2_cb.push_back(Wc);
        untilized_cache_cb.pop_front(Wc);

        out_cb.wait_front(Wc);
        uint32_t src = out_cb.get_read_ptr();
        for (uint32_t t = 0; t < Wc; ++t) {
            noc.async_write(CoreLocalMem<uint32_t>(src), cache, out_page, {}, {.page_id = slot.tile_row + c0 + t});
            src += out_page;
        }
        noc.async_write_barrier();
        out_cb.pop_front(Wc);
    }
    untilized_in_cb.pop_front(Wc);
}
