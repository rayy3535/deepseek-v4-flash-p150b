// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// data movement in for writing the M rows of a verify block into a paged cache in one program
// (rows_update_compute.cpp, writer_rows_update.cpp): this core's column chunk [c0, c0 + Wc) of the block's tiles and,
// for the tile rows of positions start and start + M - 1 (one or two), the cache tiles of that chunk.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "rows_map.hpp"

void kernel_main() {
    const uint32_t value_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(2);
    const uint32_t pos_addr = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);  // cache width in tiles
    constexpr uint32_t Wc = get_compile_time_arg_val(1);  // this program's column chunk in tiles
    constexpr uint32_t M = get_compile_time_arg_val(2);
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_heads = get_compile_time_arg_val(4);
    constexpr uint32_t modulo = get_compile_time_arg_val(5);
    constexpr uint32_t page_table_bytes = get_compile_time_arg_val(6);
    constexpr auto value_args = TensorAccessorArgs<7>();
    constexpr auto cache_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    const auto value = TensorAccessor(value_args, value_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto pt = TensorAccessor(pt_args, page_table_addr);
    const auto pos_t = TensorAccessor(pos_args, pos_addr);

    Noc noc;
    CircularBuffer cache_cb(0), in_cb(1), scratch_cb(7);
    const uint32_t in_page = get_local_cb_interface(1).fifo_page_size;
    const uint32_t cache_page = get_local_cb_interface(0).fifo_page_size;

    scratch_cb.reserve_back(1);
    const uint32_t scratch = scratch_cb.get_write_ptr();
    noc.async_read(pos_t, CoreLocalMem<uint32_t>(scratch), 4, {.page_id = 0}, {});
    noc.async_read(pt, CoreLocalMem<uint32_t>(scratch + 64), page_table_bytes, {.page_id = 0}, {});
    in_cb.reserve_back(Wc);
    for (uint32_t t = 0; t < Wc; ++t) {
        noc.async_read(
            value, CoreLocalMem<uint32_t>(in_cb.get_write_ptr() + t * in_page), in_page, {.page_id = c0 + t}, {});
    }
    noc.async_read_barrier();
    in_cb.push_back(Wc);

    const uint32_t start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[0];
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 64);
    for (uint32_t pass = 0; pass < 2; ++pass) {
        const RowSlot slot = row_slot<block_size, num_heads, Wt, modulo>(start + (pass ? M - 1 : 0), table);
        cache_cb.reserve_back(Wc);
        for (uint32_t t = 0; t < Wc; ++t) {
            noc.async_read(
                cache,
                CoreLocalMem<uint32_t>(cache_cb.get_write_ptr() + t * cache_page),
                cache_page,
                {.page_id = slot.tile_row + c0 + t},
                {});
        }
        noc.async_read_barrier();
        cache_cb.push_back(Wc);
    }
}
