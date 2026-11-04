// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Expert paging, device side of the large-page load: an expert's packed tiles arrive (DMA from pinned host memory)
// in a staging buffer whose pages hold TILES_PER_PAGE consecutive tiles; this kernel moves every tile into its
// slot of the tile-interleaved expert pool (one page per tile). Core k handles items [first, first + count) of
// the flattened (expert, tile) list; expert e of the batch goes from staging slot e to pool slot params[1 + e].
// The batch size and slots come from a small device params buffer, so every runtime argument is constant and
// the host reuses one program description per pool.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t staging_addr = get_arg_val<uint32_t>(0);
    const uint32_t pool_addr = get_arg_val<uint32_t>(1);
    const uint32_t params_addr = get_arg_val<uint32_t>(2);
    const uint32_t core_index = get_arg_val<uint32_t>(3);
    const uint32_t num_cores = get_arg_val<uint32_t>(4);

    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_expert = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_expert = tiles_per_expert / tiles_per_page;
    constexpr uint32_t depth = get_compile_time_arg_val(3);  // tiles in flight
    constexpr uint32_t params_bytes = get_compile_time_arg_val(4);
    constexpr auto staging_args = TensorAccessorArgs<5>();
    constexpr auto pool_args = TensorAccessorArgs<staging_args.next_compile_time_args_offset()>();
    constexpr auto params_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();
    const auto staging = TensorAccessor(staging_args, staging_addr);
    const auto pool = TensorAccessor(pool_args, pool_addr);
    const auto params_acc = TensorAccessor(params_args, params_addr);

    // params (uint32): [number of experts in this batch, pool slot of batch expert 0, 1, ...] written by the host
    const uint32_t params_l1 = get_write_ptr(1);
    noc_async_read(params_acc.get_noc_addr(0), params_l1, params_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* params = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(params_l1);
    const uint32_t items = params[0] * tiles_per_expert;
    const uint32_t per_core = (items + num_cores - 1) / num_cores;
    uint32_t item = core_index * per_core;
    const uint32_t end = item + per_core < items ? item + per_core : items;

    const uint32_t l1 = get_write_ptr(0);
    while (item < end) {
        const uint32_t batch = (end - item) < depth ? (end - item) : depth;
        for (uint32_t i = 0; i < batch; ++i) {
            const uint32_t it = item + i;
            const uint32_t e = it / tiles_per_expert, t = it % tiles_per_expert;
            const uint32_t page = e * pages_per_expert + t / tiles_per_page;
            noc_async_read(
                staging.get_noc_addr(page, (t % tiles_per_page) * tile_bytes), l1 + i * tile_bytes, tile_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < batch; ++i) {
            const uint32_t it = item + i;
            const uint32_t e = it / tiles_per_expert, t = it % tiles_per_expert;
            noc_async_write(l1 + i * tile_bytes, pool.get_noc_addr(params[1 + e] * tiles_per_expert + t), tile_bytes);
        }
        noc_async_write_barrier();
        item += batch;
    }
}
