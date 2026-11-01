// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// load the n x n ones block once, then this core's run of [n, n] mHC combination tiles (one per token) for the
// fused Sinkhorn.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t comb_addr = get_arg_val<uint32_t>(0);
    const uint32_t ones_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr auto comb_args = TensorAccessorArgs<0>();
    constexpr auto ones_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_comb = 0;
    constexpr uint32_t cb_ones = 1;

    const auto comb = TensorAccessor(comb_args, comb_addr);
    const auto ones = TensorAccessor(ones_args, ones_addr);
    const uint32_t comb_page = get_local_cb_interface(cb_comb).fifo_page_size;

    Noc noc;
    DataflowBuffer ones_dfb(cb_ones);
    DataflowBuffer comb_dfb(cb_comb);
    ones_dfb.reserve_back(1);
    noc.async_read(ones, ones_dfb, get_local_cb_interface(cb_ones).fifo_page_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    ones_dfb.push_back(1);
    for (uint32_t tile = start_tile; tile < start_tile + num_tiles; ++tile) {
        comb_dfb.reserve_back(1);
        noc.async_read(comb, comb_dfb, comb_page, {.page_id = tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        comb_dfb.push_back(1);
    }
}
