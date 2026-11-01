// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// mHC A3: data movement in for the split mHC map (hc_map_compute.cpp). This core owns hidden tiles
// t0 .. t0 + nt - 1 of every lane: it loads the matching mHC map weight tiles once (lane l, hidden tile t is weight
// row tile l * Ht + t) and then streams each block row's residual tiles (block row m, hidden tile t is page m * Ht + t
// of the [1, M, n, H] residual, its rows the n lanes).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t streams_addr = get_arg_val<uint32_t>(0);
    const uint32_t fn_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t nt = get_arg_val<uint32_t>(3);
    const uint32_t num_rows = get_arg_val<uint32_t>(4);

    constexpr uint32_t lanes = get_compile_time_arg_val(0);
    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(1);
    constexpr auto streams_args = TensorAccessorArgs<2>();
    constexpr auto fn_args = TensorAccessorArgs<streams_args.next_compile_time_args_offset()>();

    const auto streams = TensorAccessor(streams_args, streams_addr);
    const auto fn = TensorAccessor(fn_args, fn_addr);

    Noc noc;
    CircularBuffer streams_cb(0), fn_cb(1);
    const uint32_t streams_page = get_local_cb_interface(0).fifo_page_size;
    const uint32_t fn_page = get_local_cb_interface(1).fifo_page_size;

    fn_cb.reserve_back(lanes * nt);
    for (uint32_t l = 0; l < lanes; ++l) {
        for (uint32_t i = 0; i < nt; ++i) {
            noc.async_read(
                fn,
                CoreLocalMem<uint32_t>(fn_cb.get_write_ptr() + (l * nt + i) * fn_page),
                fn_page,
                {.page_id = l * hidden_tiles + t0 + i},
                {});
        }
    }
    noc.async_read_barrier();
    fn_cb.push_back(lanes * nt);

    for (uint32_t m = 0; m < num_rows; ++m) {
        for (uint32_t i = 0; i < nt; ++i) {
            streams_cb.reserve_back(1);
            noc.async_read(
                streams,
                CoreLocalMem<uint32_t>(streams_cb.get_write_ptr()),
                streams_page,
                {.page_id = m * hidden_tiles + t0 + i},
                {});
            noc.async_read_barrier();
            streams_cb.push_back(1);
        }
    }
}
