// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// mHC A2: write the fused epilogue's per-row pre / post / comb tiles (tile m of each output = block row m).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t pre_addr = get_arg_val<uint32_t>(0);
    const uint32_t post_addr = get_arg_val<uint32_t>(1);
    const uint32_t comb_addr = get_arg_val<uint32_t>(2);
    const uint32_t first_row = get_arg_val<uint32_t>(3);
    const uint32_t end_row = first_row + get_arg_val<uint32_t>(4);

    constexpr auto pre_args = TensorAccessorArgs<0>();
    constexpr auto post_args = TensorAccessorArgs<pre_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_pre = 16, cb_post = 17, cb_comb = 18;

    const auto pre = TensorAccessor(pre_args, pre_addr);
    const auto post = TensorAccessor(post_args, post_addr);
    const auto comb = TensorAccessor(comb_args, comb_addr);

    Noc noc;
    DataflowBuffer pre_dfb(cb_pre), post_dfb(cb_post), comb_dfb(cb_comb);
    for (uint32_t m = first_row; m < end_row; ++m) {
        pre_dfb.wait_front(1);
        noc.async_write(pre_dfb, pre, get_local_cb_interface(cb_pre).fifo_page_size, {}, {.page_id = m});
        post_dfb.wait_front(1);
        noc.async_write(post_dfb, post, get_local_cb_interface(cb_post).fifo_page_size, {}, {.page_id = m});
        comb_dfb.wait_front(1);
        noc.async_write(comb_dfb, comb, get_local_cb_interface(cb_comb).fifo_page_size, {}, {.page_id = m});
        noc.async_writes_flushed();
        pre_dfb.pop_front(1);
        post_dfb.pop_front(1);
        comb_dfb.pop_front(1);
    }
    noc.async_write_barrier();
}
