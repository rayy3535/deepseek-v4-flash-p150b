// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// write this column tile's pooled row (page c) and its sum-of-squares tile (page Wt + c).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t c = get_arg_val<uint32_t>(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(out_args, out_addr);
    Noc noc;
    DataflowBuffer dfb(16);
    const uint32_t page = get_local_cb_interface(16).fifo_page_size;
    for (uint32_t i = 0; i < 2; ++i) {
        dfb.wait_front(1);
        noc.async_write(dfb, out, page, {}, {.page_id = i * Wt + c});
        noc.async_writes_flushed();
        dfb.pop_front(1);
    }
    noc.async_write_barrier();
}
