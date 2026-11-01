// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// untilize this core's chunk of the verify block once, then for each of the two passes untilize the cache
// tile row, wait for the writer to patch the block rows into it and tilize it back (same untilize / tilize helpers as
// paged_update_cache, so the cache bytes come out as its row-by-row updates would leave them).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t Wc = get_compile_time_arg_val(0);
    constexpr uint32_t cache_cb = 0, in_cb = 1, untilized_cache_cb = 2, untilized_cache2_cb = 3, untilized_in_cb = 4,
                       out_cb = 16;

    compute_kernel_hw_startup(in_cb, untilized_in_cb);
    compute_kernel_lib::untilize<
        Wc,
        in_cb,
        untilized_in_cb,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    for (uint32_t pass = 0; pass < 2; ++pass) {
        compute_kernel_lib::untilize<Wc, cache_cb, untilized_cache_cb>(1);
        compute_kernel_lib::tilize<Wc, untilized_cache2_cb, out_cb>(1);
    }
}
