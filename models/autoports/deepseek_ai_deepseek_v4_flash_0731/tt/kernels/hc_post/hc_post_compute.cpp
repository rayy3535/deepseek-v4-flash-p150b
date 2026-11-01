// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// the mHC output map comb^T @ residual + post * output as one matmul per residual tile,
// L @ R_t (reader_hc_post.cpp builds L and R_t), fp32 accumulation, one rounding to bf16 (the op-by-op version
// rounds the matmul to bf16 and then the addcmul, so the result is not bit-identical).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

namespace {
constexpr uint32_t cb_l = tt::CBIndex::c_0;
constexpr uint32_t cb_r = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_16;
}  // namespace

void kernel_main() {
    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(0);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_l, cb_r, cb_out);
    matmul_init(cb_l, cb_r);
    uint32_t current_row = 0xFFFFFFFF;
    for (uint32_t item = first; item < first + count; ++item) {
        const uint32_t m = item / hidden_tiles;
        if (m != current_row) {
            if (current_row != 0xFFFFFFFF) {
                cb_pop_front(cb_l, 1);
            }
            cb_wait_front(cb_l, 1);
            current_row = m;
        }
        cb_wait_front(cb_r, 1);
        tile_regs_acquire();
        matmul_tiles(cb_l, cb_r, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_r, 1);
    }
    if (current_row != 0xFFFFFFFF) {
        cb_pop_front(cb_l, 1);
    }
}
