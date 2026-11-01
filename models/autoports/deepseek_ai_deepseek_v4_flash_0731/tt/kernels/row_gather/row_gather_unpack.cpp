// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// unpack each gathered BFP8_b tile (c_0) and pack it in the output format (c_16, BF16).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);
        copy_tile(cb_in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
