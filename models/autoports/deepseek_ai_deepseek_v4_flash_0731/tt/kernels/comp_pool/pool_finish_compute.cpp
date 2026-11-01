// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// output column tile c of one pooled compressor history:
//   rms = sqrt(sum of the columns' sums of squares / W + eps)      (stage 1's sq tiles, row 0 broadcast)
//   y_t = pooled_t / rms * w_t, RoPE (y * cos + (y @ T) * sin) on the trailing rope_tiles columns
//   out = y_c, or sum_t y_t @ H[t, c] for the Hadamard (indexer) history; bf16.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_sq = tt::CBIndex::c_0;
constexpr uint32_t cb_pooled = tt::CBIndex::c_1;
constexpr uint32_t cb_norm = tt::CBIndex::c_2;
constexpr uint32_t cb_cos = tt::CBIndex::c_3;
constexpr uint32_t cb_sin = tt::CBIndex::c_4;
constexpr uint32_t cb_trans = tt::CBIndex::c_5;
constexpr uint32_t cb_had = tt::CBIndex::c_6;
constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_rms = tt::CBIndex::c_17;
constexpr uint32_t cb_v = tt::CBIndex::c_18;
constexpr uint32_t cb_y = tt::CBIndex::c_19;

FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

FORCE_INLINE void mm_init(uint32_t in0, uint32_t in1) {
    reconfig_data_format(in1, in0);
    matmul_init(in0, in1);
}

FORCE_INLINE void pack_one(uint32_t cb, uint32_t dst) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t rope_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t rotate = get_compile_time_arg_val(2);
    constexpr uint32_t inv_width = get_compile_time_arg_val(3);
    constexpr uint32_t eps = get_compile_time_arg_val(4);
    const uint32_t c_out = get_arg_val<uint32_t>(0);
    const uint32_t lo = rotate ? 0 : c_out, hi = rotate ? Wt : c_out + 1;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_v, cb_trans, cb_y);

    // rms
    cb_wait_front(cb_sq, Wt);
    tile_regs_acquire();
    load(cb_sq, 0, 0);
    for (uint32_t t = 1; t < Wt; ++t) {
        load(cb_sq, t, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_width);
    add_unary_tile(0, eps);
    sqrt_tile_init();
    sqrt_tile(0);
    pack_one(cb_rms, 0);
    cb_pop_front(cb_sq, Wt);

    // y_t for the columns this core needs
    cb_wait_front(cb_rms, 1);
    cb_wait_front(cb_pooled, hi - lo);
    cb_wait_front(cb_norm, hi - lo);
    cb_wait_front(cb_cos, hi - lo);
    cb_wait_front(cb_sin, hi - lo);
    cb_wait_front(cb_trans, 1);
    for (uint32_t t = lo; t < hi; ++t) {
        const uint32_t i = t - lo;
        tile_regs_acquire();
        load(cb_pooled, i, 0);
        load(cb_rms, 0, 1);
        div_binary_tile_init();
        div_binary_tile(0, 1, 0);
        load(cb_norm, i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        if (t + rope_tiles < Wt) {
            pack_one(cb_y, 0);
            continue;
        }
        pack_one(cb_v, 0);
        cb_wait_front(cb_v, 1);
        tile_regs_acquire();
        mm_init(cb_v, cb_trans);
        matmul_tiles(cb_v, cb_trans, 0, 0, 1);
        load(cb_v, 0, 0);
        load(cb_cos, i, 2);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);
        load(cb_sin, i, 2);
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        pack_one(cb_y, 0);
        cb_pop_front(cb_v, 1);
    }

    // out (bf16)
    cb_wait_front(cb_y, hi - lo);
    tile_regs_acquire();
    if constexpr (rotate) {
        cb_wait_front(cb_had, Wt);
        mm_init(cb_y, cb_had);
        for (uint32_t k = 0; k < Wt; ++k) {
            matmul_tiles(cb_y, cb_had, k, k, 0);
        }
        cb_pop_front(cb_had, Wt);
    } else {
        load(cb_y, 0, 0);
    }
    pack_one(cb_out, 0);
    cb_pop_front(cb_y, hi - lo);
    cb_pop_front(cb_pooled, hi - lo);
    cb_pop_front(cb_norm, hi - lo);
    cb_pop_front(cb_cos, hi - lo);
    cb_pop_front(cb_sin, hi - lo);
    cb_pop_front(cb_trans, 1);
    cb_pop_front(cb_rms, 1);
}
