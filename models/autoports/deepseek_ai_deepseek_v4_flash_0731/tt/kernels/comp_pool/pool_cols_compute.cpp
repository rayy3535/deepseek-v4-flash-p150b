// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// one compressor history's pooling for this core's output column tile
// (OptimizedDecoder._comp_pool_fused; stage 2 is pool_finish_compute.cpp). With the reader's score / kv tiles:
//   m      = column max of the scores                         (reduce MAX over the rows)
//   e      = exp(s - m)                                       (m broadcast to the rows by a ones-column matmul)
//   pooled = (1 @ (kv * e)) / (1 @ e)                         (column sums by ones-row matmuls; row 0)
//   sq     = (pooled * pooled) @ 1                            (row 0: this column tile's sum of squares)
// Intermediates fp32; out: pooled, then sq.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_s = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_jcol = tt::CBIndex::c_3;
constexpr uint32_t cb_jrow = tt::CBIndex::c_4;
constexpr uint32_t cb_ones = tt::CBIndex::c_5;
constexpr uint32_t cb_norm = tt::CBIndex::c_6;
constexpr uint32_t cb_cos = tt::CBIndex::c_7;
constexpr uint32_t cb_sin = tt::CBIndex::c_8;
constexpr uint32_t cb_trans = tt::CBIndex::c_11;
constexpr uint32_t cb_had = tt::CBIndex::c_12;
constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_max = tt::CBIndex::c_17;
constexpr uint32_t cb_e = tt::CBIndex::c_18;
constexpr uint32_t cb_ke = tt::CBIndex::c_19;
constexpr uint32_t cb_pooled = tt::CBIndex::c_20;
constexpr uint32_t cb_sq = tt::CBIndex::c_21;
constexpr uint32_t cb_rms = tt::CBIndex::c_22;
constexpr uint32_t cb_v = tt::CBIndex::c_23;
constexpr uint32_t cb_normed = tt::CBIndex::c_24;

constexpr uint32_t TINY_F32 = 0x0DA24260;  // 1e-30: rows other than 0 stay 0 / tiny = 0

FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

// in0 @ in1 (+=) into dst; srcA takes in1, srcB in0
FORCE_INLINE void mm_init(uint32_t in0, uint32_t in1) {
    reconfig_data_format(in1, in0);
    matmul_init(in0, in1);
}

FORCE_INLINE void pack_range(uint32_t cb, uint32_t first, uint32_t count) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, count);
    pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < count; ++i) {
        pack_tile(first + i, cb);
    }
    cb_push_back(cb, count);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t RT = get_compile_time_arg_val(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_jcol, cb_max, cb_e);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_jcol, 1);
    cb_wait_front(cb_jrow, 1);
    cb_wait_front(cb_ones, 1);
    cb_wait_front(cb_s, RT);
    cb_wait_front(cb_k, RT);

    // column max of the scores
    reconfig_data_format(cb_s, cb_scaler);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_s, cb_scaler, cb_max);
    tile_regs_acquire();
    for (uint32_t rt = 0; rt < RT; ++rt) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_s, cb_scaler, rt, 0, 0);
    }
    pack_range(cb_max, 0, 1);
    reduce_uninit(cb_s);

    // e = exp(s - max), kv * e
    cb_wait_front(cb_max, 1);
    tile_regs_acquire();
    mm_init(cb_jcol, cb_max);
    matmul_tiles(cb_jcol, cb_max, 0, 0, 0);
    for (uint32_t rt = 0; rt < RT; ++rt) {
        load(cb_s, rt, 1 + rt);
        sub_binary_tile_init();
        sub_binary_tile(1 + rt, 0, 1 + rt);
        exp_tile_init();
        exp_tile(1 + rt);
    }
    pack_range(cb_e, 1, RT);
    cb_wait_front(cb_e, RT);
    tile_regs_acquire();
    for (uint32_t rt = 0; rt < RT; ++rt) {
        load(cb_k, rt, rt);
        load(cb_e, rt, RT + rt);
        mul_binary_tile_init();
        mul_binary_tile(rt, RT + rt, rt);
    }
    pack_range(cb_ke, 0, RT);

    // pooled = colsum(kv * e) / colsum(e)
    cb_wait_front(cb_ke, RT);
    tile_regs_acquire();
    mm_init(cb_jrow, cb_e);
    for (uint32_t rt = 0; rt < RT; ++rt) {
        matmul_tiles(cb_jrow, cb_e, 0, rt, 0);
    }
    mm_init(cb_jrow, cb_ke);
    for (uint32_t rt = 0; rt < RT; ++rt) {
        matmul_tiles(cb_jrow, cb_ke, 0, rt, 1);
    }
    binop_with_scalar_tile_init();
    add_unary_tile(0, TINY_F32);
    div_binary_tile_init();
    div_binary_tile(1, 0, 1);
    pack_range(cb_pooled, 1, 1);

    // sq = (pooled * pooled) @ ones -> row 0 holds this tile's sum of squares in every column
    cb_wait_front(cb_pooled, 1);
    tile_regs_acquire();
    load(cb_pooled, 0, 0);
    load(cb_pooled, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    pack_range(cb_sq, 0, 1);
    cb_wait_front(cb_sq, 1);
    tile_regs_acquire();
    mm_init(cb_sq, cb_ones);
    matmul_tiles(cb_sq, cb_ones, 0, 0, 0);
    load(cb_pooled, 0, 1);
    pack_range(cb_out, 1, 1);  // pooled first, then sq
    tile_regs_acquire();
    mm_init(cb_sq, cb_ones);
    matmul_tiles(cb_sq, cb_ones, 0, 0, 0);
    pack_range(cb_out, 0, 1);

    cb_pop_front(cb_sq, 1);
    cb_pop_front(cb_pooled, 1);
    cb_pop_front(cb_ke, RT);
    cb_pop_front(cb_e, RT);
    cb_pop_front(cb_max, 1);
    cb_pop_front(cb_s, RT);
    cb_pop_front(cb_k, RT);
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_jcol, 1);
    cb_pop_front(cb_jrow, 1);
    cb_pop_front(cb_ones, 1);
}
