// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// the MoE router epilogue after the gate matmul, in one program (OptimizedDecoder._moe_route_fused):
//   scores    = sqrt(softplus(logits))            fp32, on the packed logits (reader_moe_route.cpp)
//   selection = scores + bias                     fp32; the writer picks each row's top k from it
//   weights   = picked scores / their sum * scale (sums by matmul with the k x k ones block) -> bf16
// The op-by-op router rounds the scores to bf16 before the bias add and normalizes in bf16, so the expert choice
// can differ at near-ties and the weights in the last bits.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_packed = tt::CBIndex::c_2;
constexpr uint32_t cb_packed_bias = tt::CBIndex::c_3;
constexpr uint32_t cb_ones = tt::CBIndex::c_4;
constexpr uint32_t cb_picked = tt::CBIndex::c_6;
constexpr uint32_t cb_scores = tt::CBIndex::c_7;
constexpr uint32_t cb_selection = tt::CBIndex::c_8;
constexpr uint32_t cb_weights = tt::CBIndex::c_16;

constexpr uint32_t ONE_F32 = 0x3F800000;
constexpr uint32_t SOFTPLUS_THRESHOLD = 0x41A00000;  // 20.0, ttnn.softplus's default
constexpr uint32_t TINY_F32 = 0x0DA24260;            // 1e-30: padding rows stay 0 / tiny = 0

FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t scale = get_compile_time_arg_val(0);  // routed_scaling_factor, fp32 bits
    const uint32_t packed_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_ones, cb_picked, cb_scores);

    cb_wait_front(cb_packed, packed_tiles);
    cb_wait_front(cb_packed_bias, packed_tiles);
    for (uint32_t t = 0; t < packed_tiles; ++t) {
        tile_regs_acquire();
        load(cb_packed, t, 0);
        softplus_tile_init();
        softplus_tile(0, ONE_F32, ONE_F32, SOFTPLUS_THRESHOLD);
        sqrt_tile_init();
        sqrt_tile(0);
        load(cb_packed_bias, t, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_scores, 1);
        cb_reserve_back(cb_selection, 1);
        pack_reconfig_data_format(cb_scores);
        pack_tile(0, cb_scores);
        pack_tile(1, cb_selection);
        cb_push_back(cb_scores, 1);
        cb_push_back(cb_selection, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_packed, packed_tiles);
    cb_pop_front(cb_packed_bias, packed_tiles);

    cb_wait_front(cb_ones, 1);
    cb_wait_front(cb_picked, 1);
    tile_regs_acquire();
    matmul_init(cb_picked, cb_ones);
    matmul_tiles(cb_picked, cb_ones, 0, 0, 1);
    load(cb_picked, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(1, TINY_F32);
    div_binary_tile_init();
    div_binary_tile(0, 1, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scale);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_weights, 1);
    pack_reconfig_data_format(cb_weights);
    pack_tile(0, cb_weights);
    cb_push_back(cb_weights, 1);
    tile_regs_release();
    cb_pop_front(cb_picked, 1);
    cb_pop_front(cb_ones, 1);
}
