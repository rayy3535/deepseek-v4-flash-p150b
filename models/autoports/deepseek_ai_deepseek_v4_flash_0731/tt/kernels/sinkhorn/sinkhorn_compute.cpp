// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// the whole Sinkhorn of this core's mHC combination tiles (one per token) in one program.
//
// It runs the same per-step tile operations as the op-by-op version (OptimizedDecoder._sinkhorn_by_matmul):
//   sums  = value @ J (row sums) or J @ value (column sums), J the n x n ones block   (FPU matmul_tiles)
//   value = value / (sums + eps)                                                     (SFPU add_unary + div_binary)
// with every intermediate packed to the CB data format (bf16 in the deployed policy), as the separate ops store
// theirs. One column normalization, then (iters - 1) row + column pairs. The op-by-op version runs the division
// with a 16-bit DST, this kernel with the matmul's fp32 DST, so the result is not bit-identical.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_ones = tt::CBIndex::c_1;
constexpr uint32_t cb_sums = tt::CBIndex::c_2;
constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
constexpr uint32_t cb_ping = tt::CBIndex::c_4;
constexpr uint32_t cb_pong = tt::CBIndex::c_5;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

FORCE_INLINE void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_tile(0, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}

// value (front of cb_value) / (sums + eps) -> cb_next; pops cb_value.
FORCE_INLINE void normalize(uint32_t cb_value, uint32_t cb_next, bool rows, uint32_t eps_bits) {
    cb_wait_front(cb_value, 1);
    if (rows) {
        matmul_init(cb_value, cb_ones);
    } else {
        matmul_init(cb_ones, cb_value);
    }
    tile_regs_acquire();
    if (rows) {
        matmul_tiles(cb_value, cb_ones, 0, 0, 0);
    } else {
        matmul_tiles(cb_ones, cb_value, 0, 0, 0);
    }
    pack_one(cb_sums);

    cb_wait_front(cb_sums, 1);
    copy_init(cb_sums);
    tile_regs_acquire();
    copy_tile(cb_sums, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    pack_one(cb_divisor);
    cb_pop_front(cb_sums, 1);

    cb_wait_front(cb_divisor, 1);
    tile_regs_acquire();
    copy_init(cb_value);
    copy_tile(cb_value, 0, 0);
    copy_init(cb_divisor);
    copy_tile(cb_divisor, 0, 1);
    div_binary_tile_init();
    div_binary_tile(0, 1, 0);
    pack_one(cb_next);
    cb_pop_front(cb_divisor, 1);
    cb_pop_front(cb_value, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t iters = get_compile_time_arg_val(0);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(1);
    constexpr uint32_t steps = 2 * iters - 1;
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_ones, cb_in, cb_sums);
    cb_wait_front(cb_ones, 1);

    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        uint32_t current = cb_in;
        for (uint32_t step = 0; step < steps; ++step) {
            const bool rows = step % 2 == 1;  // column normalization first, then row / column pairs
            const uint32_t next = step + 1 == steps ? cb_out : (step % 2 == 0 ? cb_ping : cb_pong);
            normalize(current, next, rows, eps_bits);
            current = next;
        }
    }
    cb_pop_front(cb_ones, 1);
}
