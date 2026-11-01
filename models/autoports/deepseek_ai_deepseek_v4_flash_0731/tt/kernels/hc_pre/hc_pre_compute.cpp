// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// mHC A2: the whole mHC input-map epilogue of a decode / verify block in one program, from the split-K
// partials of raw = normed @ fn to the tiles _hc_pre returns (OptimizedDecoder._hc_pre_epilogue_fused):
//   raw   = sum_t partial[t] + bias                                   (SFPU fp32 adds, inputs unpacked to DST)
//   pre   = (sigmoid(raw[:n]) + eps) masked to row 0 cols < n         -> fp32 [1, n] tile for the collapse matmul
//   post  = 2 sigmoid(raw[n:2n]) masked to col 0 rows < n             -> the [n, 1] column _hc_post broadcasts
//   comb  = softmax(raw[2n:].view(n, n)) + eps, then Sinkhorn: one column normalization and (iters - 1) row +
//           column pairs, value / (sums + eps), sums by matmul with the n x n ones block J
// The reader moves the raw values into these tile positions. Unlike the op-by-op path and the kernel,
// intermediates stay fp32 (the Sinkhorn sums and division share one DST section; only the matmul operands pass
// through the tf32 source registers), so the result is close to but not bit-identical with either.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t cb_partial = tt::CBIndex::c_0;
constexpr uint32_t cb_base = tt::CBIndex::c_1;
constexpr uint32_t cb_raw = tt::CBIndex::c_2;
constexpr uint32_t cb_pre_in = tt::CBIndex::c_3;
constexpr uint32_t cb_post_in = tt::CBIndex::c_4;
constexpr uint32_t cb_comb_in = tt::CBIndex::c_5;
constexpr uint32_t cb_mask_row = tt::CBIndex::c_6;
constexpr uint32_t cb_mask_col = tt::CBIndex::c_7;
constexpr uint32_t cb_ones = tt::CBIndex::c_8;
constexpr uint32_t cb_ping = tt::CBIndex::c_9;
constexpr uint32_t cb_pong = tt::CBIndex::c_10;
constexpr uint32_t cb_pre_out = tt::CBIndex::c_16;
constexpr uint32_t cb_post_out = tt::CBIndex::c_17;
constexpr uint32_t cb_comb_out = tt::CBIndex::c_18;
// from_map (mHC A3)
constexpr uint32_t cb_full_ones = tt::CBIndex::c_11;
constexpr uint32_t cb_sums = tt::CBIndex::c_12;
constexpr uint32_t cb_q = tt::CBIndex::c_13;
constexpr uint32_t cb_d = tt::CBIndex::c_14;
constexpr uint32_t cb_jsum = tt::CBIndex::c_15;

constexpr uint32_t TINY_F32 = 0x0DA24260;  // 1e-30: keeps the softmax's all-zero padding rows at 0 / tiny = 0
constexpr uint32_t TWO_F32 = 0x40000000;

FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

FORCE_INLINE void pack_to(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(0, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}

FORCE_INLINE void mask(uint32_t cb_mask) {
    load(cb_mask, 0, 2);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
}

// value (front of cb_value) / (sums + div_eps) [+ eps, masked to the n x n block] -> cb_next; pops cb_value.
FORCE_INLINE void normalize(
    uint32_t cb_value, uint32_t cb_next, bool rows, uint32_t div_eps, bool add_eps, uint32_t eps) {
    cb_wait_front(cb_value, 1);
    tile_regs_acquire();
    if (rows) {
        matmul_init(cb_value, cb_ones);
        matmul_tiles(cb_value, cb_ones, 0, 0, 1);
    } else {
        matmul_init(cb_ones, cb_value);
        matmul_tiles(cb_ones, cb_value, 0, 0, 1);
    }
    load(cb_value, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(1, div_eps);
    div_binary_tile_init();
    div_binary_tile(0, 1, 0);
    if (add_eps) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps);
        mask(cb_ones);
    }
    pack_to(cb_next);
    cb_pop_front(cb_value, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t iters = get_compile_time_arg_val(0);
    constexpr uint32_t eps = get_compile_time_arg_val(1);
    constexpr uint32_t split = get_compile_time_arg_val(2);
    constexpr uint32_t from_map = get_compile_time_arg_val(3);
    constexpr uint32_t map_cores = get_compile_time_arg_val(4);
    constexpr uint32_t inv_width = get_compile_time_arg_val(5);  // 1 / (n * H), fp32 bits (from_map)
    constexpr uint32_t norm_eps = get_compile_time_arg_val(6);   // RMS norm epsilon, fp32 bits (from_map)
    constexpr uint32_t steps = 2 * iters - 1;
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_ones, cb_ping, cb_raw);

    cb_wait_front(cb_partial, split);
    cb_wait_front(cb_base, 1);
    if constexpr (from_map) {
        // the map cores' partial tiles summed per tile kind d (n lane tiles, then the Gram tile) -> cb_sums
        constexpr uint32_t kinds = split / (map_cores ? map_cores : 1);  // map_cores = 0 never reaches here
        for (uint32_t d = 0; d < kinds; ++d) {
            tile_regs_acquire();
            load(cb_partial, d, 0);
            for (uint32_t c = 1; c < map_cores; ++c) {
                load(cb_partial, c * kinds + d, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            pack_to(cb_sums);
        }
        cb_pop_front(cb_partial, split);
        // raw (row 0) = (Jsum @ Q) / sqrt((D @ 1) / (n H) + eps) + bias
        cb_wait_front(cb_q, 1);
        cb_wait_front(cb_d, 1);
        cb_wait_front(cb_jsum, 1);
        cb_wait_front(cb_full_ones, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_q, cb_jsum);
        matmul_init(cb_jsum, cb_q);
        matmul_tiles(cb_jsum, cb_q, 0, 0, 0);
        reconfig_data_format(cb_full_ones, cb_d);
        matmul_init(cb_d, cb_full_ones);
        matmul_tiles(cb_d, cb_full_ones, 0, 0, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(1, inv_width);
        add_unary_tile(1, norm_eps);
        sqrt_tile_init();
        sqrt_tile(1);
        div_binary_tile_init();
        div_binary_tile(0, 1, 0);
        load(cb_base, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        pack_to(cb_raw);
        cb_pop_front(cb_q, 1);
        cb_pop_front(cb_d, 1);
        cb_pop_front(cb_jsum, 1);
        cb_pop_front(cb_full_ones, 1);
    } else {
        tile_regs_acquire();
        load(cb_base, 0, 0);
        for (uint32_t t = 0; t < split; ++t) {
            load(cb_partial, t, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        pack_to(cb_raw);
        cb_pop_front(cb_partial, split);
    }
    cb_pop_front(cb_base, 1);

    cb_wait_front(cb_mask_row, 1);
    cb_wait_front(cb_mask_col, 1);
    cb_wait_front(cb_ones, 1);
    for (uint32_t m = 0; m < num_rows; ++m) {
        cb_wait_front(cb_pre_in, 1);
        tile_regs_acquire();
        load(cb_pre_in, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps);
        mask(cb_mask_row);
        pack_to(cb_pre_out);
        cb_pop_front(cb_pre_in, 1);

        cb_wait_front(cb_post_in, 1);
        tile_regs_acquire();
        load(cb_post_in, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, TWO_F32);
        mask(cb_mask_col);
        pack_to(cb_post_out);
        cb_pop_front(cb_post_in, 1);

        cb_wait_front(cb_comb_in, 1);
        tile_regs_acquire();
        load(cb_comb_in, 0, 0);
        exp_tile_init();
        exp_tile(0);
        mask(cb_ones);
        pack_to(cb_ping);
        cb_pop_front(cb_comb_in, 1);

        normalize(cb_ping, cb_pong, true, TINY_F32, true, eps);  // softmax over each row, + eps
        uint32_t current = cb_pong;
        for (uint32_t step = 0; step < steps; ++step) {
            const bool rows = step % 2 == 1;  // column normalization first, then row / column pairs
            const uint32_t next = step + 1 == steps ? cb_comb_out : (current == cb_ping ? cb_pong : cb_ping);
            normalize(current, next, rows, eps, false, eps);
            current = next;
        }
    }
    cb_pop_front(cb_mask_row, 1);
    cb_pop_front(cb_mask_col, 1);
    cb_pop_front(cb_ones, 1);
}
