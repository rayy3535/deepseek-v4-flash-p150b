// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// mHC A2: data movement for the fused mHC input-map epilogue (hc_pre_compute.cpp).
//
// Builds the constant tiles in L1 (pre mask: row 0 cols < n; post mask: col 0 rows < n; J: the n x n ones block),
// reads the split-K partial tiles of raw = normed @ fn and the bias row (broadcast here to the block's rows), then
// waits for the compute kernel's raw tile and scatters each of this core's rows m (block rows are split over cores,
// each core sums the partials itself) into three zero-padded fp32 tiles:
//   pre_in   (0, j)  = raw[m, j]               j < n
//   post_in  (i, 0)  = raw[m, n + i]           i < n   (the [n, 1] column _hc_post broadcasts over H)
//   comb_in  (i, j)  = raw[m, 2n + n * i + j]  i, j < n
// Moving values between tile positions is a byte copy here, so the epilogue stays exact up to the SFPU math.
// from_map (mHC A3): the inputs are instead the split mHC map's per-core partials (kernels/hc_map/), this core's
// block row only; once the compute kernel has summed them over cores, this kernel gathers lane l's row of partial
// tile l into Q (row l) and the diagonal of the Gram tile into D (row 0), from which the compute kernel builds raw
// (row 0) = sum_l Q[l, :] / rms + bias.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t ONE_F32 = 0x3F800000;

FORCE_INLINE uint32_t at(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* zeroed(CircularBuffer& cb) {
    cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    for (uint32_t w = 0; w < TILE_WORDS; ++w) {
        t[w] = 0;
    }
    return t;
}
}  // namespace

void kernel_main() {
    const uint32_t partial_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_addr = get_arg_val<uint32_t>(1);
    const uint32_t first_row = get_arg_val<uint32_t>(2);  // this core's block rows: first_row .. end_row - 1
    const uint32_t end_row = first_row + get_arg_val<uint32_t>(3);
    const uint32_t total_rows = get_arg_val<uint32_t>(4);

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t split = get_compile_time_arg_val(1);  // partial tiles this core reads
    constexpr uint32_t from_map = get_compile_time_arg_val(2);
    constexpr uint32_t map_cores = get_compile_time_arg_val(3);
    constexpr uint32_t width = n * (n + 2);
    constexpr auto partial_args = TensorAccessorArgs<4>();
    constexpr auto base_args = TensorAccessorArgs<partial_args.next_compile_time_args_offset()>();

    const auto partial = TensorAccessor(partial_args, partial_addr);
    const auto base = TensorAccessor(base_args, base_addr);

    Noc noc;
    CircularBuffer partial_cb(0), base_cb(1), raw_cb(2);
    CircularBuffer pre_cb(3), post_cb(4), comb_cb(5);
    CircularBuffer mask_row_cb(6), mask_col_cb(7), ones_cb(8);
    const uint32_t page = get_local_cb_interface(0).fifo_page_size;

    partial_cb.reserve_back(split);
    for (uint32_t t = 0; t < split; ++t) {
        // map partials: core c's tiles for block row m start at page (c * total_rows + m) * (n + 1)
        const uint32_t page_id = from_map ? ((t / (n + 1)) * total_rows + first_row) * (n + 1) + t % (n + 1) : t;
        noc.async_read(
            partial, CoreLocalMem<uint32_t>(partial_cb.get_write_ptr() + t * page), page, {.page_id = page_id}, {});
    }
    base_cb.reserve_back(1);
    noc.async_read(base, CoreLocalMem<uint32_t>(base_cb.get_write_ptr()), page, {.page_id = 0}, {});

    volatile tt_l1_ptr uint32_t* mask_row = zeroed(mask_row_cb);
    volatile tt_l1_ptr uint32_t* mask_col = zeroed(mask_col_cb);
    volatile tt_l1_ptr uint32_t* ones = zeroed(ones_cb);
    for (uint32_t i = 0; i < n; ++i) {
        mask_row[at(0, i)] = ONE_F32;
        mask_col[at(i, 0)] = ONE_F32;
        for (uint32_t j = 0; j < n; ++j) {
            ones[at(i, j)] = ONE_F32;
        }
    }
    mask_row_cb.push_back(1);
    mask_col_cb.push_back(1);
    ones_cb.push_back(1);

    noc.async_read_barrier();
    partial_cb.push_back(split);
    volatile tt_l1_ptr uint32_t* bias = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_cb.get_write_ptr());
    // Broadcast the bias to the tile rows the compute kernel reads: row 0 only for map partials (raw holds this
    // core's row in row 0), block rows 0 .. end_row - 1 (<= 32) otherwise. end_row is an absolute block row, so
    // looping to it for a prefill block wrote past this 32-row tile into the next circular buffers.
    const uint32_t bias_rows = from_map ? 1 : end_row;
    for (uint32_t m = 1; m < bias_rows; ++m) {
        for (uint32_t c = 0; c < width; ++c) {
            bias[at(m, c)] = bias[at(0, c)];
        }
    }
    base_cb.push_back(1);

    if constexpr (from_map) {
        CircularBuffer sums_cb(12), q_cb(13), d_cb(14), full_ones_cb(11), jsum_cb(15);
        volatile tt_l1_ptr uint32_t* full_ones = zeroed(full_ones_cb);
        for (uint32_t w = 0; w < TILE_WORDS; ++w) {
            full_ones[w] = ONE_F32;
        }
        full_ones_cb.push_back(1);
        volatile tt_l1_ptr uint32_t* jsum = zeroed(jsum_cb);
        for (uint32_t l = 0; l < n; ++l) {
            jsum[at(0, l)] = ONE_F32;
        }
        jsum_cb.push_back(1);
        sums_cb.wait_front(n + 1);
        volatile tt_l1_ptr uint32_t* sums = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sums_cb.get_read_ptr());
        volatile tt_l1_ptr uint32_t* q = zeroed(q_cb);
        for (uint32_t l = 0; l < n; ++l) {
            for (uint32_t c = 0; c < width; ++c) {
                q[at(l, c)] = sums[l * TILE_WORDS + at(l, c)];
            }
        }
        q_cb.push_back(1);
        volatile tt_l1_ptr uint32_t* d = zeroed(d_cb);
        for (uint32_t l = 0; l < n; ++l) {
            d[at(0, l)] = sums[n * TILE_WORDS + at(l, l)];
        }
        d_cb.push_back(1);
        sums_cb.pop_front(n + 1);
    }

    raw_cb.wait_front(1);
    volatile tt_l1_ptr uint32_t* raw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_cb.get_read_ptr());
    for (uint32_t row = first_row; row < end_row; ++row) {
        const uint32_t m = from_map ? 0 : row;  // from_map: raw holds this core's row in row 0
        volatile tt_l1_ptr uint32_t* pre = zeroed(pre_cb);
        for (uint32_t j = 0; j < n; ++j) {
            pre[at(0, j)] = raw[at(m, j)];
        }
        pre_cb.push_back(1);
        volatile tt_l1_ptr uint32_t* post = zeroed(post_cb);
        for (uint32_t i = 0; i < n; ++i) {
            post[at(i, 0)] = raw[at(m, n + i)];
        }
        post_cb.push_back(1);
        volatile tt_l1_ptr uint32_t* comb = zeroed(comb_cb);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                comb[at(i, j)] = raw[at(m, 2 * n + n * i + j)];
            }
        }
        comb_cb.push_back(1);
    }
    raw_cb.pop_front(1);
}
