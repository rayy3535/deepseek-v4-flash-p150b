// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// data movement in for the fused MoE router epilogue (moe_route_compute.cpp, writer_moe_route.cpp).
//
// The router logits [rows, E] (fp32, E / 32 tiles, block row m in tile row m) and the bias row [1, E] are read, then
// repacked so that the SFPU touches only the values: block row m, expert e lands in packed row m * (E / 32) + e / 32,
// column e % 32 (a 32-row packed tile holds 32 / (E / 32) block rows). The bias row is repeated for every block row.
// Also builds the k x k ones block J the compute kernel's row sums multiply with.

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
}  // namespace

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t expert_tiles = get_compile_time_arg_val(0);  // E / 32
    constexpr uint32_t top_k = get_compile_time_arg_val(1);
    constexpr auto logits_args = TensorAccessorArgs<2>();
    constexpr auto bias_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();

    const auto logits = TensorAccessor(logits_args, logits_addr);
    const auto bias = TensorAccessor(bias_args, bias_addr);

    Noc noc;
    CircularBuffer raw_cb(0), raw_bias_cb(1), packed_cb(2), packed_bias_cb(3), ones_cb(4);
    const uint32_t page = get_local_cb_interface(0).fifo_page_size;
    const uint32_t packed_tiles = (num_rows * expert_tiles + 31) / 32;

    // The raw tiles are staging only (never pushed): read into the reserved space, repacked below.
    raw_cb.reserve_back(expert_tiles);
    raw_bias_cb.reserve_back(expert_tiles);
    for (uint32_t t = 0; t < expert_tiles; ++t) {
        noc.async_read(logits, CoreLocalMem<uint32_t>(raw_cb.get_write_ptr() + t * page), page, {.page_id = t}, {});
        noc.async_read(bias, CoreLocalMem<uint32_t>(raw_bias_cb.get_write_ptr() + t * page), page, {.page_id = t}, {});
    }

    ones_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_cb.get_write_ptr());
    for (uint32_t w = 0; w < TILE_WORDS; ++w) {
        ones[w] = 0;
    }
    for (uint32_t i = 0; i < top_k; ++i) {
        for (uint32_t j = 0; j < top_k; ++j) {
            ones[at(i, j)] = ONE_F32;
        }
    }
    ones_cb.push_back(1);

    packed_cb.reserve_back(packed_tiles);
    packed_bias_cb.reserve_back(packed_tiles);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(packed_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* out_bias =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(packed_bias_cb.get_write_ptr());
    for (uint32_t w = 0; w < packed_tiles * TILE_WORDS; ++w) {
        out[w] = 0;
        out_bias[w] = 0;
    }
    noc.async_read_barrier();
    for (uint32_t m = 0; m < num_rows; ++m) {
        for (uint32_t t = 0; t < expert_tiles; ++t) {
            const uint32_t row = m * expert_tiles + t;
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_cb.get_write_ptr() + t * page);
            volatile tt_l1_ptr uint32_t* src_bias =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_bias_cb.get_write_ptr() + t * page);
            const uint32_t base = (row >> 5) * TILE_WORDS;
            for (uint32_t c = 0; c < 32; ++c) {
                out[base + at(row & 31, c)] = src[at(m, c)];
                out_bias[base + at(row & 31, c)] = src_bias[at(0, c)];
            }
        }
    }
    packed_cb.push_back(packed_tiles);
    packed_bias_cb.push_back(packed_tiles);
}
