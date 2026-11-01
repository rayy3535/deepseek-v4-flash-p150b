// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// top-k and data movement out for the fused MoE router epilogue.
//
// For every block row the k largest selection values (scores + bias, fp32) are found by comparing order-preserving
// integer keys of their bit patterns (largest first; equal keys keep the lower expert index, as a stable top-k does).
// The picked experts go to the uint16 [rows, k] output tile; the matching scores (bit copies) go, at the same tile
// positions, to the tile the compute kernel normalizes into the bf16 routing weights, which are then written too.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t MAX_K = 16;

FORCE_INLINE uint32_t at(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

// fp32 bits -> unsigned key with the same order as the float values
FORCE_INLINE uint32_t key(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }
}  // namespace

void kernel_main() {
    const uint32_t selected_addr = get_arg_val<uint32_t>(0);
    const uint32_t weights_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t expert_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t top_k = get_compile_time_arg_val(1);
    constexpr auto selected_args = TensorAccessorArgs<2>();
    constexpr auto weights_args = TensorAccessorArgs<selected_args.next_compile_time_args_offset()>();
    static_assert(top_k <= MAX_K, "top_k too large");

    const auto selected_out = TensorAccessor(selected_args, selected_addr);
    const auto weights_out = TensorAccessor(weights_args, weights_addr);
    const uint32_t packed_tiles = (num_rows * expert_tiles + 31) / 32;

    Noc noc;
    CircularBuffer picked_cb(6), scores_cb(7), selection_cb(8), selected_cb(17);
    DataflowBuffer weights_dfb(16);

    selected_cb.reserve_back(1);
    volatile tt_l1_ptr uint16_t* selected = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(selected_cb.get_write_ptr());
    picked_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* picked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(picked_cb.get_write_ptr());
    for (uint32_t w = 0; w < TILE_WORDS; ++w) {
        picked[w] = 0;
        selected[2 * w] = 0;
        selected[2 * w + 1] = 0;
    }

    scores_cb.wait_front(packed_tiles);
    selection_cb.wait_front(packed_tiles);
    volatile tt_l1_ptr uint32_t* scores = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scores_cb.get_read_ptr());
    volatile tt_l1_ptr uint32_t* selection =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(selection_cb.get_read_ptr());
    for (uint32_t m = 0; m < num_rows; ++m) {
        uint32_t best_key[MAX_K];
        uint32_t best_pos[MAX_K];
        uint32_t best_expert[MAX_K];
        uint32_t found = 0;
        for (uint32_t t = 0; t < expert_tiles; ++t) {
            const uint32_t row = m * expert_tiles + t;
            const uint32_t base = (row >> 5) * TILE_WORDS;
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t pos = base + at(row & 31, c);
                const uint32_t k = key(selection[pos]);
                if (found == top_k && k <= best_key[top_k - 1]) {
                    continue;
                }
                uint32_t i = found < top_k ? found++ : top_k - 1;
                while (i > 0 && best_key[i - 1] < k) {
                    best_key[i] = best_key[i - 1];
                    best_pos[i] = best_pos[i - 1];
                    best_expert[i] = best_expert[i - 1];
                    --i;
                }
                best_key[i] = k;
                best_pos[i] = pos;
                best_expert[i] = t * 32 + c;
            }
        }
        for (uint32_t j = 0; j < top_k; ++j) {
            selected[at(m, j)] = static_cast<uint16_t>(best_expert[j]);
            picked[at(m, j)] = scores[best_pos[j]];
        }
    }
    scores_cb.pop_front(packed_tiles);
    selection_cb.pop_front(packed_tiles);
    picked_cb.push_back(1);

    const uint32_t selected_page = get_local_cb_interface(17).fifo_page_size;
    noc.async_write(
        CoreLocalMem<uint32_t>(selected_cb.get_write_ptr()), selected_out, selected_page, {}, {.page_id = 0});
    weights_dfb.wait_front(1);
    noc.async_write(weights_dfb, weights_out, get_local_cb_interface(16).fifo_page_size, {}, {.page_id = 0});
    noc.async_write_barrier();
    weights_dfb.pop_front(1);
}
