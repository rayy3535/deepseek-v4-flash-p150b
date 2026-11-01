// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// the cache tile row (first tile id) and the row inside it of block position start + r, as
// paged_update_cache computes them (position wrapped by `modulo` for bounded caches, then the page table).
#pragma once
#include <cstdint>

struct RowSlot {
    uint32_t tile_row;  // tile id of the row's first column tile
    uint32_t row;       // row inside the tile
};

template <uint32_t block_size, uint32_t num_heads, uint32_t width_tiles, uint32_t modulo>
FORCE_INLINE RowSlot row_slot(uint32_t position, volatile tt_l1_ptr uint32_t* page_table) {
    const uint32_t p = modulo ? position % modulo : position;
    const uint32_t physical = page_table[p / block_size];
    constexpr uint32_t block_tile_rows = block_size / 32;
    return {physical * num_heads * block_tile_rows * width_tiles + ((p % block_size) / 32) * width_tiles, p % 32};
}
